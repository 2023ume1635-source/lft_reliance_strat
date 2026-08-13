"""
================================================================================
PHASE 5 — REGIME-ADAPTIVE SIGNAL GENERATION + KELLY SIZING (v3 fixed)
================================================================================

FIXES over v2:
  - Lowered tau_take grid so aggressive signals actually fire
  - Added signal-count stability penalty in sweep objective
  - Added long/short balance penalty (max 2.5:1 ratio)
  - Added per-day signal cap (MAX_SIGNALS_PER_DAY = 300)
  - Added per-day balance enforcement (if ratio > 2.5:1, trim the dominant side)
  - take_mult range tightened to 2.0-3.5x (was 4-6x, too aggressive)
  - Unified COOLDOWN_S = 15, MIN_HOLD_S = 3
  - No transaction costs (per user request)
"""

from __future__ import annotations
import json, time, warnings
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parent
PRED_DIR  = ROOT / "artifacts" / "preds"
FEAT_DIR  = ROOT / "output" / "features"
LABEL_DIR = ROOT / "output" / "labels"
CLEAN_DIR = ROOT / "output" / "clean"
SPLIT_F   = ROOT / "output" / "splits.json"
SIG_DIR   = ROOT / "output" / "signals"
REPORT_DIR = ROOT / "reports"
for d in (SIG_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

TICK = 0.05
INITIAL_CAPITAL    = 10_00_000.0
MAX_DAILY_LOSS     = 0.02 * INITIAL_CAPITAL   # Rs 20,000
COOLDOWN_S         = 15
HARD_GATE_REGIMES  = {4, 5}
MIN_HOLD_S         = 3

# FIX: Per-day signal cap and balance enforcement
MAX_SIGNALS_PER_DAY = 300
MAX_LONG_SHORT_RATIO = 2.5

REGIME_HPARAMS = {
    0: dict(H=20, passive_discount=0.85, take_mult=2.5, sprd_cap_t=4),
    1: dict(H=25, passive_discount=0.95, take_mult=2.8, sprd_cap_t=4),
    2: dict(H=30, passive_discount=0.90, take_mult=3.0, sprd_cap_t=5),
    3: dict(H=45, passive_discount=1.10, take_mult=3.5, sprd_cap_t=8),
    4: dict(H=40, passive_discount=1e9, take_mult=1e9, sprd_cap_t=0),
    5: dict(H=0,  passive_discount=1e9, take_mult=1e9, sprd_cap_t=0),
}

# FIX: Lower tau grids so take signals can actually fire
TAU_GRID_PASSIVE = [0.003, 0.005, 0.008, 0.012, 0.018, 0.025, 0.035]
TAU_GRID_TAKE    = [0.008, 0.012, 0.018, 0.025, 0.035, 0.050]
EDGE_FLOOR_GRID  = [0.00, 0.03, 0.06, 0.10, 0.15, 0.20]
N_MIN_PER_REG    = 200
N_MAX_PER_REG    = 5000
P_FILL_MIN_PASSIVE = 0.15
PER_TRADE_BUDGET = 500.0
MAX_PCT_DEPTH    = 0.40


def load_split():
    with open(SPLIT_F) as f: return json.load(f)

def _ymd(s: str) -> str: return s.replace("-", "")

def load_pred_day(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p)

def load_clean_day(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p, columns=["b1_p", "a1_p", "b1_q", "a1_q"])

def load_is_stale(d: str, idx: pd.Index) -> np.ndarray:
    for folder in (LABEL_DIR, FEAT_DIR):
        p = folder / f"{d}.parquet"
        if p.exists():
            try:
                add = pd.read_parquet(p, columns=["is_stale"])
                return add["is_stale"].reindex(idx).fillna(1).astype(np.int8).values
            except Exception:
                continue
    return np.zeros(len(idx), dtype=np.int8)


# ---------------------------------------------------------------------------
def compute_thresholds(train_preds: Dict[str, pd.DataFrame]) -> Tuple[Dict[int, dict], pd.DataFrame]:
    print("\n--- Regime-conditional tau sweep on TRAIN OOF (v3) ---")
    frames = []
    for d, df in train_preds.items():
        x = df.copy(); x["_file"] = d; frames.append(x)
    tr = pd.concat(frames)
    tr = tr[tr["y_valid"].astype(bool)].copy()
    y       = tr["y_take_dir"].values.astype(np.int8)
    pl      = tr["p_long_m1"].values.astype(np.float64)
    ps      = tr["p_short_m1"].values.astype(np.float64)
    eraw    = tr["edge_pred"].values.astype(np.float64)
    regimes = tr["mkt_state"].values.astype(np.int8)
    files   = tr["_file"].values
    cost_floor_proxy = 2 * TICK
    tau_table: Dict[int, dict] = {}
    sweep: List[dict] = []

    n_train_days = len(train_preds)

    for r in range(6):
        hp = REGIME_HPARAMS[r]
        if r in HARD_GATE_REGIMES:
            tau_table[r] = dict(tau_passive=9.0, tau_take=9.0, edge_floor=9.0, **hp, n=0, prec=0.0)
            sweep.append(dict(regime=r, tau_p=9.0, tau_t=9.0, edge_floor=9.0, n=0, prec=0.0, j=-1e9))
            continue
        m = regimes == r
        if m.sum() < 200:
            tau_table[r] = dict(tau_passive=0.008 * hp["passive_discount"],
                                tau_take=0.008 * hp["take_mult"],
                                edge_floor=0.06, **hp, n=0, prec=0.5)
            continue
        yr = y[m]; plr = pl[m]; psr = ps[m]; er = np.abs(eraw[m])
        files_r = files[m]
        best, best_j = None, -1e18

        # FIX: Separate grids for passive and take
        for tau_p in TAU_GRID_PASSIVE:
            for tau_t in TAU_GRID_TAKE:
                # Ensure take > passive
                if tau_t <= tau_p * hp["take_mult"] * 0.5:
                    continue
                for ef in EDGE_FLOOR_GRID:
                    tp = tau_p * hp["passive_discount"]
                    tt = tau_t  # absolute take threshold

                    lp = (plr - psr >  tp) & (er > ef)
                    sp_sig = (psr - plr >  tp) & (er > ef)
                    lt = (plr - psr >  tt) & (er > ef + cost_floor_proxy)
                    st = (psr - plr >  tt) & (er > ef + cost_floor_proxy)
                    fired = lp | sp_sig | lt | st
                    n = int(fired.sum())
                    if n < 20: continue

                    plf = plr[fired]; psf = psr[fired]; yf = yr[fired]
                    pred = np.where(plf > psf, 1, -1)
                    md = yf != 0
                    if md.sum() < 10: continue
                    prec = float((pred[md] == yf[md]).mean())
                    me = float(er[fired].mean())

                    # FIX: Count signals per day for stability
                    files_fired = files_r[fired]
                    unique_days, day_counts = np.unique(files_fired, return_counts=True)
                    n_days_with_sigs = len(unique_days)
                    mean_per_day = n / max(n_days_with_sigs, 1)
                    std_per_day = float(day_counts.std()) if len(day_counts) > 1 else 0.0

                    # FIX: Long/short balance
                    n_long = int((pred == 1).sum())
                    n_short = int((pred == -1).sum())
                    ratio = max(n_long, n_short) / max(min(n_long, n_short), 1)

                    # Objective: precision-weighted edge, with stability & balance penalties
                    j = (prec - 0.50) * me * 100.0 \
                        - max(0.0, N_MIN_PER_REG - n) / 300.0 \
                        - max(0.0, n - N_MAX_PER_REG) / 2000.0 \
                        - 5.0 * max(0.0, 0.52 - prec) \
                        - 2.0 * max(0.0, ratio - MAX_LONG_SHORT_RATIO) \
                        - 0.5 * std_per_day / max(mean_per_day, 1)  # CV penalty

                    sweep.append(dict(regime=r, tau_p=tau_p, tau_t=tau_t, edge_floor=ef,
                                      n=n, prec=prec, j=j, mean_per_day=mean_per_day,
                                      ratio=ratio))
                    if j > best_j:
                        best_j = j
                        best = dict(tau_p=tau_p, tau_t=tau_t, edge_floor=ef,
                                    prec=prec, n=n, j=j, mean_per_day=mean_per_day,
                                    ratio=ratio)

        if best is None:
            best = dict(tau_p=0.008, tau_t=0.025, edge_floor=0.06,
                        prec=0.52, n=0, j=-1e9, mean_per_day=0, ratio=1.0)

        tau_table[r] = dict(
            tau_passive=float(best["tau_p"] * hp["passive_discount"]),
            tau_take=float(best["tau_t"]),
            edge_floor=float(best["edge_floor"]), **hp,
            n=int(best["n"]), prec=float(best["prec"]))
        print(f"  regime {r}: tau_p={tau_table[r]['tau_passive']:.4f}  "
              f"tau_t={tau_table[r]['tau_take']:.4f}  "
              f"ef={tau_table[r]['edge_floor']:.2f}  "
              f"prec={tau_table[r]['prec']:.3f}  n={tau_table[r]['n']}  "
              f"~{best.get('mean_per_day', 0):.0f}/day  ratio={best.get('ratio', 1):.2f}")
    return tau_table, pd.DataFrame(sweep)


# ---------------------------------------------------------------------------
def build_signals_for_day(pred_df: pd.DataFrame, clean_df: pd.DataFrame,
                          is_stale: np.ndarray,
                          tau_table: Dict[int, dict], date: str) -> pd.DataFrame:
    df = pred_df.join(clean_df, how="left").copy()
    df["sprd"]     = (df["a1_p"] - df["b1_p"]).astype(np.float32)
    df["halfsprd"] = (0.5 * df["sprd"]).astype(np.float32)
    df["is_stale"] = is_stale
    pl  = df["p_long_m1"].values.astype(np.float64)
    ps  = df["p_short_m1"].values.astype(np.float64)
    raw = (pl - ps).astype(np.float64)
    em  = df["edge_pred"].values.astype(np.float64)
    reg = df["mkt_state"].values.astype(np.int8)
    fb  = df["p_buyfill"].values.astype(np.float64)
    fs  = df["p_sellfill"].values.astype(np.float64)
    R   = df["R_per_share"].values.astype(np.float64)
    sp  = df["sprd"].values.astype(np.float64)
    bq  = df["b1_q"].values.astype(np.float64)
    aq  = df["a1_q"].values.astype(np.float64)
    bp  = df["b1_p"].values.astype(np.float64)
    ap  = df["a1_p"].values.astype(np.float64)
    stale = df["is_stale"].values.astype(np.int8)
    valid = df["y_valid"].astype(bool).values
    T = len(df)
    sig    = np.zeros(T, dtype=np.int8)
    size   = np.zeros(T, dtype=np.int32)
    entry  = np.zeros(T, dtype=np.float32)
    stop   = np.zeros(T, dtype=np.float32)
    tgt    = np.zeros(T, dtype=np.float32)
    horizon = np.zeros(T, dtype=np.int16)
    reason = np.array([""] * T, dtype=object)
    risk   = np.ones(T, dtype=bool)

    for i in range(T):
        r = int(reg[i])
        tau = tau_table.get(r)
        if tau is None or r in HARD_GATE_REGIMES:
            risk[i] = False; reason[i] = f"gate:regime{r}"; continue
        if stale[i]:
            risk[i] = False; reason[i] = "gate:stale"; continue
        if not (np.isfinite(sp[i]) and sp[i] > 0):
            risk[i] = False; reason[i] = "gate:bad_spread"; continue
        if not valid[i]:
            risk[i] = False; reason[i] = "gate:invalid"; continue
        s = raw[i]
        sprd_cap = tau["sprd_cap_t"] * TICK
        cost = sp[i] + TICK

        # Take (aggressive) conditions
        long_t_cond = (s >  tau["tau_take"]) and (abs(em[i]) > cost + tau["edge_floor"])
        shrt_t_cond = (s < -tau["tau_take"]) and (abs(em[i]) > cost + tau["edge_floor"])
        # Passive conditions
        long_p_cond = (s >  tau["tau_passive"]) and (abs(em[i]) > tau["edge_floor"]) \
                      and (fb[i] >= P_FILL_MIN_PASSIVE) and (sp[i] <= sprd_cap)
        shrt_p_cond = (s < -tau["tau_passive"]) and (abs(em[i]) > tau["edge_floor"]) \
                      and (fs[i] >= P_FILL_MIN_PASSIVE) and (sp[i] <= sprd_cap)

        # Priority: take > passive (take is stronger conviction)
        if long_t_cond:
            sig[i] = 2; p_entry = ap[i]; side = +1; reason[i] = "take_long"
        elif shrt_t_cond:
            sig[i] = -2; p_entry = bp[i]; side = -1; reason[i] = "take_short"
        elif long_p_cond:
            sig[i] = 1; p_entry = bp[i]; side = +1; reason[i] = "post_long"
        elif shrt_p_cond:
            sig[i] = -1; p_entry = ap[i]; side = -1; reason[i] = "post_short"
        else:
            reason[i] = "no_signal"; continue

        half_s = 0.5 * sp[i]
        R_i = max(R[i], 0.25)
        dyn = float(np.clip(1.0 + abs(em[i]) / 0.3, 0.7, 1.6))
        tgt_d = R_i * dyn
        stp_d = max(half_s + 0.4 * R_i, 0.10)
        n_shares = int(max(1, PER_TRADE_BUDGET / stp_d))
        depth_ref = bq[i] if side > 0 else aq[i]
        if np.isfinite(depth_ref) and depth_ref > 0:
            n_shares = min(n_shares, int(MAX_PCT_DEPTH * depth_ref))
        n_shares = min(n_shares, int(MAX_DAILY_LOSS / max(R_i, 0.10)))
        n_shares = int(np.clip(n_shares, 1, 5000))
        size[i] = n_shares
        entry[i] = np.float32(p_entry)
        if side > 0:
            tgt[i] = np.float32(p_entry + tgt_d)
            stop[i] = np.float32(p_entry - stp_d)
        else:
            tgt[i] = np.float32(p_entry - tgt_d)
            stop[i] = np.float32(p_entry + stp_d)
        horizon[i] = np.int16(tau["H"])

    # FIX: Per-day signal cap and balance enforcement
    fired_idx = np.where(sig != 0)[0]
    if len(fired_idx) > MAX_SIGNALS_PER_DAY:
        # Keep the strongest signals (highest |raw|)
        strengths = np.abs(raw[fired_idx])
        keep_order = np.argsort(-strengths)[:MAX_SIGNALS_PER_DAY]
        drop_idx = fired_idx[np.setdiff1d(np.arange(len(fired_idx)), keep_order)]
        sig[drop_idx] = 0
        size[drop_idx] = 0
        reason[drop_idx] = "capped"
        fired_idx = np.where(sig != 0)[0]

    # Balance enforcement: if ratio > MAX_LONG_SHORT_RATIO, trim dominant side
    if len(fired_idx) > 10:
        long_idx = fired_idx[sig[fired_idx] > 0]
        short_idx = fired_idx[sig[fired_idx] < 0]
        n_long = len(long_idx); n_short = len(short_idx)
        if n_short > 0 and n_long / n_short > MAX_LONG_SHORT_RATIO:
            # Trim weakest longs
            n_keep = int(n_short * MAX_LONG_SHORT_RATIO)
            if n_keep < n_long:
                strengths_l = np.abs(raw[long_idx])
                keep_l = long_idx[np.argsort(-strengths_l)[:n_keep]]
                drop_l = np.setdiff1d(long_idx, keep_l)
                sig[drop_l] = 0; size[drop_l] = 0; reason[drop_l] = "balance_trim"
        elif n_long > 0 and n_short / n_long > MAX_LONG_SHORT_RATIO:
            n_keep = int(n_long * MAX_LONG_SHORT_RATIO)
            if n_keep < n_short:
                strengths_s = np.abs(raw[short_idx])
                keep_s = short_idx[np.argsort(-strengths_s)[:n_keep]]
                drop_s = np.setdiff1d(short_idx, keep_s)
                sig[drop_s] = 0; size[drop_s] = 0; reason[drop_s] = "balance_trim"

    out = pd.DataFrame(index=df.index)
    out["trade_date"] = df["trade_date"]
    out["mkt_state"]  = reg
    out["sig"]        = sig
    out["side"]       = np.sign(sig).astype(np.int8)
    out["size_qty"]   = size
    out["entry_ref"]  = entry
    out["stop_p"]     = stop
    out["tgt_p"]      = tgt
    out["horizon_s"]  = horizon
    out["risk_ok"]    = risk
    out["reason"]     = reason
    out["p_long"]     = pl.astype(np.float32)
    out["p_short"]    = ps.astype(np.float32)
    out["raw_sig"]    = raw.astype(np.float32)
    out["edge_pred"]  = em.astype(np.float32)
    out["p_buyfill"]  = fb.astype(np.float32)
    out["p_sellfill"] = fs.astype(np.float32)
    out["b1_p"]       = bp.astype(np.float32)
    out["a1_p"]       = ap.astype(np.float32)
    out["b1_q"]       = bq.astype(np.float32)
    out["a1_q"]       = aq.astype(np.float32)
    out["sprd"]       = sp.astype(np.float32)
    out["halfsprd"]   = df["halfsprd"].values.astype(np.float32)
    out["R_per_share"] = R.astype(np.float32)
    out["y_take_dir"] = df["y_take_dir"].values.astype(np.int8)
    out["y_valid"]    = valid
    out["_split"]     = df["_split"].values if "_split" in df.columns else ""
    return out


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 78)
    print("PHASE 5 — REGIME-ADAPTIVE SIGNALS + KELLY SIZING (v3 fixed)")
    print(f"ROOT={ROOT}  PREDS={PRED_DIR}")
    print(f"Max signals/day={MAX_SIGNALS_PER_DAY} | Max L/S ratio={MAX_LONG_SHORT_RATIO}")
    print("=" * 78)
    splits = load_split()
    train_d = [_ymd(d) for d in splits["train"]]
    test_d  = [_ymd(d) for d in splits["test"]]
    valid_d = [_ymd(d) for d in splits["valid"]]
    all_d = train_d + test_d + valid_d
    preds: Dict[str, pd.DataFrame] = {}
    for d in all_d:
        p = PRED_DIR / f"{d}.parquet"
        if not p.exists(): raise FileNotFoundError(f"Missing pred parquet {p}; run phase4 first.")
        preds[d] = load_pred_day(p)
    print(f"Loaded {len(preds)} days")
    train_p = {d: preds[d] for d in train_d}
    tau_table, sweep_df = compute_thresholds(train_p)
    sweep_df.to_csv(REPORT_DIR / "phase5_tau_sweep.csv", index=False)
    print("\nBuilding per-day signals ...")
    stats = []
    for d in all_d:
        pred_df = preds[d]
        cp = CLEAN_DIR / f"{d}.parquet"
        if not cp.exists():
            print(f"  [warn] missing clean parquet {cp}; skipped"); continue
        clean_df = load_clean_day(cp)
        is_stale = load_is_stale(d, pred_df.index)
        sig_df = build_signals_for_day(pred_df, clean_df, is_stale, tau_table, d)
        sig_df.to_parquet(SIG_DIR / f"{d}.parquet")
        split = "train" if d in train_d else "test" if d in test_d else "valid"
        n_tl = int((sig_df["sig"] == 2).sum())
        n_pl = int((sig_df["sig"] == 1).sum())
        n_ps = int((sig_df["sig"] == -1).sum())
        n_ts = int((sig_df["sig"] == -2).sum())
        n = n_tl + n_pl + n_ps + n_ts
        fired = sig_df["sig"] != 0
        if fired.sum() > 5:
            yf = sig_df.loc[fired, "y_take_dir"].values
            sf = np.sign(sig_df.loc[fired, "sig"].values)
            md = yf != 0
            prec = float((sf[md] == yf[md]).mean()) if md.sum() > 5 else float("nan")
            me = float(sig_df.loc[fired, "edge_pred"].abs().mean())
            fired_sz = sig_df.loc[fired, "size_qty"]
            msz = float(fired_sz[fired_sz > 0].mean()) if (fired_sz > 0).any() else 0.0
        else:
            prec = float("nan"); me = 0.0; msz = 0.0
        stats.append(dict(date=d, split=split, n_secs=len(sig_df),
                          n_take_long=n_tl, n_post_long=n_pl, n_post_short=n_ps, n_take_short=n_ts,
                          n_signals=n, signals_per_min=round(n / (6.25 * 60), 2),
                          mean_size_qty=round(msz, 1), mean_pred_edge=round(me, 3),
                          prec_vs_label=round(prec, 3) if not np.isnan(prec) else None))
        prec_s = f"{prec:.3f}" if not np.isnan(prec) else " nan "
        print(f"  {d} [{split:5s}] signals={n:4d}  "
              f"(take-L {n_tl:3d} / post-L {n_pl:3d} / post-S {n_ps:3d} / take-S {n_ts:3d})  "
              f"prec={prec_s}  mean_size={msz:.0f}  E[edge]={me:.3f}")
    sdf = pd.DataFrame(stats); sdf.to_csv(REPORT_DIR / "phase5_signal_stats.csv", index=False)
    print("\n--- Aggregate signal rates ---")
    cnt_map = {"train": len(train_d), "test": len(test_d), "valid": len(valid_d)}
    agg = sdf.groupby("split")[["n_take_long", "n_post_long", "n_post_short", "n_take_short", "n_signals"]].sum()
    agg["signals/day"] = (agg["n_signals"] / agg.index.map(lambda s: cnt_map[s])).round(1)
    print(agg.to_string())
    cfg = dict(TICK=TICK, INITIAL_CAPITAL=INITIAL_CAPITAL, MAX_DAILY_LOSS=MAX_DAILY_LOSS,
               COOLDOWN_S=COOLDOWN_S, MIN_HOLD_S=MIN_HOLD_S,
               PER_TRADE_BUDGET=PER_TRADE_BUDGET,
               P_FILL_MIN_PASSIVE=P_FILL_MIN_PASSIVE,
               MAX_PCT_DEPTH=MAX_PCT_DEPTH,
               MAX_SIGNALS_PER_DAY=MAX_SIGNALS_PER_DAY,
               MAX_LONG_SHORT_RATIO=MAX_LONG_SHORT_RATIO,
               HARD_GATE_REGIMES=sorted(HARD_GATE_REGIMES),
               regime_tau={str(k): v for k, v in tau_table.items()},
               REGIME_HPARAMS={str(k): v for k, v in REGIME_HPARAMS.items()},
               TAU_GRID_PASSIVE=TAU_GRID_PASSIVE, TAU_GRID_TAKE=TAU_GRID_TAKE,
               EDGE_FLOOR_GRID=EDGE_FLOOR_GRID,
               N_MIN_PER_REG=N_MIN_PER_REG, N_MAX_PER_REG=N_MAX_PER_REG)
    with open(REPORT_DIR / "phase5_config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    print(f"\nSaved: {SIG_DIR}/<yyyymmdd>.parquet x {len(all_d)}")
    print(f"       {REPORT_DIR / 'phase5_signal_stats.csv'}")
    print(f"       {REPORT_DIR / 'phase5_config.json'}")
    print(f"\nPhase 5 complete in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
