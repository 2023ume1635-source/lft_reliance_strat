"""
================================================================================
PHASE 6 — EVENT-DRIVEN FILL-AWARE BACKTEST (per-second) — v2 fixed
================================================================================

FIXES over original:
  - Unified COOLDOWN_S = 15 (was 30, inconsistent with phase5)
  - Added MIN_HOLD_S = 3: stop/target not evaluated for first 3 seconds
  - Partial fill for passive orders: fill qty = min(size_qty, available L1 depth)
  - Calibrated passive fill exponent (removed arbitrary 1.5x fudge)
  - No transaction costs (per user request)

Consumes output/signals/<yyyymmdd>.parquet from phase5.
"""

from __future__ import annotations
import json, time, warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parent
SIG_DIR  = ROOT / "output" / "signals"
SPLIT_F  = ROOT / "output" / "splits.json"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TICK = 0.05
INITIAL_CAPITAL = 10_00_000.0
MAX_DAILY_LOSS  = 0.02 * INITIAL_CAPITAL   # -Rs 20,000
COOLDOWN_S      = 15                        # FIX: unified with phase5
MIN_HOLD_S      = 3                         # FIX: minimum holding before stop/target
RNG_SEED = 42
PASSIVE_FILL_EXP = 1.0                      # FIX: removed arbitrary 1.5x fudge


def load_split():
    with open(SPLIT_F) as f: return json.load(f)

def _ymd(s): return s.replace("-", "")


def simulate_day(df: pd.DataFrame, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = df.sort_index()
    t = df.index
    rth_start = t[0].normalize().replace(hour=9, minute=15)
    rth_end   = t[0].normalize().replace(hour=15, minute=30)
    df = df[(df.index >= rth_start) & (df.index <= rth_end)].copy()
    n = len(df)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame(), dict(trades=0, pnl=0.0)

    sig     = df["sig"].values.astype(np.int8)
    entry_p = df["entry_ref"].values.astype(np.float64)
    stop_p  = df["stop_p"].values.astype(np.float64)
    tgt_p   = df["tgt_p"].values.astype(np.float64)
    H       = df["horizon_s"].values.astype(np.int16)
    risk_ok = df["risk_ok"].values.astype(bool)
    bp      = df["b1_p"].values.astype(np.float64)
    ap      = df["a1_p"].values.astype(np.float64)
    bq      = df["b1_q"].values.astype(np.float64)
    aq      = df["a1_q"].values.astype(np.float64)
    sz      = df["size_qty"].values.astype(np.int32)
    fill_b  = df["p_buyfill"].values.astype(np.float64)
    fill_s  = df["p_sellfill"].values.astype(np.float64)
    regime  = df["mkt_state"].values.astype(np.int8)

    cap = INITIAL_CAPITAL
    kill = False
    trades: List[dict] = []
    equity = np.zeros(n, dtype=np.float64)
    equity[:] = cap

    pos = dict(active=False, side=0, qty=0, entry=0.0, stop=0.0, tgt=0.0,
               entry_i=-1, H_left=0, passive=False, fill_p=0.0, posted_at=-1,
               entered=False, H_horizon=0, hold_count=0)
    cooldown = 0
    intra_pnl = 0.0

    for i in range(n):
        if pos["active"]:
            mid_i = 0.5 * (bp[i] + ap[i]) if np.isfinite(bp[i]) and np.isfinite(ap[i]) else pos["entry"]
            if pos["side"] > 0: unreal = (mid_i - pos["entry"]) * pos["qty"]
            else:               unreal = (pos["entry"] - mid_i) * pos["qty"]
        else:
            unreal = 0.0
        equity[i] = cap + unreal

        # Kill switch
        if not kill and (intra_pnl + unreal) <= -MAX_DAILY_LOSS:
            if pos["active"]:
                mid_i = 0.5 * (bp[i] + ap[i]) if np.isfinite(bp[i]) and np.isfinite(ap[i]) else pos["entry"]
                if pos["side"] > 0: pnl = (mid_i - pos["entry"]) * pos["qty"]
                else:               pnl = (pos["entry"] - mid_i) * pos["qty"]
                trades.append(dict(entry_t=df.index[pos["entry_i"]], exit_t=df.index[i],
                                   side=pos["side"], qty=pos["qty"], entry=pos["entry"],
                                   exit=mid_i, pnl=pnl, reason="kill", passive=pos["passive"],
                                   regime=int(regime[pos["entry_i"]]),
                                   hold_s=int(i - pos["entry_i"])))
                cap += pnl; intra_pnl += pnl
                pos = dict(active=False, side=0, qty=0, entry=0.0, stop=0.0, tgt=0.0,
                           entry_i=-1, H_left=0, passive=False, fill_p=0.0, posted_at=-1,
                           entered=False, H_horizon=0, hold_count=0)
            kill = True
            cooldown = COOLDOWN_S * 10
            equity[i] = cap
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if kill:
            continue

        if pos["active"]:
            mid_i = 0.5 * (bp[i] + ap[i]) if np.isfinite(bp[i]) and np.isfinite(ap[i]) else pos["entry"]

            # Passive order awaiting fill
            if not pos["entered"]:
                pos["H_left"] -= 1
                filled = False
                if pos["side"] > 0 and ap[i] <= pos["entry"]:
                    filled = True
                elif pos["side"] < 0 and bp[i] >= pos["entry"]:
                    filled = True
                else:
                    p_total = float(np.clip(pos["fill_p"], 0.02, 0.95))
                    Hh = max(int(pos["H_horizon"]), 1)
                    p_per_s = 1.0 - (1.0 - p_total) ** (PASSIVE_FILL_EXP / Hh)
                    if rng.random() < p_per_s:
                        filled = True
                # Cancel on adverse move
                if pos["side"] > 0 and bp[i] < pos["entry"] - 2 * TICK:
                    pos = dict(active=False, side=0, qty=0, entry=0.0, stop=0.0, tgt=0.0,
                               entry_i=-1, H_left=0, passive=False, fill_p=0.0, posted_at=-1,
                               entered=False, H_horizon=0, hold_count=0)
                    cooldown = 2; continue
                if pos["side"] < 0 and ap[i] > pos["entry"] + 2 * TICK:
                    pos = dict(active=False, side=0, qty=0, entry=0.0, stop=0.0, tgt=0.0,
                               entry_i=-1, H_left=0, passive=False, fill_p=0.0, posted_at=-1,
                               entered=False, H_horizon=0, hold_count=0)
                    cooldown = 2; continue
                if pos["H_left"] <= 0 and not filled:
                    pos = dict(active=False, side=0, qty=0, entry=0.0, stop=0.0, tgt=0.0,
                               entry_i=-1, H_left=0, passive=False, fill_p=0.0, posted_at=-1,
                               entered=False, H_horizon=0, hold_count=0)
                    cooldown = 5; continue
                if filled:
                    pos["entered"] = True
                    pos["entry_i"] = i
                    pos["H_left"] = pos["H_horizon"]
                    pos["hold_count"] = 0
                    # FIX: partial fill
                    avail = bq[i] if pos["side"] > 0 else aq[i]
                    if np.isfinite(avail) and avail > 0:
                        pos["qty"] = min(pos["qty"], int(avail))
                continue

            # Position entered: evaluate exits
            pos["hold_count"] += 1
            exit_reason = None; exit_price = None

            # FIX: MIN_HOLD_S — don't evaluate stop/target for first N seconds
            if pos["hold_count"] > MIN_HOLD_S:
                if pos["side"] > 0:
                    if mid_i <= pos["stop"]: exit_reason = "stop"; exit_price = pos["stop"]
                    elif mid_i >= pos["tgt"]: exit_reason = "target"; exit_price = pos["tgt"]
                else:
                    if mid_i >= pos["stop"]: exit_reason = "stop"; exit_price = pos["stop"]
                    elif mid_i <= pos["tgt"]: exit_reason = "target"; exit_price = pos["tgt"]

            # Opposite signal
            if exit_reason is None and sig[i] != 0 and np.sign(sig[i]) == -pos["side"]:
                exit_reason = "flip"; exit_price = mid_i
            # Horizon
            pos["H_left"] -= 1
            if exit_reason is None and pos["H_left"] <= 0:
                exit_reason = "time"; exit_price = mid_i
            if exit_reason is not None:
                if pos["side"] > 0: pnl = (exit_price - pos["entry"]) * pos["qty"]
                else:               pnl = (pos["entry"] - exit_price) * pos["qty"]
                trades.append(dict(entry_t=df.index[pos["entry_i"]], exit_t=df.index[i],
                                   side=pos["side"], qty=pos["qty"], entry=pos["entry"],
                                   exit=float(exit_price), pnl=float(pnl),
                                   reason=exit_reason, passive=pos["passive"],
                                   regime=int(regime[pos["entry_i"]]),
                                   hold_s=int(i - pos["entry_i"])))
                cap += pnl; intra_pnl += pnl
                pos = dict(active=False, side=0, qty=0, entry=0.0, stop=0.0, tgt=0.0,
                           entry_i=-1, H_left=0, passive=False, fill_p=0.0, posted_at=-1,
                           entered=False, H_horizon=0, hold_count=0)
                cooldown = COOLDOWN_S
                equity[i] = cap
                continue
        else:
            # No position: evaluate new signal
            if sig[i] == 0 or not risk_ok[i]: continue
            if sz[i] <= 0 or H[i] <= 0: continue
            side = int(np.sign(sig[i]))
            passive = (abs(int(sig[i])) == 1)
            if passive:
                p_entry = bp[i] if side > 0 else ap[i]
                p_stop = stop_p[i] if stop_p[i] > 0 else (p_entry - 0.10 if side > 0 else p_entry + 0.10)
                p_tgt  = tgt_p[i]  if tgt_p[i]  > 0 else (p_entry + 0.30 if side > 0 else p_entry - 0.30)
                p_fill = fill_b[i] if side > 0 else fill_s[i]
                pos = dict(active=True, side=side, qty=int(sz[i]), entry=float(p_entry),
                           stop=float(p_stop), tgt=float(p_tgt), entry_i=i,
                           H_left=int(H[i]), passive=True, fill_p=float(p_fill),
                           posted_at=i, H_horizon=int(H[i]), entered=False, hold_count=0)
            else:
                p_entry = ap[i] if side > 0 else bp[i]
                pos = dict(active=True, side=side, qty=int(sz[i]), entry=float(p_entry),
                           stop=float(stop_p[i]), tgt=float(tgt_p[i]), entry_i=i,
                           H_left=int(H[i]), passive=False, fill_p=1.0,
                           posted_at=i, H_horizon=int(H[i]), entered=True, hold_count=0)

    # EOD close
    if pos["active"] and pos["entered"]:
        i = n - 1; mid_i = 0.5 * (bp[i] + ap[i])
        if pos["side"] > 0: pnl = (mid_i - pos["entry"]) * pos["qty"]
        else:               pnl = (pos["entry"] - mid_i) * pos["qty"]
        trades.append(dict(entry_t=df.index[pos["entry_i"]], exit_t=df.index[i],
                           side=pos["side"], qty=pos["qty"], entry=pos["entry"],
                           exit=float(mid_i), pnl=float(pnl), reason="eod",
                           passive=pos["passive"], regime=int(regime[pos["entry_i"]]),
                           hold_s=int(i - pos["entry_i"])))
        cap += pnl; intra_pnl += pnl
        equity[-1] = cap

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame({"eq": equity, "ts": df.index}).set_index("ts")
    day = dict(
        date=str(df.index[0].date()), n_secs=n, n_trades=len(trades_df),
        n_long=int((trades_df.get("side", pd.Series(dtype=int)) == 1).sum()) if len(trades_df) else 0,
        n_short=int((trades_df.get("side", pd.Series(dtype=int)) == -1).sum()) if len(trades_df) else 0,
        n_passive=int((trades_df.get("passive", pd.Series(dtype=bool)) == True).sum()) if len(trades_df) else 0,
        n_aggressive=int((trades_df.get("passive", pd.Series(dtype=bool)) == False).sum()) if len(trades_df) else 0,
        pnl=float(trades_df["pnl"].sum()) if len(trades_df) else 0.0,
        kill=bool(kill),
    )
    if len(trades_df):
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        day["win_rate"] = float(len(wins) / len(trades_df))
        day["avg_win"] = float(wins["pnl"].mean()) if len(wins) else 0.0
        day["avg_loss"] = float(losses["pnl"].mean()) if len(losses) else 0.0
        day["profit_factor"] = (float(wins["pnl"].sum()) / abs(float(losses["pnl"].sum()))) \
                               if len(losses) and losses["pnl"].sum() != 0 else float("inf")
        eq = equity; peak = np.maximum.accumulate(eq); dd = eq - peak
        day["max_dd"] = float(dd.min())
        rets = np.diff(eq) / (eq[:-1] + 1e-9)
        day["sharpe_ann"] = float(np.sqrt(22500) * rets.mean() / rets.std()) if rets.std() > 1e-9 else float("nan")
        day["avg_hold_s"] = float(trades_df["hold_s"].mean())
    else:
        day.update(dict(win_rate=0.0, avg_win=0.0, avg_loss=0.0, profit_factor=float("nan"),
                        max_dd=0.0, sharpe_ann=float("nan"), avg_hold_s=0.0))
    return trades_df, eq_df, day


# ---------------------------------------------------------------------------
def summarize_split(trades: pd.DataFrame, daily: pd.DataFrame, eq_curve: pd.Series) -> dict:
    if len(daily) == 0: return dict(n_days=0)
    total_pnl = float(daily["pnl"].sum())
    wins = trades[trades["pnl"] > 0]; losses = trades[trades["pnl"] <= 0]
    pf = float(wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) and losses["pnl"].sum() != 0 else float("inf")
    eq = eq_curve.values; peak = np.maximum.accumulate(eq); dd = eq - peak
    rets = np.diff(eq) / (eq[:-1] + 1e-9)
    sharpe = float(np.sqrt(22500) * rets.mean() / rets.std()) if rets.std() > 1e-9 else float("nan")
    sortino_down = rets[rets < -1e-9]
    sortino = float(np.sqrt(22500) * rets.mean() / sortino_down.std()) if len(sortino_down) > 5 and sortino_down.std() > 1e-9 else float("nan")
    return dict(
        n_days=int(len(daily)), n_trades=int(len(trades)),
        pnl=round(total_pnl, 2), avg_daily_pnl=round(float(daily["pnl"].mean()), 2),
        win_rate=float((trades["pnl"] > 0).mean()) if len(trades) else 0.0,
        win_rate_days=float((daily["pnl"] > 0).mean()),
        avg_win=round(float(wins["pnl"].mean()), 2) if len(wins) else 0.0,
        avg_loss=round(float(losses["pnl"].mean()), 2) if len(losses) else 0.0,
        profit_factor=round(pf, 3) if np.isfinite(pf) else None,
        max_dd=round(float(dd.min()), 2),
        sharpe_ann=round(sharpe, 3) if np.isfinite(sharpe) else None,
        sortino_ann=round(sortino, 3) if np.isfinite(sortino) else None,
        n_kill_days=int(daily["kill"].sum()),
        pct_passive=float(daily["n_passive"].sum() / max(1, len(trades))),
        avg_trades_per_day=round(float(daily["n_trades"].mean()), 1),
        avg_hold_s=round(float(trades["hold_s"].mean()), 1) if len(trades) else 0.0,
    )


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 78)
    print("PHASE 6 — FILL-AWARE BACKTEST (v2 fixed)")
    print(f"ROOT={ROOT}  SIGNALS={SIG_DIR}")
    print(f"Cooldown: {COOLDOWN_S}s  |  MinHold: {MIN_HOLD_S}s")
    print("=" * 78)
    splits = load_split()
    train_d = [_ymd(d) for d in splits["train"]]
    test_d  = [_ymd(d) for d in splits["test"]]
    valid_d = [_ymd(d) for d in splits["valid"]]
    all_d = train_d + test_d + valid_d
    rng = np.random.default_rng(RNG_SEED)
    all_eqs: Dict[str, pd.Series] = {}
    day_rows: List[dict] = []
    trade_rows: List[pd.DataFrame] = []
    split_map = {}
    for d in train_d: split_map[d] = "train"
    for d in test_d:  split_map[d] = "test"
    for d in valid_d: split_map[d] = "valid"
    cur_cap = {"train": 0.0, "test": 0.0, "valid": 0.0}
    print("\nRunning per-day simulation ...")
    for d in all_d:
        p = SIG_DIR / f"{d}.parquet"
        if not p.exists():
            print(f"  [warn] missing {p}; skipped"); continue
        df = pd.read_parquet(p)
        tdf, eq, day = simulate_day(df, rng)
        day["split"] = split_map[d]
        sp = split_map[d]
        eq["eq"] = eq["eq"] - INITIAL_CAPITAL + cur_cap[sp]
        cur_cap[sp] = float(eq["eq"].iloc[-1]) if len(eq) else cur_cap[sp]
        all_eqs[d] = eq["eq"]
        if len(tdf):
            tdf["date"] = d; tdf["split"] = sp
            trade_rows.append(tdf)
        day_rows.append(day)
        print(f"  {d} [{sp:5s}] trades={day['n_trades']:4d}  "
              f"pnl=Rs{day['pnl']:>+9,.2f}  "
              f"win%={day.get('win_rate', 0):.2%}  "
              f"pf={day.get('profit_factor', 0) if np.isfinite(day.get('profit_factor', 0)) else 'inf':>5}  "
              f"maxDD=Rs{day.get('max_dd', 0):>+9,.2f}  "
              f"sharpe={day.get('sharpe_ann', float('nan')):+.2f}  "
              f"kill={day['kill']}")
    daily = pd.DataFrame(day_rows)
    trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    trades.to_csv(REPORT_DIR / "phase6_trades.csv", index=False)
    daily.to_csv(REPORT_DIR / "phase6_daily.csv", index=False)

    eq_split = {}
    for sp, ds in (("train", train_d), ("test", test_d), ("valid", valid_d)):
        eqs = [all_eqs[d] for d in ds if d in all_eqs]
        eq_split[sp] = pd.concat(eqs).sort_index() if eqs else pd.Series(dtype=float)

    summary = {}
    for sp in ("train", "test", "valid"):
        dtr = trades[trades["split"] == sp] if len(trades) else trades
        dd = daily[daily["split"] == sp]
        summary[sp] = summarize_split(dtr, dd, eq_split[sp])
    if all_eqs:
        eq_all = pd.concat([all_eqs[d] for d in all_d if d in all_eqs]).sort_index()
        summary["overall"] = summarize_split(trades, daily, eq_all)
    else:
        summary["overall"] = dict(n_days=0)

    print("\n--- Split summary ---")
    for sp in ("train", "test", "valid", "overall"):
        s = summary[sp]
        print(f"\n[{sp.upper():7s}] {s.get('n_days', 0)} days, {s.get('n_trades', 0)} trades, "
              f"P&L=Rs{s.get('pnl', 0):>+,.2f}, "
              f"win%={s.get('win_rate', 0):.2%} ({s.get('win_rate_days', 0):.0%} days), "
              f"PF={s.get('profit_factor', None)}, "
              f"maxDD=Rs{s.get('max_dd', 0):>+,.2f}, Sharpe={s.get('sharpe_ann', None)}")
    with open(REPORT_DIR / "phase6_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, sp in zip(axes, ("train", "test", "valid")):
            eq = eq_split[sp]
            if len(eq):
                ax.plot(eq.index, eq.values, lw=0.8)
                ax.axhline(0, color="k", lw=0.5)
                ax.set_title(f"{sp.upper()} equity (Rs)")
                ax.grid(alpha=0.3)
        fig.suptitle(f"Phase 6 — overall P&L = Rs{summary['overall'].get('pnl', 0):+,.0f}")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "phase6_equity.png", dpi=120)
        plt.close(fig)
        print(f"\nSaved: {REPORT_DIR / 'phase6_equity.png'}")
    except Exception as e:
        print(f"\n[info] plot skipped: {e}")
    print(f"Saved: {REPORT_DIR / 'phase6_trades.csv'}")
    print(f"       {REPORT_DIR / 'phase6_daily.csv'}")
    print(f"       {REPORT_DIR / 'phase6_summary.json'}")
    print(f"\nPhase 6 complete in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
