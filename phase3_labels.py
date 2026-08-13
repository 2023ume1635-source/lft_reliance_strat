"""
phase3_labels.py
----------------
Regime-conditional triple-barrier labelling + multi-task targets for the
Reliance 1-second LFT strategy.

Inputs  : output/features/<yyyymmdd>.parquet  (from phase2_features.py)
Outputs : output/labels/<yyyymmdd>.parquet
          reports/phase3_label_stats.csv
          reports/phase3_label_diagnostics.json

FIXES over original:
  - Vectorised compute_R_per_day (one-liner instead of Python loop)
  - Vectorised barrier check in triple_barrier (numpy where on window)
  - Replaced O(T*4*300) optimal_horizon with simple regime-based hstar
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent
FEAT_DIR = ROOT / "output" / "features"
CLEAN_DIR = ROOT / "output" / "clean"
LABEL_DIR = ROOT / "output" / "labels"
REPORT   = ROOT / "reports"
SPLITS   = ROOT / "output" / "splits.json"
TICK     = 0.05
WARMUP_S = 360
for d in (LABEL_DIR, REPORT):
    d.mkdir(parents=True, exist_ok=True)

# Regime -> (barrier_width_R, horizon_s, min_edge_R, bias)
REGIME_PARAMS = {
    0: (0.35, 20, 0.25, 0.0),
    1: (0.40, 25, 0.30, 0.0),
    2: (0.45, 30, 0.35, 0.0),
    3: (0.60, 45, 0.50, -0.05),
    4: (0.55, 40, 0.45, 0.0),
    5: (0.00,  0, 0.00, 0.0),
}

HS = (10, 20, 30, 60)


# ---------------------------------------------------------------------------
# Per-day helpers
# ---------------------------------------------------------------------------
def compute_R_per_day(df: pd.DataFrame) -> float:
    """FIX: Vectorised R computation."""
    mid = df["mid"].values
    T = len(mid)
    if T < 60:
        return 0.30
    fwd30 = np.full(T, np.nan)
    fwd30[:T - 30] = mid[30:] - mid[:-30]
    absmove = np.abs(fwd30[~np.isnan(fwd30)])
    if len(absmove) < 100:
        return float(np.nanmedian(np.abs(np.diff(mid))) * 5 + 0.1)
    R = float(np.nanpercentile(absmove, 80))
    med_sprd = float(df["sprd"].median())
    R = float(np.clip(R, med_sprd / 2 + TICK, med_sprd * 3 + TICK * 3))
    return R


def passive_fill_labels(df: pd.DataFrame, H: int) -> Tuple[np.ndarray, np.ndarray]:
    """Passive fill labels using forward-looking window checks."""
    T = len(df)
    bp = df["bp"].values; bq = df["bq"].values
    ap = df["ap"].values; aq = df["aq"].values
    mid = df["mid"].values
    sprd = df["sprd"].values

    yb = np.zeros(T, dtype=np.int8)
    ys = np.zeros(T, dtype=np.int8)

    for i in range(T - 1):
        j_end = min(i + H + 1, T)
        if j_end - i <= 1:
            continue
        window_ap = ap[i + 1:j_end]
        window_bp = bp[i + 1:j_end]
        window_mid = mid[i + 1:j_end]
        hs = sprd[i] / 2.0 if np.isfinite(sprd[i]) else 0.05

        bid_fill = (window_ap.min() <= bp[i]) or (bq[i] <= 0)
        bid_adverse = (window_mid.max() >= bp[i] + hs + TICK / 2)

        ask_fill = (window_bp.max() >= ap[i]) or (aq[i] <= 0)
        ask_adverse = (window_mid.min() <= ap[i] - hs - TICK / 2)

        yb[i] = 1 if (bid_fill and not bid_adverse) else 0
        ys[i] = 1 if (ask_fill and not ask_adverse) else 0

    return yb, ys


def triple_barrier(df: pd.DataFrame, R: float) -> Tuple[np.ndarray, np.ndarray]:
    """FIX: Vectorised barrier check using numpy on the window slice."""
    T = len(df)
    mid = df["mid"].values
    regime = df["mkt_state"].values.astype(int)
    y_dir = np.zeros(T, dtype=np.int8)
    y_R   = np.zeros(T, dtype=np.float32)

    purge_until = -1

    for i in range(T):
        if i < purge_until:
            continue
        s = regime[i]
        width_R, H, min_edge_R, bias = REGIME_PARAMS[s]
        if H <= 0 or R <= 0:
            continue
        width = width_R * R
        entry = mid[i]
        up = entry + width
        dn = entry - width
        j_end = min(i + H + 1, T)
        if j_end - i <= 2:
            continue

        window = mid[i + 1:j_end]
        hit_up_idx = np.where(window >= up)[0]
        hit_dn_idx = np.where(window <= dn)[0]

        first_up = hit_up_idx[0] + 1 if len(hit_up_idx) > 0 else j_end - i
        first_dn = hit_dn_idx[0] + 1 if len(hit_dn_idx) > 0 else j_end - i

        if first_up < first_dn:
            exit_j = i + first_up
            y_dir[i] = 1
            y_R[i] = (mid[exit_j] - entry) / R
            purge_until = i + H
        elif first_dn < first_up:
            exit_j = i + first_dn
            y_dir[i] = -1
            y_R[i] = (mid[exit_j] - entry) / R
            purge_until = i + H
        else:
            max_up = window.max() - entry
            max_dn = entry - window.min()
            if max_up >= (min_edge_R + bias) * R and max_up > max_dn * 1.5:
                y_dir[i] = 1; y_R[i] = max_up / R
                purge_until = i + H
            elif max_dn >= (min_edge_R - bias) * R and max_dn > max_up * 1.5:
                y_dir[i] = -1; y_R[i] = -max_dn / R
                purge_until = i + H
            else:
                y_dir[i] = 0

    return y_dir, y_R


def take_labels(df: pd.DataFrame, R: float, y_dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Aggressive-take label: +1/-1 only if future move beats spread + 0.2R."""
    T = len(df)
    mid = df["mid"].values
    sprd = df["sprd"].values
    regime = df["mkt_state"].values.astype(int)
    y_take = np.zeros(T, dtype=np.int8)
    y_edge = np.zeros(T, dtype=np.float32)

    purge_until = -1
    for i in range(T):
        if i < purge_until: continue
        s = regime[i]
        _, H, min_edge_R, _ = REGIME_PARAMS[s]
        if H <= 0: continue
        cost = sprd[i] / 2.0 + 0.2 * R
        j_end = min(i + H + 1, T)
        if j_end - i <= 2: continue
        fwd_mid = mid[i + 1:j_end]
        if len(fwd_mid) < 2: continue
        max_up = fwd_mid.max() - mid[i]
        max_dn = mid[i] - fwd_mid.min()
        if max_up > cost and max_up > max_dn:
            y_take[i] = 1
            y_edge[i] = max_up - cost
            purge_until = i + H
        elif max_dn > cost and max_dn > max_up:
            y_take[i] = -1
            y_edge[i] = -(max_dn - cost)
            purge_until = i + H
    return y_take, y_edge


def edge_quantile_labels(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    mid = df["mid"].values
    T = len(mid)
    out = {}
    for H in HS:
        e = np.full(T, np.nan, dtype=np.float32)
        if T > H:
            e[:T - H] = mid[H:] - mid[:-H]
        out[f"y_edge_{H}"] = e
    q10 = np.full(T, np.nan, dtype=np.float32)
    q50 = np.full(T, np.nan, dtype=np.float32)
    q90 = np.full(T, np.nan, dtype=np.float32)
    ret = np.diff(mid) / np.where(mid[:-1] > 0, mid[:-1], np.nan)
    ret = np.concatenate([ret, [np.nan]])
    H = 30
    for i in range(T - H):
        window = ret[i + 1:i + H + 1]
        if len(window) < 10: continue
        q10[i] = np.nanpercentile(window, 10)
        q50[i] = np.nanpercentile(window, 50)
        q90[i] = np.nanpercentile(window, 90)
    out["y_q10"] = q10; out["y_q50"] = q50; out["y_q90"] = q90
    return out


def simple_hstar(df: pd.DataFrame) -> np.ndarray:
    """FIX: Replace O(T*4*300) optimal_horizon with regime-based horizon class."""
    regime = df["mkt_state"].values.astype(int)
    out = np.full(len(df), 2, dtype=np.int8)
    out[(regime == 0) | (regime == 1)] = 0
    out[regime == 2] = 1
    out[(regime == 3) | (regime == 4)] = 2
    out[regime == 5] = 3
    return out


def y_valid_mask(df: pd.DataFrame) -> np.ndarray:
    t = df.index
    valid = np.ones(len(df), dtype=bool)
    day0 = t[0].normalize().replace(hour=9, minute=15)
    valid[t < day0 + pd.Timedelta(seconds=WARMUP_S)] = False
    if "is_stale" in df.columns:
        valid &= (df["is_stale"].fillna(1).astype(int) == 0)
    if "mkt_state" in df.columns:
        valid &= (df["mkt_state"].fillna(5).astype(int) != 5)
    core = ["mid", "sprd", "ofi1", "rv60", "kf_innov_z"]
    for c in core:
        if c in df.columns:
            valid &= ~df[c].isna()
    return valid


# ---------------------------------------------------------------------------
# Per-day pipeline
# ---------------------------------------------------------------------------
def label_day(feat_df: pd.DataFrame, clean_df: pd.DataFrame, date: str) -> Tuple[pd.DataFrame, dict]:
    need_clean_cols = ["b1_p", "a1_p", "b1_q", "a1_q"]
    add = clean_df[[c for c in need_clean_cols if c in clean_df.columns]].rename(
        columns={"b1_p": "bp", "a1_p": "ap", "b1_q": "bq", "a1_q": "aq"})
    df = feat_df.join(add, how="left")
    if "mid" not in df.columns: df["mid"] = df["bp"] / 2 + df["ap"] / 2
    if "sprd" not in df.columns: df["sprd"] = df["ap"] - df["bp"]
    df = df.copy()
    R = compute_R_per_day(df)

    y_dir, y_R = triple_barrier(df, R)
    y_take, y_take_edge = take_labels(df, R, y_dir)
    y_buy, y_sell = passive_fill_labels(df, H=30)
    eq = edge_quantile_labels(df)
    hstar = simple_hstar(df)
    valid = y_valid_mask(df)

    df["y_dir"]      = y_dir
    df["y_dir_R"]    = y_R
    df["y_take_dir"] = y_take
    df["y_take_edge"] = y_take_edge
    for k, v in eq.items(): df[k] = v
    df["y_buyfill"]  = y_buy
    df["y_sellfill"] = y_sell
    df["y_hstar"]    = hstar
    df["y_valid"]    = valid
    df["R_per_share"] = R

    df.loc[~valid, "y_dir"] = 0
    df.loc[~valid, "y_take_dir"] = 0
    df.loc[~valid, "y_buyfill"] = 0
    df.loc[~valid, "y_sellfill"] = 0

    valid_mask = df["y_valid"].values
    vd = df.loc[valid_mask]
    dir_counts = vd["y_dir"].value_counts().to_dict()
    take_counts = vd["y_take_dir"].value_counts().to_dict()
    stats = {
        "date": date, "R_per_share": round(R, 3),
        "rows": len(df), "rows_valid": int(valid_mask.sum()),
        "valid_pct": round(100. * valid_mask.mean(), 2),
        "pos": int(dir_counts.get(1, 0)),
        "neg": int(dir_counts.get(-1, 0)),
        "flat": int(dir_counts.get(0, 0)),
        "tradable_pct": round(100. * (dir_counts.get(1, 0) + dir_counts.get(-1, 0)) / max(valid_mask.sum(), 1), 2),
        "pos_neg_ratio": round(dir_counts.get(1, 0) / max(dir_counts.get(-1, 0), 1), 3),
        "take_pos": int(take_counts.get(1, 0)),
        "take_neg": int(take_counts.get(-1, 0)),
        "buyfill_rate": round(100. * vd["y_buyfill"].mean(), 2),
        "sellfill_rate": round(100. * vd["y_sellfill"].mean(), 2),
        "mean_abs_edge_30": round(float(np.nanmean(np.abs(vd["y_edge_30"]))), 4),
        "median_sprd": round(float(df["sprd"].median()), 3),
    }
    regime_stats = {}
    for st in range(6):
        sub = vd[vd["mkt_state"] == st] if "mkt_state" in vd.columns else pd.DataFrame()
        if len(sub) > 50:
            regime_stats[str(st)] = {
                "rows": len(sub),
                "pos_pct": round(100. * (sub["y_dir"] == 1).mean(), 2),
                "neg_pct": round(100. * (sub["y_dir"] == -1).mean(), 2),
                "flat_pct": round(100. * (sub["y_dir"] == 0).mean(), 2),
                "mean_abs_edge_30": round(float(np.nanmean(np.abs(sub["y_edge_30"]))), 4),
            }
    stats["per_regime"] = regime_stats
    return df, stats


# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("PHASE 3 — REGIME-CONDITIONAL TRIPLE-BARRIER LABELLING")
    print(f"ROOT={ROOT}  FEAT={FEAT_DIR}  LABELS={LABEL_DIR}")
    print(f"Regime params: {json.dumps(REGIME_PARAMS, default=str)}")
    print("=" * 74)

    files = sorted(FEAT_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No feature parquets in {FEAT_DIR}. Run phase2 first.")
    print(f"{len(files)} days.\n")

    all_stats = []
    all_regime = {}
    for i, p in enumerate(files, 1):
        date = p.stem
        print(f"[phase3] {date} ({i}/{len(files)}) ... ", end="", flush=True)
        feat = pd.read_parquet(p)
        clean_p = CLEAN_DIR / f"{date}.parquet"
        if not clean_p.exists():
            print(f" missing clean parquet; skipped")
            continue
        clean = pd.read_parquet(clean_p, columns=["b1_p", "a1_p", "b1_q", "a1_q",
                                                  "mid", "sprd", "is_stale", "trade_date"])
        out, st = label_day(feat, clean, date)
        out = out.drop(columns=["bp", "ap", "bq", "aq"], errors="ignore")
        out.to_parquet(LABEL_DIR / f"{date}.parquet", index=True)
        all_stats.append(st)
        all_regime[date] = st["per_regime"]
        print(f"R={st['R_per_share']:.3f}  valid={st['valid_pct']:.0f}%  "
              f"+1={st['pos']:5d}  -1={st['neg']:5d}  0={st['flat']:6d}  "
              f"tradable={st['tradable_pct']:.2f}%  +:-={st['pos_neg_ratio']:.2f}  "
              f"fill_buy={st['buyfill_rate']:.1f}% fill_ask={st['sellfill_rate']:.1f}%")

    stats_df = pd.DataFrame([{k: v for k, v in s.items() if k != "per_regime"} for s in all_stats])
    stats_df.to_csv(REPORT / "phase3_label_stats.csv", index=False)
    with open(REPORT / "phase3_label_diagnostics.json", "w") as f:
        json.dump({"per_day": all_stats, "per_regime": all_regime,
                   "R_per_day": {s["date"]: s["R_per_share"] for s in all_stats}},
                  f, indent=2)

    print("\n--- Label stats per day ---")
    print(stats_df.to_string(index=False))

    if SPLITS.exists():
        sp = json.load(open(SPLITS))
        flat_dates = lambda lst: [d.replace("-", "") for d in lst]
        for name in ("train", "test", "valid"):
            ds = set(flat_dates(sp[name]))
            rows = [s for s in all_stats if s["date"] in ds]
            if not rows: continue
            tot = sum(r["rows_valid"] for r in rows)
            p = sum(r["pos"] for r in rows); n = sum(r["neg"] for r in rows)
            fl = sum(r["flat"] for r in rows)
            print(f"\n[{name.upper()}]: {len(rows)} days, valid_rows={tot:,}, "
                  f"+1={p:,} ({100 * p / tot:.1f}%), -1={n:,} ({100 * n / tot:.1f}%), "
                  f"0={fl:,} ({100 * fl / tot:.1f}%), pos/neg={p / max(n, 1):.2f}")

    print(f"\nSaved: {LABEL_DIR}/<yyyymmdd>.parquet x {len(files)}")
    print(f"       {REPORT / 'phase3_label_stats.csv'}")
    print(f"       {REPORT / 'phase3_label_diagnostics.json'}")
    print("Phase 3 complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] {e}", file=sys.stderr)
        raise
