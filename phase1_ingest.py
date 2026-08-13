"""
phase1_ingest.py
----------------
Ingest the Reliance 1-second level-5 depth CSV files, validate schema, produce
a tidy per-day parquet in output/clean/ and a manifest + train/test/valid split.

Cross-platform paths:
    ROOT_DIR  = parent of this script
    RAW_DIR   = ROOT_DIR/input
    OUT_DIR   = ROOT_DIR/output
    MODEL_DIR = ROOT_DIR/models
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR  = ROOT_DIR / "input"
OUT_DIR  = ROOT_DIR / "output"
CLEAN_DIR = OUT_DIR / "clean"
MODEL_DIR = ROOT_DIR / "models"

for d in (OUT_DIR, CLEAN_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

TICK_SIZE = 0.05  # Reliance tick in INR
EXPECTED_COLS_BID = 5
EXPECTED_COLS_ASK = 5

# Regular Trading Hours (NSE)
RTH_START = pd.to_datetime("09:15:00").time()
RTH_END   = pd.to_datetime("15:30:00").time()

# FIX: Minimum valid RTH rows to keep a day
MIN_RTH_ROWS = 500

# Date-split: 50/30/20
TRAIN_FRAC, TEST_FRAC, VALID_FRAC = 0.50, 0.30, 0.20

# ---------------------------------------------------------------------------
# 1. Discover files
# ---------------------------------------------------------------------------
def discover_files(raw_dir: Path) -> list[Path]:
    pat = raw_dir / "reliance_depth_*.csv"
    files = sorted(glob.glob(str(pat)))
    if not files:
        raise FileNotFoundError(f"No CSVs found under {raw_dir}")
    return [Path(f) for f in files]


def parse_trade_date(p: Path) -> pd.Timestamp:
    m = re.search(r"reliance_depth_(\d{8})_(\d{6})\.csv", p.name)
    if not m:
        raise ValueError(f"Cannot parse date from {p.name}")
    return pd.to_datetime(m.group(1), format="%Y%m%d")


# ---------------------------------------------------------------------------
# 2. Load + tidy a single day
# ---------------------------------------------------------------------------
BID_LEVELS = [(f"b{i+1}_o", f"b{i+1}_q", f"b{i+1}_p") for i in range(5)]
ASK_LEVELS = [(f"a{i+1}_p", f"a{i+1}_q", f"a{i+1}_o") for i in range(5)]

CANONICAL_COLS = ["ts", "ltp", "recv_time", "depth_time", "quote_time",
                  "depth_age_s", "quote_age_s", "is_stale",
                  "depth_ticks", "quote_ticks", "other_ticks", "exch_time"]
for o, q, p in BID_LEVELS:
    CANONICAL_COLS += [p, q, o]
for p, q, o in ASK_LEVELS:
    CANONICAL_COLS += [p, q, o]


def load_day(path: Path) -> tuple[pd.DataFrame, int, int]:
    """Returns (df, n_pre_rth_dropped, n_bad_quotes)."""
    df = pd.read_csv(path, header=0)
    ncols = df.shape[1]
    expected_ncols = 1 + 3*5 + 3*5 + 11
    if ncols != expected_ncols:
        raise ValueError(f"{path.name}: expected {expected_ncols} cols, got {ncols}")

    rename = {}
    rename[df.columns[0]] = "ts"
    col_idx = 1
    for i in range(5):
        rename[df.columns[col_idx]]   = f"b{i+1}_o"
        rename[df.columns[col_idx+1]] = f"b{i+1}_q"
        rename[df.columns[col_idx+2]] = f"b{i+1}_p"
        col_idx += 3
    for i in range(5):
        rename[df.columns[col_idx]]   = f"a{i+1}_p"
        rename[df.columns[col_idx+1]] = f"a{i+1}_q"
        rename[df.columns[col_idx+2]] = f"a{i+1}_o"
        col_idx += 3
    trailing = ["ltp", "recv_time", "depth_time", "quote_time",
                "depth_age_s", "quote_age_s", "is_stale",
                "depth_ticks", "quote_ticks", "other_ticks", "exch_time"]
    for name in trailing:
        rename[df.columns[col_idx]] = name
        col_idx += 1

    df = df.rename(columns=rename)
    keep = ["ts"] + [f"b{i+1}_p" for i in range(5)] + [f"b{i+1}_q" for i in range(5)] + \
           [f"b{i+1}_o" for i in range(5)] + [f"a{i+1}_p" for i in range(5)] + \
           [f"a{i+1}_q" for i in range(5)] + [f"a{i+1}_o" for i in range(5)] + trailing
    df = df[keep].copy()

    num_cols = [c for c in df.columns if c not in ("ts", "recv_time", "depth_time", "quote_time", "exch_time")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    trade_date = parse_trade_date(path)
    df["datetime"] = pd.to_datetime(trade_date.strftime("%Y-%m-%d") + " " + df["ts"].astype(str),
                                    errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    df["trade_date"] = trade_date

    df = df[~df.index.duplicated(keep="last")]
    n_raw = len(df)

    # RTH filter
    rth_mask = (df.index.time >= RTH_START) & (df.index.time <= RTH_END)
    # FIX: explicit count of dropped pre/post-RTH rows
    n_pre_rth_dropped = int((~rth_mask).sum())
    df = df.loc[rth_mask].copy()

    # Clean bad quotes
    price_cols = [f"b{i+1}_p" for i in range(5)] + [f"a{i+1}_p" for i in range(5)]
    qty_cols   = [f"b{i+1}_q" for i in range(5)] + [f"a{i+1}_q" for i in range(5)]
    for c in price_cols:
        df.loc[df[c] <= 0, c] = np.nan
    for c in qty_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=0)

    bad_book = (df["b1_p"].isna()) | (df["a1_p"].isna()) | (df["b1_p"] >= df["a1_p"])
    n_bad = int(bad_book.sum())
    df.loc[bad_book, ["b1_p", "a1_p"]] = np.nan

    df[price_cols] = df[price_cols].ffill().bfill()
    df[qty_cols]   = df[qty_cols].ffill().fillna(0)
    df = df.dropna(subset=["b1_p", "a1_p"])

    # Re-index to strict 1-second RTH grid
    full_idx = pd.date_range(start=df.index.min().floor('s'),
                             end=df.index.max().ceil('s'), freq='s')
    rth_idx = full_idx[(full_idx.time >= RTH_START) & (full_idx.time <= RTH_END)]
    df = df.reindex(rth_idx)
    df.index.name = "datetime"
    df[price_cols] = df[price_cols].ffill().bfill()
    df[qty_cols]   = df[qty_cols].ffill().fillna(0)
    df["ltp"]      = df["ltp"].ffill().bfill()
    for c in ["b1_o", "b2_o", "b3_o", "b4_o", "b5_o", "a1_o", "a2_o", "a3_o", "a4_o", "a5_o",
              "is_stale", "depth_ticks", "quote_ticks", "other_ticks"]:
        df[c] = df[c].ffill().fillna(0)
    for c in ["depth_age_s", "quote_age_s"]:
        df[c] = df[c].ffill().fillna(0)
    df["trade_date"] = trade_date

    return df, n_pre_rth_dropped, n_bad


# ---------------------------------------------------------------------------
# 3. Baseline canonical series
# ---------------------------------------------------------------------------
def add_basics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bp"] = df["b1_p"]
    df["ap"] = df["a1_p"]
    df["bq"] = df["b1_q"]
    df["aq"] = df["a1_q"]

    df["mid"]   = (df["bp"] + df["ap"]) / 2.0
    df["sprd"]  = df["ap"] - df["bp"]
    df["sprd_t"] = (df["sprd"] / TICK_SIZE).round().astype(int)
    bq, aq = df["bq"], df["aq"]
    df["micro"] = (df["bp"] * aq + df["ap"] * bq) / (bq + aq).replace(0, np.nan)
    df["imb1"]  = (bq - aq) / (bq + aq).replace(0, np.nan)

    df["tot_bq"] = sum(df[f"b{i+1}_q"] for i in range(5))
    df["tot_aq"] = sum(df[f"a{i+1}_q"] for i in range(5))
    df["tot_dp"] = df["tot_bq"] + df["tot_aq"]
    df["depth_imb"] = (df["tot_bq"] - df["tot_aq"]) / df["tot_dp"].replace(0, np.nan)

    b_vwap = sum(df[f"b{i+1}_p"] * df[f"b{i+1}_q"] for i in range(5)) / df["tot_bq"].replace(0, np.nan)
    a_vwap = sum(df[f"a{i+1}_p"] * df[f"a{i+1}_q"] for i in range(5)) / df["tot_aq"].replace(0, np.nan)
    df["b_vwap"] = b_vwap
    df["a_vwap"] = a_vwap

    df["ltp_side"] = np.where(df["ltp"] >= df["ap"], 1,
                     np.where(df["ltp"] <= df["bp"], -1, 0))

    df["mid_diff"] = df["mid"].diff().abs()
    df["mid_unch"] = (df["mid_diff"] == 0).astype(int)

    return df


# ---------------------------------------------------------------------------
# 4. Per-day summary stats
# ---------------------------------------------------------------------------
def day_stats(df: pd.DataFrame, path: Path, n_pre_rth: int, n_bad: int) -> dict:
    t_start, t_end = df.index.min(), df.index.max()
    duration_s = (t_end - t_start).total_seconds() + 1
    expected = int(duration_s)
    return {
        "file":              path.name,
        "trade_date":        df["trade_date"].iloc[0].strftime("%Y-%m-%d"),
        "rows":              len(df),
        "expected_rth_rows": expected,
        "gap_pct":           100.0 * (1.0 - len(df) / max(expected, 1)),
        "t_start":           t_start.strftime("%H:%M:%S"),
        "t_end":             t_end.strftime("%H:%M:%S"),
        "n_pre_rth_dropped": int(n_pre_rth),
        "n_bad_quotes":      int(n_bad),
        "pct_stale":         100.0 * df["is_stale"].fillna(0).mean(),
        "median_sprd_t":     float(df["sprd_t"].median()),
        "q95_sprd_t":        float(df["sprd_t"].quantile(0.95)),
        "q99_sprd_t":        float(df["sprd_t"].quantile(0.99)),
        "median_mid":        float(df["mid"].median()),
        "min_mid":           float(df["mid"].min()),
        "max_mid":           float(df["mid"].max()),
        "mid_range_pct":     100.0 * (df["mid"].max() - df["mid"].min()) / df["mid"].median(),
        "avg_tick_per_s":    float(df["depth_ticks"].diff().clip(lower=0).mean()),
    }


# ---------------------------------------------------------------------------
# 5. Split by date
# ---------------------------------------------------------------------------
def split_dates(dates: list[pd.Timestamp]) -> dict[str, list[str]]:
    dates = sorted(dates)
    n = len(dates)
    n_train = int(round(n * TRAIN_FRAC))
    n_test  = int(round(n * TEST_FRAC))
    n_valid = n - n_train - n_test
    train = dates[:n_train]
    test  = dates[n_train:n_train + n_test]
    valid = dates[n_train + n_test:]
    assert len(train) + len(test) + len(valid) == n
    return {
        "train": [d.strftime("%Y-%m-%d") for d in train],
        "test":  [d.strftime("%Y-%m-%d") for d in test],
        "valid": [d.strftime("%Y-%m-%d") for d in valid],
    }


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("PHASE 1 — INGEST")
    print(f"ROOT_DIR : {ROOT_DIR}")
    print(f"RAW_DIR  : {RAW_DIR}")
    print(f"OUT_DIR  : {OUT_DIR}")
    print("=" * 72)
    files = discover_files(RAW_DIR)
    print(f"Found {len(files)} CSV files.\n")

    records = []
    for path in files:
        print(f"[ingest] {path.name} ... ", end="", flush=True)
        df, n_pre_rth, n_bad = load_day(path)

        # FIX: minimum-row guard
        if len(df) < MIN_RTH_ROWS:
            print(f"SKIPPED (only {len(df)} RTH rows, need >= {MIN_RTH_ROWS})")
            continue

        df = add_basics(df)
        outp = CLEAN_DIR / f"{parse_trade_date(path).strftime('%Y%m%d')}.parquet"
        df.to_parquet(outp, index=True)
        st = day_stats(df, path, n_pre_rth, n_bad)
        records.append(st)
        print(f"rows={st['rows']:6d}  med_sprd={st['median_sprd_t']:.1f}t  "
              f"q95_sprd={st['q95_sprd_t']:.1f}t  stale={st['pct_stale']:.2f}%  "
              f"bad_book={st['n_bad_quotes']:5d}  preRTH_dropped={st['n_pre_rth_dropped']:6d}")

    if not records:
        print("[FATAL] No valid days after filtering.")
        sys.exit(1)

    manifest = pd.DataFrame(records).sort_values("trade_date").reset_index(drop=True)
    manifest.to_csv(OUT_DIR / "manifest.csv", index=False)

    dates = [pd.to_datetime(d) for d in manifest["trade_date"].tolist()]
    splits = split_dates(dates)
    with open(OUT_DIR / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print("\n" + "-" * 72)
    print("MANIFEST (per-day, sorted by date):")
    print(manifest.to_string(index=False))

    print("\n" + "-" * 72)
    print("SPLITS (date-wise, no peeking):")
    for k in ("train", "test", "valid"):
        print(f"  {k.upper():5s} ({len(splits[k])} days): {splits[k]}")

    print(f"\nSaved : {OUT_DIR / 'manifest.csv'}")
    print(f"        {OUT_DIR / 'splits.json'}")
    print(f"        {CLEAN_DIR}/<yyyymmdd>.parquet x {len(records)}")
    print("Phase 1 complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] {e}", file=sys.stderr)
        raise
