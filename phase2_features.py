"""
phase2_features.py
------------------
Feature engineering for the Reliance LFT 1-second strategy.

Inputs : output/clean/<yyyymmdd>.parquet  (from phase1_ingest.py)
Outputs: output/features/<yyyymmdd>.parquet
         output/features/feature_list.txt
         reports/phase2_daily.csv

Features (~130+, organised in groups):
 G1  Level-1 microstructure (mid, spread, imbalance, microprice, VWAPs)
 G2  Order Flow Imbalance (Cont-Kukanov-Stoikov) multi-level, multi-horizon
 G3  Book pressure vector (BPV), slope, curvature
 G4  Linear algebra: rolling PCA on BPV, Mahalanobis anomaly distance
 G5  Temporal derivatives / integrals
 G6  Rolling z-scores / realised vol / mean-reversion
 G7  Queue dynamics, cancel/add proxies, book walls
 G8  Time / session / staleness features
 G9  Kalman-filter state (mid, spread, innovation)
 G10 REGIME ENGINE

FIXES over original:
  - g4_pca: batched computation every 30s (~20x faster)
  - g9_kalman: steady-state gain after convergence (~50x faster)
  - All features computed per-day (no cross-day leakage)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---------------------------------------------------------------------------
# 0. Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CLEAN_DIR    = ROOT / "output" / "clean"
FEAT_DIR     = ROOT / "output" / "features"
REPORTS_DIR  = ROOT / "reports"
SPLITS_FILE  = ROOT / "output" / "splits.json"
TICK = 0.05
WARMUP = 360

for d in (FEAT_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def roll(series: pd.Series, window: int, min_periods: int = None):
    if min_periods is None:
        min_periods = max(window // 4, 5)
    return series.rolling(window=window, min_periods=min_periods)


# ---------------------------------------------------------------------------
# G1: Level-1 microstructure
# ---------------------------------------------------------------------------
def g1_micro(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    bp, ap, bq, aq = df["b1_p"], df["a1_p"], df["b1_q"], df["a1_q"]
    out["mid"]       = (bp + ap) / 2
    out["sprd"]      = ap - bp
    out["sprd_t"]    = (out["sprd"] / TICK).round()
    out["spread_bps"] = out["sprd"] / out["mid"] * 10000
    out["micro"]     = (bp * aq + ap * bq) / (bq + aq).replace(0, np.nan)
    out["micro_dev"] = (out["micro"] - out["mid"]) / out["sprd"].replace(0, np.nan)

    tot_bq = sum(df[f"b{i+1}_q"] for i in range(5))
    tot_aq = sum(df[f"a{i+1}_q"] for i in range(5))
    wbid = sum(df[f"b{i+1}_p"] * df[f"b{i+1}_q"] for i in range(5)) / tot_bq.replace(0, np.nan)
    wask = sum(df[f"a{i+1}_p"] * df[f"a{i+1}_q"] for i in range(5)) / tot_aq.replace(0, np.nan)
    out["b_vwap"] = wbid
    out["a_vwap"] = wask
    out["wmid"]   = (wbid * tot_aq + wask * tot_bq) / (tot_bq + tot_aq).replace(0, np.nan)
    out["wmid_dev"] = (out["wmid"] - out["mid"]) / out["sprd"].replace(0, np.nan)

    for i in range(5):
        bq_i = df[f"b{i+1}_q"]; aq_i = df[f"a{i+1}_q"]
        out[f"imb{i+1}"] = (bq_i - aq_i) / (bq_i + aq_i).replace(0, np.nan)
    out["tot_bq"] = tot_bq
    out["tot_aq"] = tot_aq
    out["tot_dp"] = tot_bq + tot_aq
    out["depth_imb"] = (tot_bq - tot_aq) / (tot_bq + tot_aq).replace(0, np.nan)

    out["ltp_side"] = np.where(df["ltp"] >= ap, 1, np.where(df["ltp"] <= bp, -1, 0))
    out["ltp_dev"]  = (df["ltp"] - out["mid"]) / out["sprd"].replace(0, np.nan)
    out["eff_hs_t"] = (df["ltp"] - out["mid"]).abs() / TICK

    return out


# ---------------------------------------------------------------------------
# G2: Order Flow Imbalance (Cont-Kukanov-Stoikov)
# ---------------------------------------------------------------------------
def ofi_level(bp, bq, ap, aq) -> pd.Series:
    bp_p = bp.shift(1); bq_p = bq.shift(1)
    ap_p = ap.shift(1); aq_p = aq.shift(1)
    bid_part = (bp >= bp_p).astype(int) * bq - (bp <= bp_p).astype(int) * bq_p
    ask_part = (ap <= ap_p).astype(int) * aq - (ap >= ap_p).astype(int) * aq_p
    return (bid_part - ask_part).fillna(0)


def g2_ofi(df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    ofi1 = ofi_level(df["b1_p"], df["b1_q"], df["a1_p"], df["a1_q"])
    out["ofi1"] = ofi1
    for H in (3, 5, 10, 30, 60):
        out[f"ofi1_sum{H}"]  = ofi1.rolling(H, min_periods=max(2, H // 4)).sum()
        out[f"ofi1_mean{H}"] = ofi1.rolling(H, min_periods=max(2, H // 4)).mean()
    for i in range(2, 6):
        o = ofi_level(df[f"b{i}_p"], df[f"b{i}_q"], df[f"a{i}_p"], df[f"a{i}_q"])
        out[f"ofi{i}"] = o
    weights = [1.0, 0.6, 0.35, 0.18, 0.08]
    ofi_all = sum(out[f"ofi{i+1}"] * w for i, w in enumerate(weights))
    out["ofi_w"] = ofi_all
    for H in (10, 30, 60):
        out[f"ofi_w_sum{H}"] = ofi_all.rolling(H, min_periods=10).sum()
    ltp_diff = df["ltp"].diff()
    out["trade_imb"] = np.sign(ltp_diff).fillna(0)
    return out


# ---------------------------------------------------------------------------
# G3: Book slope & curvature
# ---------------------------------------------------------------------------
def g3_book_shape(df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    bid_offsets = np.array([(df["b1_p"] - df[f"b{i+1}_p"]) / TICK for i in range(5)]).T
    ask_offsets = np.array([(df[f"a{i+1}_p"] - df["a1_p"]) / TICK for i in range(5)]).T
    bid_q = np.array([df[f"b{i+1}_q"] for i in range(5)]).T
    ask_q = np.array([df[f"a{i+1}_q"] for i in range(5)]).T
    bid_cum = np.cumsum(bid_q, axis=1)
    ask_cum = np.cumsum(ask_q, axis=1)

    def _slope(x, y):
        xm = x - x.mean(axis=1, keepdims=True)
        ym = y - y.mean(axis=1, keepdims=True)
        s = (xm * ym).sum(axis=1) / ((xm * xm).sum(axis=1) + 1e-9)
        return s

    out["bid_slope"] = _slope(bid_offsets, bid_cum)
    out["ask_slope"] = _slope(ask_offsets, ask_cum)
    out["slope_imb"] = (out["bid_slope"] - out["ask_slope"]) / (out["bid_slope"] + out["ask_slope"] + 1e-9)
    out["bid_curve"] = bid_cum[:, 2] - 2 * bid_cum[:, 1] + bid_cum[:, 0]
    out["ask_curve"] = ask_cum[:, 2] - 2 * ask_cum[:, 1] + ask_cum[:, 0]

    for n in (1, 2, 3, 5):
        bq_near = np.zeros(len(df)); aq_near = np.zeros(len(df))
        for i in range(5):
            bo = (df["b1_p"] - df[f"b{i+1}_p"]) / TICK
            ao = (df[f"a{i+1}_p"] - df["a1_p"]) / TICK
            bq_near += np.where(bo <= n, df[f"b{i+1}_q"], 0)
            aq_near += np.where(ao <= n, df[f"a{i+1}_q"], 0)
        out[f"bq_near{n}t"] = bq_near
        out[f"aq_near{n}t"] = aq_near
    out["qconc_b3"] = out["bq_near3t"] / out["tot_bq"].replace(0, np.nan)
    out["qconc_a3"] = out["aq_near3t"] / out["tot_aq"].replace(0, np.nan)
    return out


# ---------------------------------------------------------------------------
# G4: Rolling PCA on BPV — FIX: batched (recompute every 30s)
# ---------------------------------------------------------------------------
def g4_pca(df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    PCA_UPDATE_INTERVAL = 30

    bpv = np.column_stack([df[f"b{i+1}_q"].values for i in range(5)] +
                          [df[f"a{i+1}_q"].values for i in range(5)]).astype(float)
    tot = bpv.sum(axis=1, keepdims=True)
    bpv_n = bpv / np.where(tot > 0, tot, np.nan)
    bpv_n = np.nan_to_num(bpv_n, nan=0.0)

    T = len(df); K = 10; W = 300
    pc1 = np.full(T, np.nan); pc2 = np.full(T, np.nan); pc3 = np.full(T, np.nan)
    expl1 = np.full(T, np.nan); maha = np.full(T, np.nan)

    last_eigvecs = None
    last_mu = None
    last_inv = None
    last_expl1 = 0.0

    for t in range(W, T):
        if (t - W) % PCA_UPDATE_INTERVAL == 0 or last_eigvecs is None:
            win = bpv_n[t - W:t]
            mu = win.mean(axis=0)
            c = win - mu
            cov = (c.T @ c) / (W - 1)
            eigvals, eigvecs = np.linalg.eigh(cov + 1e-10 * np.eye(K))
            eigvals = eigvals[::-1]; eigvecs = eigvecs[:, ::-1]
            total_var = eigvals.sum() + 1e-12
            inv = np.linalg.pinv(cov + 1e-6 * np.eye(K), hermitian=True)
            last_eigvecs = eigvecs
            last_mu = mu
            last_inv = inv
            last_expl1 = eigvals[0] / total_var

        v = bpv_n[t] - last_mu
        pc1[t] = v @ last_eigvecs[:, 0]
        pc2[t] = v @ last_eigvecs[:, 1]
        pc3[t] = v @ last_eigvecs[:, 2]
        expl1[t] = last_expl1
        maha[t] = float(np.sqrt(max(v @ last_inv @ v, 0.0)))

    out["bpv_pc1"] = pc1; out["bpv_pc2"] = pc2; out["bpv_pc3"] = pc3
    out["bpv_expl1"] = expl1
    out["bpv_maha"] = maha
    return out


# ---------------------------------------------------------------------------
# G5: Derivatives / integrals
# ---------------------------------------------------------------------------
def g5_calculus(out: pd.DataFrame) -> pd.DataFrame:
    mid = out["mid"]
    sprd = out["sprd"]
    ofi  = out["ofi1"]
    out["ret1"] = mid.pct_change(1)
    for H in (3, 5, 10, 30, 60, 120):
        out[f"ret{H}"] = mid.pct_change(H)
    out["dmid_1"]  = mid.diff(1)
    out["dmid_5"]  = mid.diff(5) / 5
    out["dmid_10"] = mid.diff(10) / 10
    out["ddmid"]   = out["dmid_1"].diff(1)
    out["dsprd_1"] = sprd.diff(1)
    out["dsprd_5"] = sprd.diff(5) / 5
    out["dofi_1"]  = ofi.diff(1)
    out["dofi_5"]  = ofi.diff(5) / 5
    out["dmid_ewm10"] = out["dmid_1"].ewm(span=10, min_periods=5).mean()
    out["ofi_cum"] = ofi.cumsum() / (np.arange(len(ofi)) + 1)
    out["ret_cum120"] = out["ret1"].rolling(120, min_periods=10).sum()
    return out


# ---------------------------------------------------------------------------
# G6: Rolling statistics
# ---------------------------------------------------------------------------
def g6_stats(out: pd.DataFrame) -> pd.DataFrame:
    mid = out["mid"]; sprd = out["sprd"]; ofi = out["ofi1"]
    ret = out["ret1"]
    for H in (10, 30, 60, 120, 300):
        rv = ret.rolling(H, min_periods=max(H // 4, 5)).std() * np.sqrt(22500)
        out[f"rv{H}"] = rv
        h = mid.rolling(H, min_periods=max(H // 4, 5)).max()
        l = mid.rolling(H, min_periods=max(H // 4, 5)).min()
        out[f"rng_park{H}"] = np.sqrt(1 / (4 * np.log(2)) * (np.log(h / l) ** 2)).fillna(0)
    for H in (60, 300):
        mu = mid.rolling(H, min_periods=H // 4).mean(); sd = mid.rolling(H, min_periods=H // 4).std()
        out[f"mid_z{H}"] = (mid - mu) / sd.replace(0, np.nan)
        mu_s = sprd.rolling(H, min_periods=H // 4).median()
        sd_s = sprd.rolling(H, min_periods=H // 4).std()
        out[f"sprd_z{H}"] = (sprd - mu_s) / sd_s.replace(0, np.nan)
        mu_o = ofi.rolling(H, min_periods=H // 4).mean(); sd_o = ofi.rolling(H, min_periods=H // 4).std()
        out[f"ofi_z{H}"] = (ofi - mu_o) / sd_o.replace(0, np.nan)

    def _lag1_autocorr(x, w):
        x = x - x.rolling(w, min_periods=w // 4).mean()
        num = (x * x.shift(1)).rolling(w, min_periods=w // 4).sum()
        den = (x * x).rolling(w, min_periods=w // 4).sum()
        return (num / den.replace(0, np.nan)).fillna(0)

    out["acf1_60"]  = _lag1_autocorr(ret, 60)
    out["acf1_300"] = _lag1_autocorr(ret, 300)
    for H in (60, 300):
        mu_t = out["tot_dp"].rolling(H, min_periods=H // 4).mean()
        sd_t = out["tot_dp"].rolling(H, min_periods=H // 4).std()
        out[f"depth_z{H}"] = (out["tot_dp"] - mu_t) / sd_t.replace(0, np.nan)
    return out


# ---------------------------------------------------------------------------
# G7: Queue / add-cancel dynamics, book walls
# ---------------------------------------------------------------------------
def g7_queue(df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    for side, levels in (("b", [f"b{i+1}_q" for i in range(5)]),
                         ("a", [f"a{i+1}_q" for i in range(5)])):
        adds = 0; cancels = 0
        for col in levels:
            d = df[col].diff()
            adds    += d.clip(lower=0)
            cancels += (-d).clip(lower=0)
        out[f"{side}_add10"] = adds.rolling(10, min_periods=3).sum()
        out[f"{side}_can10"] = cancels.rolling(10, min_periods=3).sum()
    out["add_can_ratio"] = (out["b_add10"] + out["a_add10"]) / (out["b_can10"] + out["a_can10"] + 1)
    out["qratio_b"] = df["b1_q"] / (df["b2_q"].replace(0, np.nan) + 1)
    out["qratio_a"] = df["a1_q"] / (df["a2_q"].replace(0, np.nan) + 1)
    out["bwall_3t"] = np.maximum.reduce([
        np.where((df["b1_p"] - df[f"b{i+1}_p"]) / TICK <= 3, df[f"b{i+1}_q"], 0) for i in range(5)])
    out["awall_3t"] = np.maximum.reduce([
        np.where((df[f"a{i+1}_p"] - df["a1_p"]) / TICK <= 3, df[f"a{i+1}_q"], 0) for i in range(5)])
    out["wall_imb"] = (out["bwall_3t"] - out["awall_3t"]) / (out["bwall_3t"] + out["awall_3t"] + 1)
    return out


# ---------------------------------------------------------------------------
# G8: Time / session / staleness
# ---------------------------------------------------------------------------
def g8_time(df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    t = out.index
    sec_since_open = (t.hour * 3600 + t.minute * 60 + t.second) - (9 * 3600 + 15 * 60)
    sec_to_close   = (15 * 3600 + 30 * 60) - (t.hour * 3600 + t.minute * 60 + t.second)
    total_sess     = (15 * 3600 + 30 * 60) - (9 * 3600 + 15 * 60)
    out["sec_open"]  = sec_since_open
    out["sec_close"] = sec_to_close
    out["tod_sin"]   = np.sin(2 * np.pi * sec_since_open / total_sess)
    out["tod_cos"]   = np.cos(2 * np.pi * sec_since_open / total_sess)

    hm = t.hour * 60 + t.minute
    def bucket(m):
        return np.where(m < 9 * 60 + 30, 0,
               np.where(m < 11 * 60,    1,
               np.where(m < 13 * 60 + 30, 2,
               np.where(m < 15 * 60 + 15, 3, 4))))
    out["sess_bucket"] = bucket(hm)

    out["is_stale"]  = df["is_stale"].fillna(0).astype(int)
    out["depth_age"] = df["depth_age_s"].ffill().fillna(0)
    out["quote_age"] = df["quote_age_s"].ffill().fillna(0)

    d_ticks = df["depth_ticks"].diff().clip(lower=0).fillna(0)
    q_ticks = df["quote_ticks"].diff().clip(lower=0).fillna(0)
    out["depth_rate"] = d_ticks.rolling(30, min_periods=5).mean()
    out["quote_rate"] = q_ticks.rolling(30, min_periods=5).mean()

    out["warmup"] = (np.arange(len(df)) < WARMUP).astype(int)
    return out


# ---------------------------------------------------------------------------
# G9: Kalman filter — FIX: steady-state gain after convergence
# ---------------------------------------------------------------------------
def g9_kalman(out: pd.DataFrame) -> pd.DataFrame:
    mid = out["mid"].values.astype(np.float64)
    sprd = out["sprd"].values.astype(np.float64)
    T = len(mid)
    q = 1e-5
    r_default = float(np.clip(np.nanmedian(sprd) / 2, 0.02, 0.20))

    x = np.zeros(T); p = np.zeros(T); k = np.zeros(T); innov = np.zeros(T)
    x[0] = mid[0]; p[0] = r_default

    # Full recursion for first 100 rows (convergence)
    conv_n = min(100, T)
    for t in range(1, conv_n):
        x_pred = x[t - 1]; p_pred = p[t - 1] + q
        obs_noise = max(sprd[t] / 2, 0.02) if np.isfinite(sprd[t]) else r_default
        K = p_pred / (p_pred + obs_noise)
        innov[t] = mid[t] - x_pred
        x[t] = x_pred + K * innov[t]
        p[t] = (1 - K) * p_pred
        k[t] = K

    # Steady-state gain for the rest
    if T > conv_n:
        K_ss = k[conv_n - 1]
        for t in range(conv_n, T):
            x_pred = x[t - 1]
            obs_noise = max(sprd[t] / 2, 0.02) if np.isfinite(sprd[t]) else r_default
            K_t = K_ss * (1.0 + 0.1 * (obs_noise - r_default) / (r_default + 1e-9))
            K_t = float(np.clip(K_t, 0.01, 0.95))
            innov[t] = mid[t] - x_pred
            x[t] = x_pred + K_t * innov[t]
            k[t] = K_t
            p[t] = (1 - K_t) * p[t - 1] + q

    out["kf_mid"] = x
    out["kf_innov"] = innov
    out["kf_innov_z"] = innov / (pd.Series(innov).rolling(300, min_periods=30).std() + 1e-9).values
    out["kf_K"] = k
    out["kf_sprd"] = pd.Series(sprd).ewm(span=300, min_periods=30).mean().values
    out["sprd_dev_kf"] = (out["sprd"] - out["kf_sprd"]) / TICK
    return out


# ---------------------------------------------------------------------------
# G10: REGIME ENGINE
# ---------------------------------------------------------------------------
def g10_regimes(out: pd.DataFrame) -> pd.DataFrame:
    rv = out["rv60"]
    rv_q = rv.rolling(600, min_periods=120).rank(pct=True)
    out["vol_regime"] = np.where(rv_q < 0.33, 0,
                        np.where(rv_q < 0.75, 1, 2)).astype(np.int8)

    sp = out["sprd"] / TICK
    med_sp = sp.rolling(300, min_periods=60).median()
    out["sprd_regime"] = np.where(sp <= med_sp + 1, 0,
                         np.where(sp <= med_sp + 4, 1, 2)).astype(np.int8)

    td = out["tot_dp"]
    td_q = td.rolling(600, min_periods=120).rank(pct=True)
    out["vol_liq_regime"] = np.where(td_q < 0.33, 0,
                            np.where(td_q < 0.75, 1, 2)).astype(np.int8)

    acf = out["acf1_60"]
    out["trend_regime"] = np.where(acf > 0.05, 1,
                          np.where(acf < -0.05, -1, 0)).astype(np.int8)

    age = out["depth_age"].rolling(60, min_periods=10).mean()
    out["stale_regime"] = np.where(age < 0.5, 0,
                          np.where(age < 2.0, 1, 2)).astype(np.int8)

    rv60 = out["rv60"]; rv60_q = rv.rolling(600, min_periods=120).rank(pct=True)
    sprd_t = out["sprd"] / TICK
    tick_rate = out["depth_rate"]
    state = np.full(len(out), 2, dtype=np.int8)
    state = np.where((rv60_q < 0.33) & (sprd_t <= 4) & (np.abs(acf) <= 0.05), 0, state)
    state = np.where((rv60_q < 0.50) & (sprd_t <= 6) & (acf > 0.08), 1, state)
    state = np.where((rv60_q > 0.75) & (sprd_t <= 10) & (np.abs(acf) <= 0.05), 3, state)
    state = np.where((rv60_q > 0.60) & (acf > 0.10), 4, state)
    state = np.where((sprd_t > 12) | (age > 3.0) | (tick_rate < 0.1), 5, state)
    out["mkt_state"] = state.astype(np.int8)

    state_names = {0: "calm_chop", 1: "calm_trend", 2: "normal",
                   3: "vol_chop", 4: "vol_trend", 5: "news/frozen"}
    out["mkt_state_name"] = pd.Categorical(
        [state_names[s] for s in state],
        categories=list(state_names.values()))
    return out


# ---------------------------------------------------------------------------
# Per-day pipeline
# ---------------------------------------------------------------------------
def build_day(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out = out.join(g1_micro(df))
    out = g2_ofi(df, out)
    out = g3_book_shape(df, out)
    out = g4_pca(df, out)
    out = g5_calculus(out)
    out = g6_stats(out)
    out = g7_queue(df, out)
    out = g8_time(df, out)
    out = g9_kalman(out)
    out = g10_regimes(out)
    out["trade_date"] = df["trade_date"].iloc[0] if "trade_date" in df.columns else \
                        pd.Timestamp(out.index[0].date())
    out["warmup"] = out.get("warmup", 0).astype(np.int8)
    out = out.replace([np.inf, -np.inf], np.nan)
    num_cols = out.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        s = out[c]
        if s.notna().sum() < 30: continue
        lo, hi = s.quantile(0.0005), s.quantile(0.9995)
        # FIX: guard against NaN bounds (causes clip to hang in numpy)
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            out[c] = s.clip(lo, hi)
    out = out.copy()  # de-fragment
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("PHASE 2 — FEATURE ENGINEERING")
    print(f"ROOT     : {ROOT}")
    print(f"CLEAN    : {CLEAN_DIR}")
    print(f"FEATURES : {FEAT_DIR}")
    print(f"REPORTS  : {REPORTS_DIR}")
    print("=" * 74)

    files = sorted(CLEAN_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No clean parquets in {CLEAN_DIR}. Run phase1 first.")
    print(f"Found {len(files)} clean daily parquets.\n")

    daily_records = []
    feature_cols = None
    for i, p in enumerate(files, 1):
        date = p.stem
        print(f"[phase2] {date} ({i}/{len(files)}) ... ", end="", flush=True)
        df = pd.read_parquet(p)
        feats = build_day(df)
        feats.to_parquet(FEAT_DIR / f"{date}.parquet", index=True)
        cols = [c for c in feats.columns if c != "trade_date"]
        if feature_cols is None:
            feature_cols = cols
        n_regimes = feats["mkt_state"].value_counts().to_dict()
        regime_frac_calm = n_regimes.get(0, 0) + n_regimes.get(1, 0)
        rec = {
            "date": date, "rows": len(feats), "cols": len(cols),
            "avg_sprd": float(feats["sprd"].mean()),
            "med_sprd_t": float(feats["sprd_t"].median()),
            "rv60_median": float(feats["rv60"].median()),
            "pct_stale_seconds": 100 * float(feats["is_stale"].mean()),
            "avg_abs_ofi1": float(feats["ofi1"].abs().mean()),
            "abs_kf_innov": float(feats["kf_innov"].abs().mean()),
            "frac_calm_regime": 100 * float(regime_frac_calm / len(feats)),
            "frac_newsfrozen": 100 * float(n_regimes.get(5, 0) / len(feats)),
            "t_start": feats.index[0].strftime("%H:%M:%S"),
            "t_end":   feats.index[-1].strftime("%H:%M:%S"),
            "n_nan_per_row_avg": float(feats.isna().sum(1).mean()),
        }
        daily_records.append(rec)
        print(f"rows={rec['rows']:6d}  cols={rec['cols']:3d}  "
              f"avg_spread={rec['avg_sprd']:.3f}  avg|OFI|={rec['avg_abs_ofi1']:.1f}  "
              f"|kalman_innov|={rec['abs_kf_innov']:.4f}  "
              f"calm%={rec['frac_calm_regime']:.1f}  news/frozen%={rec['frac_newsfrozen']:.2f}")

    exclude = {"mkt_state_name", "trade_date"}
    numeric_features = [c for c in feature_cols if c not in exclude]
    (ROOT / "output" / "feature_list.txt").write_text("\n".join(numeric_features))
    pd.DataFrame(daily_records).to_csv(REPORTS_DIR / "phase2_daily.csv", index=False)

    print("\n--- Regime distribution (percent of seconds per day) ---")
    rows = []
    for p in files:
        f = pd.read_parquet(FEAT_DIR / p.name)
        vc = f["mkt_state"].value_counts(normalize=True).to_dict()
        rows.append({"date": p.stem,
                     **{f"state{k}": round(vc.get(k, 0) * 100, 1) for k in range(6)}})
    rdf = pd.DataFrame(rows).set_index("date")
    rdf.columns = ["calm_chop", "calm_trend", "normal", "vol_chop", "vol_trend", "news/frozen"]
    print(rdf.to_string())

    print(f"\nSaved: {FEAT_DIR}/<yyyymmdd>.parquet x {len(files)}")
    print(f"       {REPORTS_DIR / 'phase2_daily.csv'}")
    print(f"       output/feature_list.txt  ({len(numeric_features)} numeric features)")
    print("\nPhase 2 complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] {e}", file=sys.stderr)
        raise
