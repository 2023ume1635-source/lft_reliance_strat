#!/usr/bin/env python3
"""
================================================================================
LIVE TRADING BOT — RELIND 1-second scalping (FIXED v3)
================================================================================

FIXES applied over v2:
  v2 fixes (all still present):
    1. sys.path.insert BEFORE load_model
    2. NaN mid guard on time-stop exit
    3. R_per_share estimated from buffer
    4. other_ticks uses snap["other_tick_count"]
    5. engine.decide() receives only tail(1)
    6. add_snapshot uses passed ts
    7. Equity log flushes every row
    8. import sys at top
    9. EOD close guards against NaN
   10. Independent header logic
   11. state.on_other() for unrecognised ticks
   12. MIN_HOLD_S = 3
   13. Websocket watchdog
   14. Unified COOLDOWN_S = 15
   15. No transaction costs

  v3 NEW fixes (capital compounding):
   16. new_day() NO LONGER resets capital to INITIAL_CAPITAL.
       Capital compounds across days: if day 1 ends at Rs 10,01,000,
       day 2 starts at Rs 10,01,000.
   17. start_of_day_cap = current capital at day boundary (for kill-switch reference).
   18. resume_from_logs() sets start_of_day_cap = resumed capital (not INITIAL_CAPITAL).
   19. resume kill-check uses intra_pnl (not capital vs INITIAL_CAPITAL).

  v4 CRITICAL fix (entry price was 0):
   20. build_day() output does NOT contain raw book columns (b1_p, a1_p, b1_q, a1_q).
       Without explicitly copying them from the input frame, decide() reads
       b1_p=0.0 → entry_price=0 → target=0.5 → instantly "hit" by mid≈1267
       → every trade shows fake profit = exit_price × qty.
       FIX: copy raw book columns from feats_frame into feats_full after build_day().

Usage:
    python live_trading_bot.py
    python live_trading_bot.py --retrain
    python live_trading_bot.py --duration 600
"""

from __future__ import annotations
import os, sys, csv, json, time, pickle, signal, threading, logging, argparse
import warnings
from pathlib import Path
from copy import deepcopy
from datetime import datetime, date as date_cls, time as dtime
from collections import deque
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
INPUT_DIR    = ROOT / "input"
CLEAN_DIR    = ROOT / "output" / "clean"
FEAT_DIR     = ROOT / "output" / "features"
LABEL_DIR    = ROOT / "output" / "labels"
PRED_DIR     = ROOT / "artifacts" / "preds"
MODEL_DIR    = ROOT / "artifacts" / "models"
SPLIT_F      = ROOT / "output" / "splits.json"
LIVE_DIR     = ROOT / "live_output"
LIVE_CSV_DIR = LIVE_DIR / "raw"
LIVE_LOG_DIR = LIVE_DIR / "logs"
for d in (LIVE_DIR, LIVE_CSV_DIR, LIVE_LOG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

TICK = 0.05
INITIAL_CAPITAL = 10_00_000.0
MAX_DAILY_LOSS = 0.02 * INITIAL_CAPITAL   # Rs 20,000 absolute daily loss limit
COOLDOWN_S = 15
MIN_HOLD_S = 3
MAX_NOTIONAL_FRAC = 0.25
MAX_QTY = 400
MIN_QTY = 10
CLOSE_NO_NEW_AFTER = dtime(15, 29, 0)
CLOSE_CANCEL_PENDING_AT = dtime(15, 29, 45)
RING_BUFFER_SEC = 700
WARMUP_S = 360

STOCK_CODE = "RELIND"
EXCHANGE_CODE = "NSE"
PRODUCT_TYPE = "cash"

BID_LEVELS = 5; ASK_LEVELS = 5
HEADER_LIVE = ["ts"]
for i in range(1, BID_LEVELS + 1): HEADER_LIVE += [f"b{i}_o", f"b{i}_q", f"b{i}_p"]
for i in range(1, ASK_LEVELS + 1): HEADER_LIVE += [f"a{i}_p", f"a{i}_q", f"a{i}_o"]
HEADER_LIVE += ["ltp", "recv_time", "depth_time", "quote_time",
                "depth_age_s", "quote_age_s", "is_stale", "depth_ticks", "quote_ticks",
                "other_ticks", "exch_time"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("live")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="RELIND live scalper (fixed v3)")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--paper", action="store_true",
                   help="(placeholder) Send paper orders via Breeze")
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--stale-threshold", type=float, default=8.0)
    p.add_argument("--model-path", type=str, default=str(MODEL_DIR / "phase4_models_live.pkl"))
    return p.parse_args()

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def retrain_all_days() -> dict:
    log.info("Retraining production model on ALL historical days ...")
    import lightgbm as lgb
    from phase4_model import (LGB_CLF_PARAMS, LGB_REG_PARAMS, LGB_BIN_PARAMS,
                              N_BOOST_CLF, N_BOOST_REG, N_BOOST_BIN, LR_STOP,
                              DROP_LEAK, load_feature_list, load_all_labels)
    feats_all = load_feature_list()
    feats = [c for c in feats_all if c not in DROP_LEAK]
    big = load_all_labels(feats)
    cat_feats = [c for c in ("vol_regime", "sprd_regime", "vol_liq_regime", "trend_regime",
                             "stale_regime", "mkt_state", "sess_bucket") if c in feats]
    for c in feats:
        if c not in big.columns: big[c] = 0.0
        if c not in cat_feats:
            big[c] = pd.to_numeric(big[c], errors="coerce")
    big[feats] = big[feats].replace([np.inf, -np.inf], np.nan)
    big[feats] = big.groupby("_file", group_keys=False)[feats].ffill().fillna(0.0)
    for c in cat_feats: big[c] = big[c].fillna(0).astype(np.int32)
    na_edge = big["y_edge_30"].isna()
    if na_edge.any():
        big.loc[na_edge, "y_valid"] = False
        big.loc[na_edge, "y_edge_30"] = 0.0
    for c in ("y_buyfill", "y_sellfill"): big[c] = big[c].fillna(0).astype(np.int8)
    big["y_take_dir"] = big["y_take_dir"].fillna(0).astype(np.int8)

    files_sorted = sorted(big["_file"].unique())
    last_day = files_sorted[-1]
    tr_mask = (big["_file"] != last_day) & big["y_valid"].astype(bool)
    va_mask = (big["_file"] == last_day) & big["y_valid"].astype(bool)
    Xtr = big.loc[tr_mask, feats]; Xv = big.loc[va_mask, feats]
    wtr = np.ones(Xtr.shape[0], dtype=np.float32)
    wv  = np.ones(Xv.shape[0], dtype=np.float32)

    log.info(f"  training: {len(Xtr):,} rows, val ({last_day}): {len(Xv):,}")
    dtr_clf = lgb.Dataset(Xtr, label=big.loc[tr_mask, "y_take_dir"] + 1, weight=wtr, free_raw_data=False)
    dva_clf = lgb.Dataset(Xv, label=big.loc[va_mask, "y_take_dir"] + 1, weight=wv, reference=dtr_clf, free_raw_data=False)
    b_clf = lgb.train(LGB_CLF_PARAMS, dtr_clf, num_boost_round=N_BOOST_CLF + 200,
                      valid_sets=[dtr_clf, dva_clf], valid_names=["tr", "va"],
                      callbacks=[lgb.early_stopping(LR_STOP, verbose=False), lgb.log_evaluation(0)])
    dtr_reg = lgb.Dataset(Xtr, label=big.loc[tr_mask, "y_edge_30"], weight=wtr, free_raw_data=False)
    dva_reg = lgb.Dataset(Xv, label=big.loc[va_mask, "y_edge_30"], weight=wv, reference=dtr_reg, free_raw_data=False)
    b_reg = lgb.train(LGB_REG_PARAMS, dtr_reg, num_boost_round=N_BOOST_REG + 200,
                      valid_sets=[dtr_reg, dva_reg], valid_names=["tr", "va"],
                      callbacks=[lgb.early_stopping(LR_STOP, verbose=False), lgb.log_evaluation(0)])
    dtr_buy = lgb.Dataset(Xtr, label=big.loc[tr_mask, "y_buyfill"], weight=wtr, free_raw_data=False)
    dva_buy = lgb.Dataset(Xv, label=big.loc[va_mask, "y_buyfill"], weight=wv, reference=dtr_buy, free_raw_data=False)
    b_buy = lgb.train(LGB_BIN_PARAMS, dtr_buy, num_boost_round=N_BOOST_BIN + 100,
                      valid_sets=[dtr_buy, dva_buy], valid_names=["tr", "va"],
                      callbacks=[lgb.early_stopping(LR_STOP, verbose=False), lgb.log_evaluation(0)])
    dtr_sell = lgb.Dataset(Xtr, label=big.loc[tr_mask, "y_sellfill"], weight=wtr, free_raw_data=False)
    dva_sell = lgb.Dataset(Xv, label=big.loc[va_mask, "y_sellfill"], weight=wv, reference=dtr_sell, free_raw_data=False)
    b_sell = lgb.train(LGB_BIN_PARAMS, dtr_sell, num_boost_round=N_BOOST_BIN + 100,
                       valid_sets=[dtr_sell, dva_sell], valid_names=["tr", "va"],
                       callbacks=[lgb.early_stopping(LR_STOP, verbose=False), lgb.log_evaluation(0)])

    feats_final = list(feats)
    log.info(f"  model trained. features: {len(feats_final)}")
    artifacts = dict(feats_final=feats_final, b_clf=b_clf, b_reg=b_reg, b_buy=b_buy, b_sell=b_sell)
    with open(MODEL_DIR / "phase4_models_live.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    return artifacts


def load_model(args) -> dict:
    p = Path(args.model_path)
    if args.retrain or not p.exists():
        return retrain_all_days()
    log.info(f"Loading live model from {p}")
    with open(p, "rb") as f:
        art = pickle.load(f)
    try:
        n_model = art["b_clf"].num_feature()
        if len(art["feats_final"]) != n_model:
            log.warning("Model feature count mismatch; retraining.")
            return retrain_all_days()
    except Exception as e:
        log.warning(f"Could not verify model ({e}); retraining.")
        return retrain_all_days()
    return art

# ---------------------------------------------------------------------------
# Tau table
# ---------------------------------------------------------------------------
def load_tau_table() -> Dict[int, dict]:
    cfg_path = ROOT / "reports" / "phase5_config.json"
    if not cfg_path.exists():
        log.warning("phase5_config.json not found; using conservative defaults")
        return {0: dict(tau_passive=0.005, tau_t=0.025, edge_floor=0.10, H=20, sprd_cap_t=4),
                1: dict(tau_passive=0.006, tau_t=0.030, edge_floor=0.10, H=25, sprd_cap_t=4),
                2: dict(tau_passive=0.008, tau_t=0.040, edge_floor=0.10, H=30, sprd_cap_t=5),
                3: dict(tau_passive=0.010, tau_t=0.060, edge_floor=0.15, H=45, sprd_cap_t=8),
                4: dict(tau_passive=9.0, tau_t=9.0, edge_floor=9.0, H=0, sprd_cap_t=0),
                5: dict(tau_passive=9.0, tau_t=9.0, edge_floor=9.0, H=0, sprd_cap_t=0)}
    with open(cfg_path) as f: cfg = json.load(f)
    rt = cfg["regime_tau"]
    out = {}
    for k, v in rt.items():
        out[int(k)] = dict(tau_passive=v["tau_passive"], tau_t=v["tau_take"],
                           edge_floor=v["edge_floor"], H=v["H"], sprd_cap_t=v["sprd_cap_t"],
                           passive_discount=v.get("passive_discount", 1.0),
                           take_mult=v.get("take_mult", 3.0))
    return out

# ---------------------------------------------------------------------------
# Breeze helpers
# ---------------------------------------------------------------------------
def _safe_float(x, default=np.nan):
    try: return float(x)
    except Exception: return default

def _depth_objs(depth):
    if depth is None: return []
    if isinstance(depth, dict): return [depth]
    if isinstance(depth, list): return [d for d in depth if isinstance(d, dict)]
    return []

def _depth_get(depth, key, level=None):
    objs = _depth_objs(depth)
    for d in objs:
        if key in d and d.get(key) not in (None, ""):
            return d.get(key)
    if level is not None and 0 <= level - 1 < len(objs):
        d = objs[level - 1]
        if key in d and d.get(key) not in (None, ""):
            return d.get(key)
    return ""

def _build_depth_row(depth):
    row = []
    for i in range(1, BID_LEVELS + 1):
        row.append(_depth_get(depth, f"BuyNoOfOrders-{i}", i))
        row.append(_depth_get(depth, f"BestBuyQty-{i}", i))
        row.append(_depth_get(depth, f"BestBuyRate-{i}", i))
    for i in range(1, ASK_LEVELS + 1):
        row.append(_depth_get(depth, f"BestSellRate-{i}", i))
        row.append(_depth_get(depth, f"BestSellQty-{i}", i))
        row.append(_depth_get(depth, f"SellNoOfOrders-{i}", i))
    return row

def _extract_ltp(tick, cur=None):
    for k in ("last", "ltp", "LTP", "last_price", "lastPrice", "close"):
        v = tick.get(k)
        if v not in (None, ""): return v
    return cur if cur is not None else ""

def _extract_exchange_time(tick):
    for k in ("exchange_time", "exchangeTime", "time", "datetime", "ltt",
              "last_trade_time", "lastTradeTime", "ltt_time", "timestamp"):
        v = tick.get(k)
        if v not in (None, ""): return str(v)
    return ""

# ---------------------------------------------------------------------------
# Market state
# ---------------------------------------------------------------------------
class MarketState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_depth: Optional[List[dict]] = None
        self.latest_ltp: Optional[float] = None
        self.latest_exchange_time: str = ""
        self.last_depth_dt: Optional[datetime] = None
        self.last_quote_dt: Optional[datetime] = None
        self.last_any_dt: Optional[datetime] = None
        self.depth_tick_count = 0
        self.quote_tick_count = 0
        self.other_tick_count = 0

    def on_quote(self, ltp, exchange_time=""):
        with self.lock:
            self.last_any_dt = datetime.now()
            self.last_quote_dt = self.last_any_dt
            if ltp not in (None, ""): self.latest_ltp = _safe_float(ltp)
            if exchange_time: self.latest_exchange_time = str(exchange_time)
            self.quote_tick_count += 1

    def on_depth(self, depth, ltp=None, exchange_time=""):
        with self.lock:
            self.last_any_dt = datetime.now()
            self.last_depth_dt = self.last_any_dt
            self.latest_depth = deepcopy(_depth_objs(depth))
            if ltp not in (None, ""): self.latest_ltp = _safe_float(ltp)
            if exchange_time: self.latest_exchange_time = str(exchange_time)
            self.depth_tick_count += 1

    def on_other(self):
        with self.lock:
            self.last_any_dt = datetime.now()
            self.other_tick_count += 1

    def snapshot(self) -> dict:
        with self.lock:
            return dict(latest_depth=deepcopy(self.latest_depth),
                        latest_ltp=self.latest_ltp,
                        latest_exchange_time=self.latest_exchange_time,
                        last_depth_dt=self.last_depth_dt,
                        last_quote_dt=self.last_quote_dt,
                        last_any_dt=self.last_any_dt,
                        depth_tick_count=self.depth_tick_count,
                        quote_tick_count=self.quote_tick_count,
                        other_tick_count=self.other_tick_count)

# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------
class LiveFeatureBuffer:
    def __init__(self, stale_thr: float):
        self.stale_thr = stale_thr
        self.buf: deque = deque(maxlen=RING_BUFFER_SEC)

    def add_snapshot(self, snap: dict, ts: datetime):
        depth = snap["latest_depth"]
        depth_row = _build_depth_row(depth)
        def f(x): return _safe_float(x, np.nan)
        row = {"ts": ts}
        for i in range(BID_LEVELS):
            o = depth_row[i * 3]; q = depth_row[i * 3 + 1]; p = depth_row[i * 3 + 2]
            row[f"b{i+1}_o"] = f(o); row[f"b{i+1}_q"] = f(q); row[f"b{i+1}_p"] = f(p)
        off = BID_LEVELS * 3
        for i in range(ASK_LEVELS):
            p = depth_row[off + i * 3]; q = depth_row[off + i * 3 + 1]; o = depth_row[off + i * 3 + 2]
            row[f"a{i+1}_p"] = f(p); row[f"a{i+1}_q"] = f(q); row[f"a{i+1}_o"] = f(o)
        row["ltp"] = _safe_float(snap["latest_ltp"], np.nan)
        ddt = snap["last_depth_dt"]; qdt = snap["last_quote_dt"]
        depth_age = (ts - ddt).total_seconds() if ddt else 99.0
        quote_age = (ts - qdt).total_seconds() if qdt else 99.0
        row["recv_time"] = ts.strftime("%H:%M:%S.%f")[:-3]
        row["depth_time"] = ddt.strftime("%H:%M:%S.%f")[:-3] if ddt else ""
        row["quote_time"] = qdt.strftime("%H:%M:%S.%f")[:-3] if qdt else ""
        row["depth_age_s"] = round(depth_age, 3)
        row["quote_age_s"] = round(quote_age, 3)
        row["is_stale"] = int(depth_age > self.stale_thr)
        row["depth_ticks"] = int(snap["depth_tick_count"])
        row["quote_ticks"] = int(snap["quote_tick_count"])
        row["other_ticks"] = int(snap["other_tick_count"])
        row["exch_time"] = snap["latest_exchange_time"] or ""
        self.buf.append(row)

    def to_phase1_frame(self) -> pd.DataFrame:
        if len(self.buf) < 30: return pd.DataFrame()
        df = pd.DataFrame(list(self.buf)).set_index("ts").sort_index()
        df["trade_date"] = pd.Timestamp(df.index[-1].date())
        for c in [f"b{i}_p" for i in range(1, 6)] + [f"b{i}_q" for i in range(1, 6)] + \
                 [f"b{i}_o" for i in range(1, 6)] + [f"a{i}_p" for i in range(1, 6)] + \
                 [f"a{i}_q" for i in range(1, 6)] + [f"a{i}_o" for i in range(1, 6)] + ["ltp"]:
            if c in df.columns: df[c] = df[c].ffill().bfill()
        return df

# ---------------------------------------------------------------------------
# Signal engine
# ---------------------------------------------------------------------------
class LiveSignalEngine:
    def __init__(self, artifacts: dict, tau_table: dict):
        self.feats = artifacts["feats_final"]
        self.b_clf = artifacts["b_clf"]; self.b_reg = artifacts["b_reg"]
        self.b_buy = artifacts["b_buy"]; self.b_sell = artifacts["b_sell"]
        self.tau = tau_table
        self.per_trade_budget = 500.0
        self.p_fill_min = 0.15
        self.max_pct_depth = 0.40

    def predict_last(self, feat_row: pd.DataFrame):
        cols = self.feats
        X = feat_row.copy()
        for c in cols:
            if c not in X.columns: X[c] = 0.0
        X = X[cols].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p = self.b_clf.predict(X)[0]
        e = float(self.b_reg.predict(X)[0])
        pb = float(self.b_buy.predict(X)[0])
        ps = float(self.b_sell.predict(X)[0])
        ms = int(feat_row["mkt_state"].iloc[-1]) if "mkt_state" in feat_row.columns else 2
        return float(p[2]), float(p[0]), e, pb, ps, ms

    def estimate_R(self, feat_row: pd.DataFrame) -> float:
        if "mid" not in feat_row.columns or len(feat_row) < 60:
            return 0.45
        mid = feat_row["mid"].values
        if len(mid) > 30:
            moves = np.abs(mid[30:] - mid[:-30])
            moves = moves[np.isfinite(moves)]
            if len(moves) > 20:
                return float(np.clip(np.percentile(moves, 80), 0.15, 2.0))
        return 0.45

    def decide(self, feats_last: pd.DataFrame) -> dict:
        p_l, p_s, edge, fb, fs, regime = self.predict_last(feats_last)
        raw = p_l - p_s
        sprd = float(feats_last["sprd"].iloc[-1]) if "sprd" in feats_last.columns else 0.10
        R = self.estimate_R(feats_last)
        bp = float(feats_last["b1_p"].iloc[-1]); ap = float(feats_last["a1_p"].iloc[-1])
        bq = float(feats_last["b1_q"].iloc[-1]); aq = float(feats_last["a1_q"].iloc[-1])
        is_stale = int(feats_last["is_stale"].iloc[-1]) if "is_stale" in feats_last.columns else 1
        warm = int(feats_last["warmup"].iloc[-1]) if "warmup" in feats_last.columns else 1
        out = dict(p_l=p_l, p_s=p_s, raw=raw, edge=edge, fb=fb, fs=fs, regime=regime,
                   sig=0, side=0, size=0, entry=0.0, stop=0.0, tgt=0.0, H=0,
                   reason="", risk_ok=True)
        t = self.tau.get(regime)
        if t is None or regime in (4, 5):
            out["reason"] = f"gate:regime{regime}"; out["risk_ok"] = False; return out
        if is_stale or warm or not np.isfinite(sprd) or sprd <= 0:
            out["reason"] = "gate:stale/warmup"; out["risk_ok"] = False; return out
        sprd_cap = t["sprd_cap_t"] * TICK
        cost = sprd + TICK
        long_t = (raw > t["tau_t"]) and (abs(edge) > cost + t["edge_floor"])
        shrt_t = (raw < -t["tau_t"]) and (abs(edge) > cost + t["edge_floor"])
        long_p = (raw > t["tau_passive"]) and (abs(edge) > t["edge_floor"]) \
                 and (fb >= self.p_fill_min) and (sprd <= sprd_cap)
        shrt_p = (raw < -t["tau_passive"]) and (abs(edge) > t["edge_floor"]) \
                 and (fs >= self.p_fill_min) and (sprd <= sprd_cap)
        if long_t:
            out["sig"] = 2; out["side"] = +1; p_entry = ap; reason = "take_long"
        elif shrt_t:
            out["sig"] = -2; out["side"] = -1; p_entry = bp; reason = "take_short"
        elif long_p:
            out["sig"] = 1; out["side"] = +1; p_entry = bp; reason = "post_long"
        elif shrt_p:
            out["sig"] = -1; out["side"] = -1; p_entry = ap; reason = "post_short"
        else:
            out["reason"] = "no_signal"; return out
        half_s = 0.5 * sprd
        R_i = max(R, 0.25)
        dyn = float(np.clip(1.0 + abs(edge) / 0.3, 0.7, 1.6))
        tgt_d = R_i * dyn
        stp_d = max(half_s + 0.4 * R_i, 0.10)
        n = int(max(1, self.per_trade_budget / stp_d))
        depth = bq if out["side"] > 0 else aq
        if np.isfinite(depth) and depth > 0:
            n = min(n, int(self.max_pct_depth * depth))
        n = min(n, int(MAX_DAILY_LOSS / max(R_i, 0.10)))
        n = int(np.clip(n, 1, 5000))
        # FIX: leverage cap + noise floor
        entry_for_cap = float(p_entry) if np.isfinite(p_entry) else 1.0
        n = min(n, int(MAX_NOTIONAL_FRAC * INITIAL_CAPITAL / max(entry_for_cap, 0.01)))
        n = min(n, MAX_QTY)
        if n < MIN_QTY:
            n = MIN_QTY
        out["size"] = int(n); out["entry"] = float(p_entry)
        if out["side"] > 0:
            out["tgt"] = float(p_entry + tgt_d); out["stop"] = float(p_entry - stp_d)
        else:
            out["tgt"] = float(p_entry - tgt_d); out["stop"] = float(p_entry + stp_d)
        out["H"] = int(t["H"]); out["reason"] = reason
        return out

# ---------------------------------------------------------------------------
# Portfolio — FIX #16,17: capital compounds across days
# ---------------------------------------------------------------------------
class LivePortfolio:
    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.start_of_day_cap = INITIAL_CAPITAL  # reference for kill-switch
        self.pos = self._empty_pos()
        self.cooldown = 0
        self.killed = False
        self.intra_pnl = 0.0
        self.trades_today: List[dict] = []
        self.rng = np.random.default_rng(42)
        self.n_trades = 0
        self.last_valid_mid = np.nan

    def _empty_pos(self):
        return dict(active=False, entered=False, side=0, qty=0, entry=0.0,
                    stop=0.0, tgt=0.0, entry_i=None, entry_t=None, H_left=0,
                    H_horizon=0, passive=False, fill_p=0.0, posted_at=0,
                    posted_price=0.0, hold_count=0)

    def new_day(self):
        # FIX #16: DO NOT reset capital. It compounds across days.
        # FIX #17: start_of_day_cap = current (compounded) capital.
        #          The kill-switch measures intra_pnl against MAX_DAILY_LOSS
        #          which is an absolute Rs 20,000 — independent of capital level.
        self.start_of_day_cap = self.capital
        self.pos = self._empty_pos()
        self.cooldown = 0
        self.killed = False
        self.intra_pnl = 0.0
        self.trades_today = []
        self.last_valid_mid = np.nan

    def equity(self, mid: float) -> float:
        if not self.pos["active"] or not self.pos["entered"]:
            return self.capital
        m = mid if np.isfinite(mid) else self.last_valid_mid
        if not np.isfinite(m): return self.capital
        if self.pos["side"] > 0: return self.capital + (m - self.pos["entry"]) * self.pos["qty"]
        else: return self.capital + (self.pos["entry"] - m) * self.pos["qty"]

    def tick(self, ts: datetime, sig: dict, bp: float, ap: float,
             bq: float, aq: float, sprd: float, regime: int):
        mid = 0.5 * (bp + ap) if np.isfinite(bp) and np.isfinite(ap) else np.nan
        if np.isfinite(mid): self.last_valid_mid = mid
        else: mid = self.last_valid_mid

        # Kill switch: measures TODAY's loss only (intra_pnl resets each day)
        if not self.killed:
            unreal = 0.0
            if self.pos["active"] and self.pos["entered"] and np.isfinite(mid):
                if self.pos["side"] > 0: unreal = (mid - self.pos["entry"]) * self.pos["qty"]
                else: unreal = (self.pos["entry"] - mid) * self.pos["qty"]
            if self.intra_pnl + unreal <= -MAX_DAILY_LOSS:
                if self.pos["active"] and self.pos["entered"] and np.isfinite(mid):
                    pnl = (mid - self.pos["entry"]) * self.pos["qty"] if self.pos["side"] > 0 \
                          else (self.pos["entry"] - mid) * self.pos["qty"]
                    self._close_trade(ts, float(mid), "kill", pnl)
                self.killed = True
                self.pos = self._empty_pos()
                return
        # FIX: close-time pending cancellation / no-new-orders guard
        if ts.time() >= CLOSE_CANCEL_PENDING_AT:
            if self.pos["active"] and not self.pos["entered"]:
                self.pos = self._empty_pos(); self.cooldown = 2; return
        if ts.time() >= CLOSE_NO_NEW_AFTER:
            if self.pos["active"] and not self.pos["entered"]:
                self.pos = self._empty_pos(); self.cooldown = 2; return
            # Block new signals after 15:29:00
            sig["sig"] = 0
            sig["reason"] = "close_guard_no_new"
            sig["risk_ok"] = False

        if self.cooldown > 0:
            self.cooldown -= 1; return
        if self.killed: return

        # Passive order awaiting fill
        if self.pos["active"] and not self.pos["entered"]:
            self.pos["H_left"] -= 1
            filled = False
            side = self.pos["side"]
            if np.isfinite(ap) and np.isfinite(bp):
                if side > 0 and ap <= self.pos["posted_price"]: filled = True
                elif side < 0 and bp >= self.pos["posted_price"]: filled = True
                else:
                    fp = float(np.clip(self.pos["fill_p"], 0.02, 0.95))
                    Hh = max(int(self.pos["H_horizon"]), 1)
                    p_per_s = 1.0 - (1.0 - fp) ** (1.0 / Hh)
                    if self.rng.random() < p_per_s: filled = True
                if side > 0 and bp < self.pos["posted_price"] - 2 * TICK:
                    self.pos = self._empty_pos(); self.cooldown = 2; return
                if side < 0 and ap > self.pos["posted_price"] + 2 * TICK:
                    self.pos = self._empty_pos(); self.cooldown = 2; return
            if self.pos["H_left"] <= 0 and not filled:
                self.pos = self._empty_pos(); self.cooldown = 2; return
            if filled:
                self.pos["entered"] = True
                self.pos["entry"] = float(self.pos["posted_price"])
                self.pos["entry_t"] = ts
                self.pos["H_left"] = int(self.pos["H_horizon"])
                self.pos["hold_count"] = 0
                avail = bq if side > 0 else aq
                if np.isfinite(avail) and avail > 0:
                    self.pos["qty"] = min(self.pos["qty"], int(avail))
            return

        # Position entered: evaluate exits
        if self.pos["active"] and self.pos["entered"]:
            if not np.isfinite(mid): return
            self.pos["hold_count"] += 1
            exit_reason = None; exit_price = None
            if self.pos["hold_count"] > MIN_HOLD_S:
                if self.pos["side"] > 0:
                    if mid <= self.pos["stop"]: exit_reason = "stop"; exit_price = self.pos["stop"]
                    elif mid >= self.pos["tgt"]: exit_reason = "target"; exit_price = self.pos["tgt"]
                else:
                    if mid >= self.pos["stop"]: exit_reason = "stop"; exit_price = self.pos["stop"]
                    elif mid <= self.pos["tgt"]: exit_reason = "target"; exit_price = self.pos["tgt"]
            if exit_reason is None and sig["sig"] != 0 and np.sign(sig["sig"]) == -self.pos["side"]:
                exit_reason = "flip"; exit_price = float(mid)
            self.pos["H_left"] -= 1
            if exit_reason is None and self.pos["H_left"] <= 0:
                exit_reason = "time"; exit_price = float(mid)
            if exit_reason is not None:
                if self.pos["side"] > 0: pnl = (exit_price - self.pos["entry"]) * self.pos["qty"]
                else: pnl = (self.pos["entry"] - exit_price) * self.pos["qty"]
                self._close_trade(ts, float(exit_price), exit_reason, float(pnl))
            return

        # Flat: evaluate new signal
        if sig["sig"] == 0 or not sig["risk_ok"]: return
        side = int(sig["side"]); passive = abs(int(sig["sig"])) == 1
        if passive:
            self.pos = dict(active=True, entered=False, side=side, qty=int(sig["size"]),
                            entry=0.0, stop=float(sig["stop"]), tgt=float(sig["tgt"]),
                            entry_i=None, entry_t=None, H_left=int(sig["H"]),
                            H_horizon=int(sig["H"]), passive=True,
                            fill_p=sig["fb"] if side > 0 else sig["fs"],
                            posted_at=ts, posted_price=float(sig["entry"]), hold_count=0)
        else:
            p_entry = float(ap if side > 0 else bp) if np.isfinite(ap) and np.isfinite(bp) else float(sig["entry"])
            self.pos = dict(active=True, entered=True, side=side, qty=int(sig["size"]),
                            entry=p_entry, stop=float(sig["stop"]), tgt=float(sig["tgt"]),
                            entry_i=ts, entry_t=ts, H_left=int(sig["H"]),
                            H_horizon=int(sig["H"]), passive=False, fill_p=1.0,
                            posted_at=ts, posted_price=p_entry, hold_count=0)

    def _close_trade(self, ts, exit_price, reason, pnl):
        t = dict(entry_t=self.pos["entry_t"], exit_t=ts, side=self.pos["side"],
                 qty=self.pos["qty"], entry=float(self.pos["entry"]), exit=float(exit_price),
                 pnl=float(pnl), reason=reason, passive=self.pos["passive"])
        self.trades_today.append(t)
        self.capital += pnl
        self.intra_pnl += pnl
        self.n_trades += 1
        self.pos = self._empty_pos()
        self.cooldown = COOLDOWN_S
        return t

# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------
class LiveCsvLogger:
    def __init__(self, today_str: str):
        self.today = today_str
        self.raw_path = LIVE_CSV_DIR / f"reliance_depth_{today_str}_{datetime.now().strftime('%H%M%S')}.csv"
        self.trade_path = LIVE_LOG_DIR / f"trades_{today_str}.csv"
        self.equity_path = LIVE_LOG_DIR / f"equity_{today_str}.csv"
        self.raw_f = open(self.raw_path, "w", newline="", encoding="utf-8")
        self.raw_w = csv.writer(self.raw_f)
        self.raw_w.writerow(HEADER_LIVE); self.raw_f.flush()
        trade_exists = self.trade_path.exists() and self.trade_path.stat().st_size > 0
        self.trade_f = open(self.trade_path, "a", newline="", encoding="utf-8")
        self.trade_w = csv.writer(self.trade_f)
        if not trade_exists:
            self.trade_w.writerow(["entry_t", "exit_t", "side", "qty", "entry", "exit",
                                   "pnl", "reason", "passive"])
            self.trade_f.flush()
        eq_exists = self.equity_path.exists() and self.equity_path.stat().st_size > 0
        self.eq_f = open(self.equity_path, "a", newline="", encoding="utf-8")
        self.eq_w = csv.writer(self.eq_f)
        if not eq_exists:
            self.eq_w.writerow(["ts", "equity", "position_side", "position_qty",
                                "position_entry", "unrealised", "intra_pnl", "kill"])
            self.eq_f.flush()
        log.info(f"Raw CSV: {self.raw_path}")
        log.info(f"Trade log: {self.trade_path}")
        log.info(f"Equity log: {self.equity_path}")

    def write_raw(self, row_list: list):
        self.raw_w.writerow(row_list); self.raw_f.flush()

    def write_trade(self, t: dict):
        self.trade_w.writerow([
            t["entry_t"].strftime("%Y-%m-%d %H:%M:%S") if t["entry_t"] is not None else "",
            t["exit_t"].strftime("%Y-%m-%d %H:%M:%S"),
            t["side"], t["qty"], round(t["entry"], 3), round(t["exit"], 3),
            round(t["pnl"], 2), t["reason"], int(t["passive"])])
        self.trade_f.flush()

    def write_eq(self, ts, eq, port: LivePortfolio, mid):
        if port.pos["active"] and port.pos["entered"] and np.isfinite(mid):
            side = port.pos["side"]; qty = port.pos["qty"]; entry = port.pos["entry"]
            un = (mid - entry) * qty if side > 0 else (entry - mid) * qty
        else:
            side = 0; qty = 0; entry = 0.0; un = 0.0
        self.eq_w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), round(eq, 2),
                            side, qty, round(entry, 3), round(un, 2),
                            round(port.intra_pnl, 2), int(port.killed)])
        self.eq_f.flush()

    def close(self):
        for f in (self.raw_f, self.trade_f, self.eq_f):
            try: f.flush(); f.close()
            except Exception: pass

# ---------------------------------------------------------------------------
# Tick handler
# ---------------------------------------------------------------------------
def make_on_ticks(state: MarketState):
    def on_ticks(ticks):
        if not isinstance(ticks, dict):
            state.on_other(); return
        if ticks.get("quotes") == "Quotes Data":
            state.on_quote(ltp=_extract_ltp(ticks), exchange_time=_extract_exchange_time(ticks))
            return
        if "depth" in ticks:
            state.on_depth(ticks.get("depth"), ltp=_extract_ltp(ticks),
                           exchange_time=_extract_exchange_time(ticks))
            return
        ltp = _extract_ltp(ticks, None)
        if ltp not in (None, ""):
            state.on_quote(ltp=ltp, exchange_time=_extract_exchange_time(ticks))
        else:
            state.on_other()
    return on_ticks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_rth(ts: datetime) -> bool:
    return dtime(9, 15) <= ts.time() <= dtime(15, 30)

def seconds_since_open(ts: datetime) -> int:
    return int((ts - datetime.combine(ts.date(), dtime(9, 15))).total_seconds())

# ---------------------------------------------------------------------------
# FIX #18,19: Resume with correct compounding logic
# ---------------------------------------------------------------------------
def resume_from_logs(port: LivePortfolio, logger: LiveCsvLogger) -> str:
    trade_p = logger.trade_path; eq_p = logger.equity_path
    if not trade_p.exists() and not eq_p.exists():
        return "fresh start"
    realised_pnl = 0.0; n_trades = 0
    if trade_p.exists():
        try:
            tdf = pd.read_csv(trade_p)
            if "pnl" in tdf.columns and len(tdf) > 0:
                realised_pnl = float(pd.to_numeric(tdf["pnl"], errors="coerce").fillna(0).sum())
                n_trades = int(len(tdf))
        except Exception as e:
            return f"fresh start (trades unreadable: {e})"
    last_eq = INITIAL_CAPITAL; killed = False
    if eq_p.exists():
        try:
            edf = pd.read_csv(eq_p)
            if len(edf) > 0:
                last = edf.iloc[-1]
                last_eq = float(pd.to_numeric(last.get("equity", INITIAL_CAPITAL), errors="coerce") or INITIAL_CAPITAL)
                killed = bool(int(pd.to_numeric(last.get("kill", 0), errors="coerce") or 0))
        except Exception: pass
    # FIX #18: capital = last known equity (compounded from all prior days)
    port.capital = float(last_eq)
    # FIX #18: start_of_day_cap = current capital (what we resumed with today)
    port.start_of_day_cap = port.capital
    # intra_pnl = today's realised so far (from trade log)
    port.intra_pnl = float(realised_pnl)
    port.n_trades = n_trades
    port.killed = bool(killed)
    port.cooldown = 0
    port.pos = port._empty_pos()
    port.trades_today = []
    # FIX #19: kill-check uses intra_pnl (today's loss), NOT capital vs INITIAL
    if port.intra_pnl <= -MAX_DAILY_LOSS:
        port.killed = True
    return (f"resumed: capital=Rs{port.capital:,.2f} (compounded) "
            f"dayPnL={port.intra_pnl:+,.2f} trades={port.n_trades} killed={port.killed}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    load_dotenv()

    # FIX #1: sys.path BEFORE any phase imports
    sys.path.insert(0, str(ROOT))

    print("=" * 78)
    print("LIVE TRADING BOT — RELIND (fixed v3, capital compounds across days)")
    print(f"ROOT={ROOT}")
    print(f"Cooldown={COOLDOWN_S}s | MinHold={MIN_HOLD_S}s | Warmup={WARMUP_S}s")
    print(f"Initial capital=Rs{INITIAL_CAPITAL:,.0f} | Daily loss limit=Rs{MAX_DAILY_LOSS:,.0f}")
    print("=" * 78)

    artifacts = load_model(args)
    expected_n = len(artifacts["feats_final"])
    got_n = artifacts["b_clf"].num_feature()
    if expected_n != got_n:
        log.warning("Model/feature mismatch; retraining.")
        try: Path(args.model_path).unlink()
        except Exception: pass
        artifacts = load_model(args)
    tau_table = load_tau_table()
    log.info(f"Features: {len(artifacts['feats_final'])}")

    from phase2_features import build_day, WARMUP as PH2_WARMUP

    # Breeze connect
    api_key = os.getenv("ICICI_APP_KEY")
    api_secret = os.getenv("ICICI_SECRET_KEY")
    session_token = os.getenv("ICICI_SESSION_TOKEN")
    breeze = None
    if not all([api_key, api_secret, session_token]):
        log.warning("Missing Breeze credentials — OFFLINE mode.")
    else:
        try:
            from breeze_connect import BreezeConnect
            breeze = BreezeConnect(api_key=api_key)
            breeze.generate_session(api_secret=api_secret, session_token=session_token)
            log.info("Breeze session established.")
        except Exception as e:
            log.error(f"BreezeConnect init failed: {e}"); breeze = None

    state = MarketState()
    if breeze is not None:
        breeze.on_ticks = make_on_ticks(state)
        try:
            breeze.ws_connect(); time.sleep(3)
            breeze.subscribe_feeds(exchange_code=EXCHANGE_CODE, stock_code=STOCK_CODE,
                                   product_type=PRODUCT_TYPE, get_market_depth=True,
                                   get_exchange_quotes=True)
            log.info(f"Subscribed to {STOCK_CODE}.")
        except Exception as e:
            log.error(f"Subscription failed: {e}"); breeze = None

    buf = LiveFeatureBuffer(stale_thr=args.stale_threshold)
    engine = LiveSignalEngine(artifacts, tau_table)
    port = LivePortfolio()
    today_str = datetime.now().strftime("%Y%m%d")

    # Replay existing raw CSV
    existing = sorted(LIVE_CSV_DIR.glob(f"reliance_depth_{today_str}_*.csv"))
    replay_n = 0
    if existing:
        replay_path = existing[-1]
        try:
            rdf = pd.read_csv(replay_path)
            def _parse_ts(t_str):
                try:
                    return datetime.combine(datetime.now().date(),
                                            datetime.strptime(str(t_str)[:8], "%H:%M:%S").time())
                except Exception: return None
            rdf["_ts"] = rdf.iloc[:, 0].apply(_parse_ts) if rdf.shape[1] > 0 else None
            rdf = rdf.dropna(subset=["_ts"]).tail(RING_BUFFER_SEC).reset_index(drop=True)
            if len(rdf) > 0 and rdf.shape[1] >= len(HEADER_LIVE):
                try:
                    last_dtc = int(pd.to_numeric(rdf["depth_ticks"], errors="coerce").ffill().iloc[-1])
                    last_qtc = int(pd.to_numeric(rdf["quote_ticks"], errors="coerce").ffill().iloc[-1])
                except Exception: last_dtc, last_qtc = 0, 0
                with state.lock:
                    state.depth_tick_count = max(state.depth_tick_count, last_dtc)
                    state.quote_tick_count = max(state.quote_tick_count, last_qtc)
                depth_kw = [(f"BuyNoOfOrders-{i}", f"BestBuyQty-{i}", f"BestBuyRate-{i}") for i in range(1, 6)]
                ask_kw = [(f"BestSellRate-{i}", f"BestSellQty-{i}", f"SellNoOfOrders-{i}") for i in range(1, 6)]
                for _, r in rdf.iterrows():
                    depth = []; ci = 1
                    for bo, bq_k, bp_k in depth_kw:
                        depth.append({bo: r.iloc[ci], bq_k: r.iloc[ci+1], bp_k: r.iloc[ci+2]}); ci += 3
                    for ap_k, aq_k, ao in ask_kw:
                        depth.append({ap_k: r.iloc[ci], aq_k: r.iloc[ci+1], ao: r.iloc[ci+2]}); ci += 3
                    try:
                        ltp_v = _safe_float(r.get("ltp", np.nan), np.nan)
                        depth_str = str(r.get("depth_time", ""))
                        ddt = None
                        if depth_str:
                            try: ddt = datetime.combine(r["_ts"].date(),
                                                         datetime.strptime(depth_str[:8], "%H:%M:%S").time())
                            except Exception: ddt = None
                        with state.lock:
                            state.last_depth_dt = ddt or r["_ts"]
                            state.last_quote_dt = ddt or r["_ts"]
                            state.last_any_dt = r["_ts"]
                            state.latest_depth = depth
                            state.latest_ltp = ltp_v if np.isfinite(ltp_v) else None
                        buf.add_snapshot(state.snapshot(), r["_ts"])
                        replay_n += 1
                    except Exception: continue
                log.info(f"Replayed {replay_n} rows from {replay_path.name}.")
        except Exception as e:
            log.warning(f"Replay failed ({e}); empty buffer.")

    logger = LiveCsvLogger(today_str)
    log.info(resume_from_logs(port, logger))

    stop_event = threading.Event()
    def shutdown(signum=None, frame=None):
        if stop_event.is_set(): return
        log.info("Shutting down ...")
        stop_event.set()
        if breeze:
            try:
                breeze.unsubscribe_feeds(exchange_code=EXCHANGE_CODE, stock_code=STOCK_CODE,
                                         product_type=PRODUCT_TYPE, get_market_depth=True,
                                         get_exchange_quotes=True)
                breeze.ws_disconnect()
            except Exception: pass
    signal.signal(signal.SIGINT, shutdown); signal.signal(signal.SIGTERM, shutdown)

    next_tick = int(time.time()) + 1
    start = time.time()
    rth_announced = False; in_rth = False
    last_day = datetime.now().date()
    sig_counts = {0: 0, 1: 0, -1: 0, 2: 0, -2: 0}

    while not stop_event.is_set():
        now = time.time()
        if stop_event.wait(timeout=max(0.0, next_tick - now)): break
        ts = datetime.now()

        snap = state.snapshot()

        # Watchdog / reconnect
        feed_stall_s = 99.0
        if snap["last_any_dt"] is not None:
            feed_stall_s = (ts - snap["last_any_dt"]).total_seconds()
        if in_rth and feed_stall_s > 10:
            log.warning(f"Feed stall {feed_stall_s:.0f}s — attempting reconnect.")
            if breeze is not None:
                try:
                    breeze.ws_connect()
                    breeze.subscribe_feeds(exchange_code=EXCHANGE_CODE, stock_code=STOCK_CODE,
                                           product_type=PRODUCT_TYPE, get_market_depth=True,
                                           get_exchange_quotes=True)
                    log.info("Reconnect attempted.")
                except Exception as e:
                    log.error(f"Reconnect failed: {e}")
            sig_out["sig"] = 0
            sig_out["reason"] = "watchdog:stale_feed"
            sig_out["risk_ok"] = False
        if in_rth and snap["last_any_dt"] is not None:
            tick_age = (ts - snap["last_any_dt"]).total_seconds()
            if tick_age > 10 and ts.second % 30 == 0:
                log.warning(f"No tick for {tick_age:.0f}s — check websocket.")

        # Day rollover — capital compounds, no reset
        if ts.date() != last_day:
            try:
                if port.pos["active"] and port.pos["entered"]:
                    m = port.last_valid_mid
                    bp_l = _safe_float(_depth_get(state.latest_depth, "BestBuyRate-1", 1), np.nan)
                    ap_l = _safe_float(_depth_get(state.latest_depth, "BestSellRate-1", 1), np.nan)
                    if np.isfinite(bp_l) and np.isfinite(ap_l): m = 0.5 * (bp_l + ap_l)
                    if np.isfinite(m):
                        pnl = (m - port.pos["entry"]) * port.pos["qty"] if port.pos["side"] > 0 \
                              else (port.pos["entry"] - m) * port.pos["qty"]
                        port._close_trade(ts, float(m), "eod", float(pnl))
            except Exception: pass
            for t in port.trades_today: logger.write_trade(t)
            logger.close()
            last_day = ts.date(); today_str = last_day.strftime("%Y%m%d")
            logger = LiveCsvLogger(today_str)
            # FIX #16: new_day() keeps capital, just resets intra-day state
            port.new_day()
            log.info(f"New day {today_str}. Compounded capital = Rs{port.capital:,.2f}")
            sig_counts = {0: 0, 1: 0, -1: 0, 2: 0, -2: 0}
            buf = LiveFeatureBuffer(stale_thr=args.stale_threshold)
            rth_announced = False

        # Write raw row
        if snap["latest_depth"] is not None:
            depth_row = _build_depth_row(snap["latest_depth"])
            ddt = snap["last_depth_dt"]; qdt = snap["last_quote_dt"]
            depth_age = (ts - ddt).total_seconds() if ddt else 99.0
            quote_age = (ts - qdt).total_seconds() if qdt else 99.0
            ltp_v = snap["latest_ltp"] if snap["latest_ltp"] is not None else ""
            raw_row = [ts.strftime("%H:%M:%S")] + list(depth_row) + \
                      [ltp_v, ts.strftime("%H:%M:%S.%f")[:-3],
                       ddt.strftime("%H:%M:%S.%f")[:-3] if ddt else "",
                       qdt.strftime("%H:%M:%S.%f")[:-3] if qdt else "",
                       round(depth_age, 3), round(quote_age, 3),
                       int(depth_age > args.stale_threshold),
                       snap["depth_tick_count"], snap["quote_tick_count"],
                       snap["other_tick_count"], snap["latest_exchange_time"] or ""]
            if len(raw_row) < len(HEADER_LIVE): raw_row += [""] * (len(HEADER_LIVE) - len(raw_row))
            else: raw_row = raw_row[:len(HEADER_LIVE)]
            logger.write_raw(raw_row)
            buf.add_snapshot(snap, ts)

        # Outside RTH
        if not is_rth(ts):
            if in_rth: log.info("Market closed."); in_rth = False
            if ts.second == 0 and ts.minute % 15 == 0:
                log.info(f"[pre/post-RTH] {ts.strftime('%H:%M:%S')} waiting ...")
            next_tick += 1; continue
        in_rth = True
        if not rth_announced:
            log.info("RTH started."); rth_announced = True

        sec_open = seconds_since_open(ts)
        feats_frame = buf.to_phase1_frame()
        sig_out = dict(sig=0, risk_ok=False, reason="buffer_warmup", side=0, size=0,
                       entry=0.0, stop=0.0, tgt=0.0, H=0,
                       p_l=0.5, p_s=0.5, raw=0.0, edge=0.0, fb=0.0, fs=0.0, regime=2)
        bp_live = ap_live = bq_live = aq_live = sprd_live = np.nan
        MIN_BUFFER_SEC = 600

        if len(feats_frame) >= 60 and snap["latest_depth"] is not None:
            try:
                bp_live = _safe_float(_depth_get(snap["latest_depth"], "BestBuyRate-1", 1), np.nan)
                ap_live = _safe_float(_depth_get(snap["latest_depth"], "BestSellRate-1", 1), np.nan)
                bq_live = _safe_float(_depth_get(snap["latest_depth"], "BestBuyQty-1", 1), np.nan)
                aq_live = _safe_float(_depth_get(snap["latest_depth"], "BestSellQty-1", 1), np.nan)
                sprd_live = ap_live - bp_live if np.isfinite(bp_live) and np.isfinite(ap_live) else np.nan
                feats_full = build_day(feats_frame)
                # FIX #20: build_day() does NOT include raw book columns (b1_p, a1_p etc.)
                # in its output — it only produces computed features (mid, sprd, ofi...).
                # Without this copy, decide() reads b1_p=0.0 → entry=0 → fake P&L.
                for _c in ["b1_p", "a1_p", "b1_q", "a1_q"]:
                    if _c in feats_frame.columns:
                        feats_full[_c] = feats_frame[_c].values
                needed = artifacts["feats_final"] + ["mkt_state", "is_stale", "warmup", "sprd",
                                                     "b1_p", "a1_p", "b1_q", "a1_q"]
                for c in needed:
                    if c not in feats_full.columns: feats_full[c] = 0.0
                feats_full["warmup"] = (np.arange(len(feats_full)) < PH2_WARMUP).astype(int)
                if sec_open < WARMUP_S: feats_full["warmup"] = 1
                sig_out = engine.decide(feats_full.tail(1).copy())
            except Exception as e:
                log.debug(f"feature/pred error: {e}")
                sig_out = dict(sig=0, risk_ok=False, reason=f"err:{str(e)[:30]}", side=0, size=0,
                               entry=0.0, stop=0.0, tgt=0.0, H=0,
                               p_l=0.5, p_s=0.5, raw=0.0, edge=0.0, fb=0.0, fs=0.0, regime=2)

        if sec_open < WARMUP_S or len(feats_frame) < MIN_BUFFER_SEC:
            sig_out["risk_ok"] = False
            sig_out["reason"] = f"warmup({len(feats_frame)}/{MIN_BUFFER_SEC}s)"
        elif port.killed:
            sig_out["risk_ok"] = False; sig_out["reason"] = "killed"

        mid_live = 0.5 * (bp_live + ap_live) if np.isfinite(bp_live) and np.isfinite(ap_live) else np.nan

        port.tick(ts, sig_out, bp_live, ap_live, bq_live, aq_live,
                  sprd_live if np.isfinite(sprd_live) else 0.0, sig_out["regime"])

        for t in port.trades_today:
            if t["exit_t"] == ts: logger.write_trade(t)
        port.trades_today = [tt for tt in port.trades_today if tt["exit_t"] != ts]

        eq = port.equity(mid_live)
        logger.write_eq(ts, eq, port, mid_live if np.isfinite(mid_live) else port.last_valid_mid)
        sig_counts[int(sig_out["sig"])] += 1

        # Console
        pos_str = "FLAT"
        if port.pos["active"] and port.pos["entered"]:
            side_c = "LONG" if port.pos["side"] > 0 else "SHRT"
            m = mid_live if np.isfinite(mid_live) else port.last_valid_mid
            un = ((m - port.pos["entry"]) * port.pos["qty"] if port.pos["side"] > 0
                  else (port.pos["entry"] - m) * port.pos["qty"]) if np.isfinite(m) else 0.0
            pos_str = f"{side_c} {port.pos['qty']:>4} @ {port.pos['entry']:>8.2f} un={un:+8.0f}"
        elif port.pos["active"]:
            side_c = "w-LONG" if port.pos["side"] > 0 else "w-SHRT"
            pos_str = f"{side_c} {port.pos['qty']:>4} @ {port.pos['posted_price']:>8.2f} (await)"

        depth_age = 99.0
        try:
            ddt = snap["last_depth_dt"]
            depth_age = (ts - ddt).total_seconds() if ddt else 99.0
        except Exception: pass

        print(f"{ts.strftime('%H:%M:%S')} | age={depth_age:4.1f}s reg={sig_out['regime']} "
              f"pL={sig_out['p_l']:.3f} pS={sig_out['p_s']:.3f} raw={sig_out['raw']:+.3f} "
              f"edge={sig_out['edge']:+.3f} sprd={sprd_live if np.isfinite(sprd_live) else 0:.2f} "
              f"sig={sig_out['sig']:>+2} [{sig_out['reason'][:18]:18s}] "
              f"| {pos_str:46s} | Eq=Rs{eq:>10,.0f}  dayPnL={port.intra_pnl:+,.0f} "
              f"trades={port.n_trades:>4} kill={int(port.killed)}",
              end="\r", file=sys.stdout, flush=True)

        next_tick += 1
        if args.duration is not None and time.time() - start >= args.duration:
            log.info("Duration reached."); break

    # Final shutdown
    try:
        if port.pos["active"] and port.pos["entered"]:
            m = port.last_valid_mid
            bp_l = _safe_float(_depth_get(state.latest_depth, "BestBuyRate-1", 1), np.nan)
            ap_l = _safe_float(_depth_get(state.latest_depth, "BestSellRate-1", 1), np.nan)
            if np.isfinite(bp_l) and np.isfinite(ap_l): m = 0.5 * (bp_l + ap_l)
            if np.isfinite(m):
                pnl = (m - port.pos["entry"]) * port.pos["qty"] if port.pos["side"] > 0 \
                      else (port.pos["entry"] - m) * port.pos["qty"]
                t = port._close_trade(datetime.now(), float(m), "shutdown", float(pnl))
                if t: logger.write_trade(t)
    except Exception: pass
    for t in port.trades_today: logger.write_trade(t)
    logger.close()
    print("")
    print("=" * 78)
    log.info(f"Session finished. Capital = Rs{port.capital:,.2f} | "
             f"Day P&L = Rs{port.intra_pnl:+,.2f} | trades={port.n_trades} | killed={port.killed}")
    log.info(f"Signal counts: {sig_counts}")


if __name__ == "__main__":
    main()
