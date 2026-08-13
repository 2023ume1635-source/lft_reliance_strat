"""
================================================================================
PHASE 4 — REGIME-ADAPTIVE MULTI-TASK LIGHTGBM MODEL + MICROPRICE BASELINE +
          CALIBRATION + ISOTONIC BLEND + PURGED/EMBARGO TIME-SERIES CV
================================================================================

Inputs  : output/labels/<yyyymmdd>.parquet
          output/splits.json
          output/feature_list.txt

Outputs : artifacts/models/*.pkl
          artifacts/preds/*.parquet
          reports/phase4_cv.csv
          reports/phase4_test_metrics.csv
          reports/phase4_regime_metrics.csv
          reports/phase4_feature_importance.csv
          reports/phase4_model_summary.json

FIXES over original:
  - Sample weight = recency * (1 + |y_dir_R|) for classifier
  - Better NaN handling in metrics
"""

from __future__ import annotations
import os, sys, json, warnings, time, pickle, math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.special import softmax

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (f1_score, roc_auc_score, balanced_accuracy_score,
                             brier_score_loss, mean_absolute_error, r2_score, log_loss)

import lightgbm as lgb

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
FEAT_DIR = ROOT / "output" / "features"
LABEL_DIR = ROOT / "output" / "labels"
CLEAN_DIR = ROOT / "output" / "clean"
SPLIT_F = ROOT / "output" / "splits.json"
FLIST_F = ROOT / "output" / "feature_list.txt"

ART_DIR = ROOT / "artifacts"
MODEL_DIR = ART_DIR / "models"
PRED_DIR = ART_DIR / "preds"
REPORT_DIR = ROOT / "reports"
for d in (ART_DIR, MODEL_DIR, PRED_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

TICK = 0.05
CV_FOLDS = 5
EMBARGO_S = 60
PURGE_S = 30
RECENCY_LAMBDA = 0.98
SEED = 42
np.random.seed(SEED)

LGB_CLF_PARAMS = dict(
    objective="multiclass", num_class=3, metric="multi_logloss",
    learning_rate=0.04, num_leaves=63, max_depth=-1, min_data_in_leaf=80,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
    lambda_l1=0.1, lambda_l2=0.5, max_bin=127, seed=SEED,
    deterministic=True, force_row_wise=True, verbosity=-1,
)
LGB_REG_PARAMS = dict(
    objective="regression", metric="mae",
    learning_rate=0.04, num_leaves=63, min_data_in_leaf=100,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
    lambda_l1=0.1, lambda_l2=0.5, max_bin=127, seed=SEED,
    deterministic=True, force_row_wise=True, verbosity=-1,
)
LGB_BIN_PARAMS = dict(
    objective="binary", metric="binary_logloss",
    learning_rate=0.04, num_leaves=31, min_data_in_leaf=120,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
    lambda_l1=0.2, lambda_l2=1.0, max_bin=127, is_unbalance=True,
    seed=SEED, verbosity=-1,
)
N_BOOST_CLF = 600
N_BOOST_REG = 500
N_BOOST_BIN = 400
LR_STOP = 50

# ---------------------------------------------------------------------------
def load_split() -> Dict[str, List[str]]:
    with open(SPLIT_F, "r") as f:
        return json.load(f)

def load_feature_list() -> List[str]:
    return pd.read_csv(FLIST_F, header=None)[0].astype(str).tolist()

DROP_LEAK = {"mid", "b_vwap", "a_vwap", "wmid", "kf_mid", "kf_sprd"}

def _date_key(p: Path) -> str:
    return p.stem

def load_all_labels(feats_ok: List[str]) -> pd.DataFrame:
    files = sorted(LABEL_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No labelled parquets in {LABEL_DIR}. Run phase3 first.")
    frames = []
    for p in files:
        d = _date_key(p)
        df = pd.read_parquet(p)
        df["trade_date"] = pd.to_datetime(d, format="%Y%m%d").date()
        df["_file"] = d
        frames.append(df)
    big = pd.concat(frames, axis=0).sort_index()
    for c in ("y_valid", "y_take_dir", "y_dir", "y_edge_30", "y_buyfill", "y_sellfill",
              "mkt_state", "imb1", "sprd", "mid", "wmid", "kf_innov", "R_per_share"):
        if c not in big.columns:
            raise ValueError(f"Missing expected column {c} in labels parquet.")
    return big

# ---------------------------------------------------------------------------
def microprice_signal(df: pd.DataFrame) -> np.ndarray:
    imb1 = df["imb1"].values.astype(np.float64).clip(0.02, 0.98)
    sp = df["sprd"].values.astype(np.float64)
    s_imb = (df["imb1"].groupby(df["_file"] if "_file" in df.columns else df["trade_date"])
               .transform(lambda s: s.rolling(60, min_periods=20).std())
               .fillna(0.2).clip(lower=0.05)).values
    k = np.sqrt(np.clip(sp, TICK, 1.0) / (2 * TICK))
    z = (imb1 - 0.5) / s_imb
    z = np.clip(z, -6, 6)
    logits = np.stack([-k * z, np.zeros_like(z), k * z], axis=1)
    return softmax(logits, axis=1).astype(np.float32)

# ---------------------------------------------------------------------------
def make_cv_folds(train_dates: List[str]) -> List[Tuple[List[str], List[str]]]:
    assert len(train_dates) >= 2
    folds = []
    for val_idx in range(1, len(train_dates)):
        folds.append((train_dates[:val_idx], [train_dates[val_idx]]))
    return folds

def recency_weights(dates: pd.Series, train_dates: List[str]) -> np.ndarray:
    latest = pd.to_datetime(train_dates).max()
    day_diffs = (pd.to_datetime(dates).values - np.datetime64(latest)) / np.timedelta64(1, 'D')
    day_diffs = -np.array(day_diffs, dtype=float)
    return (RECENCY_LAMBDA ** day_diffs).astype(np.float32)

# ---------------------------------------------------------------------------
def clf_metrics(y_true, p_pred):
    y = y_true.astype(int)
    p_pred = np.asarray(p_pred, dtype=np.float64)
    m = np.isfinite(y) & np.all(np.isfinite(p_pred), axis=1)
    if m.sum() < 10:
        return dict(f1_macro=float("nan"), bacc_dir=float("nan"),
                    auc_macro=float("nan"), logloss=float("nan"), dir_acc=float("nan"))
    y = y[m]; p_pred = p_pred[m]
    y_enc = y + 1
    pred_class = p_pred.argmax(axis=1) - 1
    f1_macro = f1_score(y, pred_class, average="macro", zero_division=0)
    mask = (y != 0)
    if mask.sum() > 10 and len(np.unique(y[mask])) > 1:
        bacc = balanced_accuracy_score(y[mask], pred_class[mask])
        try: auc = roc_auc_score(y_enc, p_pred, multi_class="ovo", average="macro", labels=[0, 1, 2])
        except Exception: auc = float("nan")
        dir_pred = np.where(p_pred[mask, 2] > p_pred[mask, 0], 1, -1)
        dir_acc = float((dir_pred == y[mask]).mean())
    else:
        bacc = float("nan"); auc = float("nan"); dir_acc = float("nan")
    try: ll = float(log_loss(y_enc, p_pred, labels=[0, 1, 2]))
    except Exception: ll = float("nan")
    return dict(f1_macro=float(f1_macro), bacc_dir=float(bacc),
                auc_macro=float(auc), logloss=ll, dir_acc=dir_acc)

def reg_metrics(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 10: return dict(mae=float("nan"), r2=float("nan"))
    return dict(mae=float(mean_absolute_error(y_true[m], y_pred[m])),
                r2=float(r2_score(y_true[m], y_pred[m])))

def bin_metrics(y_true, p_pred):
    m = np.isfinite(y_true) & np.isfinite(p_pred)
    if m.sum() < 10: return dict(brier=float("nan"), logloss=float("nan"))
    return dict(brier=float(brier_score_loss(y_true[m], p_pred[m])),
                logloss=float(log_loss(y_true[m], p_pred[m], labels=[0, 1])))

# ---------------------------------------------------------------------------
def train_classifier(Xtr, ytr, wtr, Xv, yv, wv, params, nboost, stop):
    dtr = lgb.Dataset(Xtr, label=ytr + 1, weight=wtr, free_raw_data=False)
    dva = lgb.Dataset(Xv, label=yv + 1, weight=wv, reference=dtr, free_raw_data=False)
    return lgb.train(params, dtr, num_boost_round=nboost, valid_sets=[dtr, dva],
                     valid_names=["tr", "va"],
                     callbacks=[lgb.early_stopping(stop, verbose=False), lgb.log_evaluation(0)])

def train_regressor(Xtr, ytr, wtr, Xv, yv, wv, params, nboost, stop):
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, free_raw_data=False)
    dva = lgb.Dataset(Xv, label=yv, weight=wv, reference=dtr, free_raw_data=False)
    return lgb.train(params, dtr, num_boost_round=nboost, valid_sets=[dtr, dva],
                     valid_names=["tr", "va"],
                     callbacks=[lgb.early_stopping(stop, verbose=False), lgb.log_evaluation(0)])

def train_binary(Xtr, ytr, wtr, Xv, yv, wv, params, nboost, stop):
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, free_raw_data=False)
    dva = lgb.Dataset(Xv, label=yv, weight=wv, reference=dtr, free_raw_data=False)
    return lgb.train(params, dtr, num_boost_round=nboost, valid_sets=[dtr, dva],
                     valid_names=["tr", "va"],
                     callbacks=[lgb.early_stopping(stop, verbose=False), lgb.log_evaluation(0)])

# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 78)
    print("PHASE 4 — REGIME-ADAPTIVE MULTI-TASK LIGHTGBM MODEL")
    print("=" * 78)

    splits = load_split()
    to_ymd = lambda x: x.replace("-", "")
    train_d = [to_ymd(d) for d in splits["train"]]
    test_d  = [to_ymd(d) for d in splits["test"]]
    valid_d = [to_ymd(d) for d in splits["valid"]]
    print(f"TRAIN ({len(train_d)}): {train_d}")
    print(f"TEST  ({len(test_d)}): {test_d}")
    print(f"VALID ({len(valid_d)}): {valid_d}")

    all_feats = load_feature_list()
    feats = [c for c in all_feats if c not in DROP_LEAK]
    cat_feats = [c for c in ("vol_regime", "sprd_regime", "vol_liq_regime",
                             "trend_regime", "stale_regime", "mkt_state", "sess_bucket") if c in feats]
    print(f"Features: {len(all_feats)} -> after leak-drop: {len(feats)}")

    print("\nLoading labels ...")
    big = load_all_labels(feats)
    print(f"Total rows: {len(big):,}")

    for c in feats:
        if c not in big.columns: big[c] = 0.0
        if c not in cat_feats: big[c] = pd.to_numeric(big[c], errors="coerce")
    big[feats] = big[feats].replace([np.inf, -np.inf], np.nan)
    big[feats] = big.groupby("_file", group_keys=False)[feats].ffill().fillna(0.0)
    for c in cat_feats: big[c] = big[c].fillna(0).astype(np.int32)
    edge_tail_na = big["y_edge_30"].isna()
    if edge_tail_na.any():
        big.loc[edge_tail_na, "y_valid"] = False
        big.loc[edge_tail_na, "y_edge_30"] = 0.0
    for c in ("y_buyfill", "y_sellfill"): big[c] = big[c].fillna(0).astype(np.int8)
    big["y_take_dir"] = big["y_take_dir"].fillna(0).astype(np.int8)
    big["y_dir"] = big["y_dir"].fillna(0).astype(np.int8)

    big["_split"] = ""
    big.loc[big["_file"].isin(train_d), "_split"] = "train"
    big.loc[big["_file"].isin(test_d), "_split"] = "test"
    big.loc[big["_file"].isin(valid_d), "_split"] = "valid"

    # FIX: sample weight includes edge magnitude
    if "y_dir_R" in big.columns:
        big["_clf_weight"] = (1.0 + big["y_dir_R"].abs().fillna(0)).astype(np.float32)
    else:
        big["_clf_weight"] = 1.0

    # Microprice baseline
    print("Computing microprice baseline (M3) ...")
    mp = microprice_signal(big)
    big["p_short_m3"] = mp[:, 0]; big["p_flat_m3"] = mp[:, 1]; big["p_long_m3"] = mp[:, 2]

    # CV
    print(f"\n--- Walk-forward CV ({len(train_d) - 1} folds) ---")
    cv_records = []
    oof_p_m1 = np.zeros(((big["_split"] == "train").sum(), 3), dtype=np.float32)
    oof_edge = np.zeros((big["_split"] == "train").sum(), dtype=np.float32)
    oof_bfill = np.zeros((big["_split"] == "train").sum(), dtype=np.float32)
    oof_sfill = np.zeros((big["_split"] == "train").sum(), dtype=np.float32)
    oof_y = big.loc[big["_split"] == "train", "y_take_dir"].values
    oof_yedge = big.loc[big["_split"] == "train", "y_edge_30"].values

    date_to_pos = {}
    n = 0
    for file in sorted(big["_file"].unique()):
        n_d = (big["_file"] == file).sum()
        date_to_pos[file] = (n, n + n_d)
        n += n_d

    folds = make_cv_folds(train_d)
    clf_boosters = []; reg_boosters = []; bfill_boosters = []; sfill_boosters = []
    fi_accum = np.zeros(len(feats), dtype=np.float64)

    for fi, (tr_dates, va_dates) in enumerate(folds, 1):
        print(f"\n[Fold {fi}/{len(folds)}] TRAIN={tr_dates} VAL={va_dates}")
        tr_mask = big["_file"].isin(tr_dates)
        va_mask = big["_file"].isin(va_dates)
        last_tr_day = tr_dates[-1]
        i0, i1 = date_to_pos[last_tr_day]
        tr_purge = np.zeros(len(big), dtype=bool)
        tr_purge[max(0, i1 - PURGE_S - EMBARGO_S):i1] = True
        tr_mask = tr_mask & (~tr_purge)
        tr_use = tr_mask & big["y_valid"].astype(bool)
        va_use = va_mask & big["y_valid"].astype(bool)

        w_tr = recency_weights(big.loc[tr_use, "trade_date"], splits["train"]) * big.loc[tr_use, "_clf_weight"].values
        w_va = big.loc[va_use, "_clf_weight"].values.astype(np.float32)

        Xtr = big.loc[tr_use, feats]; Xv = big.loc[va_use, feats]
        ytr_clf = big.loc[tr_use, "y_take_dir"].values; yv_clf = big.loc[va_use, "y_take_dir"].values
        ytr_edge = big.loc[tr_use, "y_edge_30"].values; yv_edge = big.loc[va_use, "y_edge_30"].values
        ytr_buy = big.loc[tr_use, "y_buyfill"].values; yv_buy = big.loc[va_use, "y_buyfill"].values
        ytr_sell = big.loc[tr_use, "y_sellfill"].values; yv_sell = big.loc[va_use, "y_sellfill"].values
        print(f"   tr={len(Xtr):,} va={len(Xv):,}")

        b_clf = train_classifier(Xtr, ytr_clf, w_tr, Xv, yv_clf, w_va, LGB_CLF_PARAMS, N_BOOST_CLF, LR_STOP)
        b_reg = train_regressor(Xtr, ytr_edge, w_tr, Xv, yv_edge, w_va, LGB_REG_PARAMS, N_BOOST_REG, LR_STOP)
        b_buy = train_binary(Xtr, ytr_buy, w_tr, Xv, yv_buy, w_va, LGB_BIN_PARAMS, N_BOOST_BIN, LR_STOP)
        b_sell = train_binary(Xtr, ytr_sell, w_tr, Xv, yv_sell, w_va, LGB_BIN_PARAMS, N_BOOST_BIN, LR_STOP)
        clf_boosters.append(b_clf); reg_boosters.append(b_reg)
        bfill_boosters.append(b_buy); sfill_boosters.append(b_sell)
        fi_accum += np.array(b_clf.feature_importance(importance_type="gain"))

        va_all = va_mask
        p_clf_va = b_clf.predict(big.loc[va_all, feats])
        e_clf_va = b_reg.predict(big.loc[va_all, feats])
        bf_va = b_buy.predict(big.loc[va_all, feats])
        sf_va = b_sell.predict(big.loc[va_all, feats])
        idx_va_global = np.where(va_all.values & (big["_split"] == "train").values)[0]
        train_pos = np.where((big["_split"] == "train").values)[0]
        local_idx = np.searchsorted(train_pos, idx_va_global)
        oof_p_m1[local_idx] = p_clf_va; oof_edge[local_idx] = e_clf_va
        oof_bfill[local_idx] = bf_va; oof_sfill[local_idx] = sf_va

        met = clf_metrics(yv_clf, p_clf_va[big.loc[va_all, "y_valid"].values])
        met_e = reg_metrics(yv_edge, e_clf_va[big.loc[va_all, "y_valid"].values])
        cv_records.append(dict(fold=fi, val_dates=",".join(va_dates), **met,
                               edge_mae=met_e["mae"], edge_r2=met_e["r2"]))
        print(f"   dir-acc={met['dir_acc']:.3f} F1={met['f1_macro']:.3f} AUC={met['auc_macro']:.3f} edge_MAE={met_e['mae']:.3f}")

    cv_df = pd.DataFrame(cv_records)
    cv_df.to_csv(REPORT_DIR / "phase4_cv.csv", index=False)
    print(f"\nMean CV: dir-acc={cv_df['dir_acc'].mean():.3f} F1={cv_df['f1_macro'].mean():.3f}")

    # Fill oldest day with fold-1
    oldest = train_d[0]
    oldest_mask = (big["_file"] == oldest).values
    train_pos = np.where((big["_split"] == "train").values)[0]
    oldest_local = np.searchsorted(train_pos, np.where(oldest_mask)[0])
    X_oldest = big.loc[oldest_mask, feats]
    oof_p_m1[oldest_local] = clf_boosters[0].predict(X_oldest)
    oof_edge[oldest_local] = reg_boosters[0].predict(X_oldest)
    oof_bfill[oldest_local] = bfill_boosters[0].predict(X_oldest)
    oof_sfill[oldest_local] = sfill_boosters[0].predict(X_oldest)

    # Feature importance
    fi_accum /= max(fi_accum.sum(), 1e-9)
    fi_df = pd.DataFrame({"feature": feats, "gain": fi_accum}).sort_values("gain", ascending=False).reset_index(drop=True)
    fi_df.to_csv(REPORT_DIR / "phase4_feature_importance.csv", index=False)
    print("\nTop-20 features:")
    for _, r in fi_df.head(20).iterrows(): print(f"  {r['feature']:<22s} {r['gain']:.4f}")

    dead = fi_df[fi_df["gain"] < 1e-5]["feature"].tolist()
    feats_final = [c for c in feats if c not in dead]
    print(f"Pruned {len(dead)} dead features -> {len(feats_final)} final")

    # Isotonic calibration
    print("\nIsotonic calibration ...")
    def _fit_cal(y_bin, p):
        if y_bin.sum() < 20 or (1 - y_bin).sum() < 20: return None
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y_bin); return iso
    cal_long = _fit_cal((oof_y == 1).astype(int), oof_p_m1[:, 2])
    cal_short = _fit_cal((oof_y == -1).astype(int), oof_p_m1[:, 0])
    def _cal(p, iso): return iso.predict(p) if iso is not None else p

    oof_p = np.zeros_like(oof_p_m1)
    oof_p[:, 2] = _cal(oof_p_m1[:, 2], cal_long)
    oof_p[:, 0] = _cal(oof_p_m1[:, 0], cal_short)
    oof_p[:, 1] = (1 - oof_p[:, 0] - oof_p[:, 2]).clip(0, 1)
    s = oof_p.sum(axis=1, keepdims=True); s[s < 1e-9] = 1
    oof_p /= s

    # Blend weights
    print("Optimising blend weights ...")
    oof_regimes = big.loc[big["_split"] == "train", "mkt_state"].values
    oof_m3_l = big.loc[big["_split"] == "train", "p_long_m3"].values
    oof_m3_s = big.loc[big["_split"] == "train", "p_short_m3"].values
    def _grid_search(pl_m1, ps_m1, pl_m3, ps_m3, edge_pred, y):
        best_w, best_sc = 0.5, -1e9
        for w in np.linspace(0, 1, 21):
            pl = w * pl_m1 + (1 - w) * pl_m3; ps = w * ps_m1 + (1 - w) * ps_m3
            sig = (pl - ps) * (0.25 + np.abs(edge_pred))
            mask = np.abs(y) > 1e-9
            pnl = sig[mask] * y[mask]
            sc = pnl.mean() / (pnl.std() + 1e-9) if pnl.std() > 1e-9 else 0.0
            if sc > best_sc: best_sc, best_w = sc, w
        return float(best_w), float(best_sc)
    regime_w = {}
    for r in range(6):
        m = oof_regimes == r
        if m.sum() < 200: regime_w[r] = 0.6; continue
        w, _ = _grid_search(oof_p[m, 2], oof_p[m, 0], oof_m3_l[m], oof_m3_s[m], oof_edge[m], oof_yedge[m])
        regime_w[r] = w
    w_global, _ = _grid_search(oof_p[:, 2], oof_p[:, 0], oof_m3_l, oof_m3_s, oof_edge, oof_yedge)
    print(f"   global w_M1={w_global:.2f}")

    # Retrain final
    print("\nRetraining final models ...")
    tr_mask_f = big["_file"].isin(train_d[:-1]) & big["y_valid"].astype(bool)
    va_mask_f = big["_file"].isin([train_d[-1]]) & big["y_valid"].astype(bool)
    w_tr_f = recency_weights(big.loc[tr_mask_f, "trade_date"], splits["train"]) * big.loc[tr_mask_f, "_clf_weight"].values
    w_va_f = big.loc[va_mask_f, "_clf_weight"].values.astype(np.float32)
    Xtr_f = big.loc[tr_mask_f, feats_final]; Xv_f = big.loc[va_mask_f, feats_final]
    b_clf_final = train_classifier(Xtr_f, big.loc[tr_mask_f, "y_take_dir"].values, w_tr_f,
                                   Xv_f, big.loc[va_mask_f, "y_take_dir"].values, w_va_f,
                                   LGB_CLF_PARAMS, N_BOOST_CLF + 200, LR_STOP)
    b_reg_final = train_regressor(Xtr_f, big.loc[tr_mask_f, "y_edge_30"].values, w_tr_f,
                                  Xv_f, big.loc[va_mask_f, "y_edge_30"].values, w_va_f,
                                  LGB_REG_PARAMS, N_BOOST_REG + 200, LR_STOP)
    b_buy_final = train_binary(Xtr_f, big.loc[tr_mask_f, "y_buyfill"].values, w_tr_f,
                               Xv_f, big.loc[va_mask_f, "y_buyfill"].values, w_va_f,
                               LGB_BIN_PARAMS, N_BOOST_BIN + 100, LR_STOP)
    b_sell_final = train_binary(Xtr_f, big.loc[tr_mask_f, "y_sellfill"].values, w_tr_f,
                                Xv_f, big.loc[va_mask_f, "y_sellfill"].values, w_va_f,
                                LGB_BIN_PARAMS, N_BOOST_BIN + 100, LR_STOP)

    # Score all
    print("Scoring ...")
    all_p_raw = b_clf_final.predict(big[feats_final])
    all_e_pred = b_reg_final.predict(big[feats_final])
    all_bfill_p = b_buy_final.predict(big[feats_final])
    all_sfill_p = b_sell_final.predict(big[feats_final])

    all_p = np.zeros_like(all_p_raw)
    all_p[:, 2] = _cal(all_p_raw[:, 2], cal_long)
    all_p[:, 0] = _cal(all_p_raw[:, 0], cal_short)
    all_p[:, 1] = (1 - all_p[:, 0] - all_p[:, 2]).clip(0, 1)
    s = all_p.sum(axis=1, keepdims=True); s[s < 1e-9] = 1; all_p /= s

    # Replace TRAIN with OOF
    train_pos_g = np.where((big["_split"] == "train").values)[0]
    all_p[train_pos_g] = oof_p
    all_e_pred[train_pos_g] = oof_edge
    all_bfill_p[train_pos_g] = oof_bfill
    all_sfill_p[train_pos_g] = oof_sfill

    regimes_all = big["mkt_state"].values
    w_per_row = np.array([regime_w.get(int(r), w_global) for r in regimes_all])
    p_long_blend = w_per_row * all_p[:, 2] + (1 - w_per_row) * big["p_long_m3"].values
    p_short_blend = w_per_row * all_p[:, 0] + (1 - w_per_row) * big["p_short_m3"].values
    p_flat_blend = (1 - p_long_blend - p_short_blend).clip(0, 1)
    sp = p_long_blend + p_short_blend + p_flat_blend
    p_long_blend /= sp; p_short_blend /= sp; p_flat_blend /= sp

    big["p_long_m1"] = all_p[:, 2].astype(np.float32)
    big["p_short_m1"] = all_p[:, 0].astype(np.float32)
    big["p_flat_m1"] = all_p[:, 1].astype(np.float32)
    big["edge_pred"] = all_e_pred.astype(np.float32)
    big["p_buyfill"] = all_bfill_p.astype(np.float32)
    big["p_sellfill"] = all_sfill_p.astype(np.float32)
    big["w_blend"] = w_per_row.astype(np.float32)
    big["p_long"] = p_long_blend.astype(np.float32)
    big["p_short"] = p_short_blend.astype(np.float32)
    big["p_flat"] = p_flat_blend.astype(np.float32)
    big["raw_sig"] = (p_long_blend - p_short_blend).astype(np.float32)
    big["expected_edge"] = (big["raw_sig"] * (0.25 + big["edge_pred"].abs())).astype(np.float32)

    # Metrics
    print("\n--- Holdout metrics ---")
    for split in ("train", "test", "valid"):
        mv = (big["_split"] == split) & big["y_valid"].astype(bool)
        if mv.sum() < 50: continue
        y_clf = big.loc[mv, "y_take_dir"].values
        p_clf = np.stack([big.loc[mv, "p_short"].values, big.loc[mv, "p_flat"].values, big.loc[mv, "p_long"].values], axis=1)
        met = clf_metrics(y_clf, p_clf)
        print(f"[{split.upper()}] dir-acc={met['dir_acc']:.3f} F1={met['f1_macro']:.3f} AUC={met['auc_macro']:.3f}")

    # Save predictions
    for file, grp in big.groupby("_file"):
        keep_cols = ["trade_date", "mkt_state", "y_valid", "y_take_dir", "y_edge_30",
                     "y_buyfill", "y_sellfill", "R_per_share",
                     "p_long_m3", "p_short_m3", "p_flat_m3",
                     "p_long_m1", "p_short_m1", "p_flat_m1",
                     "p_buyfill", "p_sellfill", "edge_pred",
                     "w_blend", "p_long", "p_short", "p_flat", "raw_sig", "expected_edge", "_split"]
        keep = [c for c in keep_cols if c in grp.columns]
        grp[keep].to_parquet(PRED_DIR / f"{file}.parquet")

    # Save artifacts
    artifacts = dict(feats_final=feats_final, cat_feats=[c for c in cat_feats if c in feats_final],
                     b_clf=b_clf_final, b_reg=b_reg_final, b_buy=b_buy_final, b_sell=b_sell_final,
                     cal_long=cal_long, cal_short=cal_short,
                     w_global=w_global, regime_w=regime_w,
                     params_clf=LGB_CLF_PARAMS, params_reg=LGB_REG_PARAMS, params_bin=LGB_BIN_PARAMS,
                     dead_features=dead, cv_mean=cv_df.mean(numeric_only=True).to_dict())
    with open(MODEL_DIR / "phase4_models.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    summary = dict(n_features_final=len(feats_final), n_pruned_dead=len(dead),
                   w_global=w_global, regime_w=regime_w,
                   cv_dir_acc=float(cv_df['dir_acc'].mean()),
                   runtime_s=round(time.time() - t0, 1))
    with open(REPORT_DIR / "phase4_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nPhase 4 complete in {summary['runtime_s']:.0f}s.")


if __name__ == "__main__":
    main()
