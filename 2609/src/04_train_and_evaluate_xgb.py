"""XGBoost training / evaluation script for S6E9.

Imports FE functions from 03_feature_engineering_xgb.py, runs StratifiedKFold CV and
(optionally) writes submission / OOF / importance artifacts.

Usage:
    uv run 04_train_and_evaluate_xgb.py --pattern base
    uv run 04_train_and_evaluate_xgb.py --pattern te_exact --folds 3 --sample 0.3
    uv run 04_train_and_evaluate_xgb.py --pattern final --save
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import importlib
fe = importlib.import_module("03_feature_engineering_xgb")

TARGET = fe.TARGET


# ---------------------------------------------------------------------------
# pattern definitions
# ---------------------------------------------------------------------------
# Each pattern is a dict describing how to build the feature matrix.
#   enc          : "cat" (native category) | "ord" | "ohe"
#   arith        : None | "meaningful" | "all"
#   row_agg      : bool
#   ce_cols      : list of raw columns to count-encode (target-free)
#   ce_inter     : list of (a, b) pairs to count-encode as interaction keys
#   te_cols      : list of raw columns to target-encode (fold-safe)
#   te_inter     : list of (a, b) pairs to target-encode as interaction keys
#   te_smoothing : smoothing strength
ALL_COLS = fe.NUMERIC_COLS + fe.CATEGORICAL_COLS
LOWCARD_ALL = fe.LOW_CARD_NUMERIC + fe.CATEGORICAL_COLS


def _p(**kw):
    base = dict(
        enc="cat",
        arith=None,
        row_agg=False,
        ce_cols=None,
        ce_inter=None,
        te_cols=None,
        te_inter=None,
        te_smoothing=20.0,
        te_smoothings=None,   # list -> Triple TE (one column per smoothing)
        te_min_samples=1,
        te_nested=False,
        drop_raw_cat=False,
        digits=False,         # digit features (x // 10**k) % 10
        smooth_keys=False,    # coarse income/commute keys fed to TE (and CE)
        smooth_keys_ce=False,
    )
    base.update(kw)
    return base


HIGH_CARD_NUMERIC = ["Annual_Income_USD", "Daily_Commute_km"]
MID_CARD = HIGH_CARD_NUMERIC + ["Age"]


PATTERNS = {
    # --- encoding-style comparison (XGBoost specific) ---
    "base": _p(),
    "ord": _p(enc="ord"),
    "ohe": _p(enc="ohe"),
    # --- arithmetic ---
    "arith": _p(arith="meaningful"),
    "arith_all": _p(arith="all"),
    "row_agg": _p(row_agg=True),
    # --- count / frequency encoding ---
    "ce_cat": _p(ce_cols=fe.CATEGORICAL_COLS),
    "ce_all": _p(ce_cols=ALL_COLS),
    "ce_inter": _p(ce_inter=fe.cat_pairs()),
    # --- target encoding ---
    "te_cat": _p(te_cols=fe.CATEGORICAL_COLS),
    "te_lowcard": _p(te_cols=LOWCARD_ALL),
    "te_exact": _p(te_cols=ALL_COLS),
    "te_exact_s2": _p(te_cols=ALL_COLS, te_smoothing=2.0),
    "te_exact_s100": _p(te_cols=ALL_COLS, te_smoothing=100.0),
    "te_inter": _p(te_inter=fe.cat_pairs()),
    "te_inter_all": _p(te_inter=fe.cat_pairs(ALL_COLS)),
    # --- combinations (filled in below / via CLI) ---
    "te_exact_inter": _p(te_cols=ALL_COLS, te_inter=fe.cat_pairs()),
    "te_exact_ce": _p(te_cols=ALL_COLS, ce_cols=ALL_COLS),
    "te_exact_ord": _p(enc="ord", te_cols=ALL_COLS),
    "te_exact_ohe": _p(enc="ohe", te_cols=ALL_COLS),
    # --- round 2: isolate the high-cardinality numeric TE effect ---
    "te_high": _p(te_cols=HIGH_CARD_NUMERIC),
    "te_mid": _p(te_cols=MID_CARD),
    "te_num": _p(te_cols=fe.NUMERIC_COLS),
    "te_exact_s5": _p(te_cols=ALL_COLS, te_smoothing=5.0),
    "te_exact_s50": _p(te_cols=ALL_COLS, te_smoothing=50.0),
    "te_exact_ms10": _p(te_cols=ALL_COLS, te_min_samples=10),
    # high-card numeric x categorical interaction TE
    "te_hx_cat": _p(
        te_cols=ALL_COLS,
        te_inter=[(n, c) for n in HIGH_CARD_NUMERIC for c in fe.CATEGORICAL_COLS],
    ),
    "te_hx_home": _p(
        te_cols=ALL_COLS,
        te_inter=[(n, c) for n in HIGH_CARD_NUMERIC
                  for c in ("Home_Charging_Possible", "Subsidy_Available")],
    ),
    "te_exact_arith": _p(te_cols=ALL_COLS, arith="meaningful"),
    "te_exact_ce_high": _p(te_cols=ALL_COLS, ce_cols=MID_CARD),
    # --- round 3: tune the winning high-cardinality TE ---
    "te_high_s5": _p(te_cols=HIGH_CARD_NUMERIC, te_smoothing=5.0),
    "te_high_s50": _p(te_cols=HIGH_CARD_NUMERIC, te_smoothing=50.0),
    "te_high_s100": _p(te_cols=HIGH_CARD_NUMERIC, te_smoothing=100.0),
    "te_high_ord": _p(enc="ord", te_cols=HIGH_CARD_NUMERIC),
    "te_high_ce": _p(te_cols=HIGH_CARD_NUMERIC, ce_cols=HIGH_CARD_NUMERIC),
    # --- round 4: nested (inner-OOF) target encoding ---
    "nte_exact": _p(te_cols=ALL_COLS, te_nested=True),
    "nte_exact_ce": _p(te_cols=ALL_COLS, te_nested=True, ce_cols=ALL_COLS),
    "nte_high": _p(te_cols=HIGH_CARD_NUMERIC, te_nested=True),
    # XGBoost-specific differentiation: ordinal encoding instead of native
    # category dtype (LightGBM uses native category -> extra ensemble diversity)
    "nte_exact_ce_ord": _p(
        enc="ord", te_cols=ALL_COLS, te_nested=True, ce_cols=ALL_COLS
    ),
    "nte_exact_ce_ohe": _p(
        enc="ohe", te_cols=ALL_COLS, te_nested=True, ce_cols=ALL_COLS
    ),
    # final configuration (set after round 4)
    "final": _p(enc="ord", te_cols=ALL_COLS, te_nested=True, ce_cols=ALL_COLS),
    # --- round 5: reference_URL.md follow-up -------------------------------
    # Triple TE = smooth auto / 10 / 100 as three separate columns
    "tte": _p(
        enc="ord", te_cols=ALL_COLS, te_nested=True, ce_cols=ALL_COLS,
        te_smoothings=("auto", 10.0, 100.0),
    ),
    "tte20": _p(
        enc="ord", te_cols=ALL_COLS, te_nested=True, ce_cols=ALL_COLS,
        te_smoothings=("auto", 20.0, 100.0),
    ),
    # reference = `final`, but routed through the fast numpy TE path
    # (mathematically identical to te_nested with te_smoothing=20)
    "ref20": _p(
        enc="ord", te_cols=ALL_COLS, te_nested=True, ce_cols=ALL_COLS,
        te_smoothings=(20.0,),
    ),
    # Smooth Keys only (single smoothing, m=20) -> isolates the key effect
    "sk": _p(
        enc="ord", te_cols=ALL_COLS, te_nested=True, ce_cols=ALL_COLS,
        te_smoothings=(20.0,), smooth_keys=True,
    ),
    "dig_only": _p(
        enc="ord", te_cols=ALL_COLS, te_nested=True, ce_cols=ALL_COLS,
        te_smoothings=(20.0,), digits=True,
    ),
    # Triple TE + Smooth Keys
    "tte_sk": _p(
        enc="ord", te_cols=ALL_COLS, te_nested=True, ce_cols=ALL_COLS,
        te_smoothings=("auto", 10.0, 100.0), smooth_keys=True,
    ),
    "tte_sk_ce": _p(
        enc="ord", te_cols=ALL_COLS, te_nested=True, ce_cols=ALL_COLS,
        te_smoothings=("auto", 10.0, 100.0), smooth_keys=True, smooth_keys_ce=True,
    ),
    # digit features (use together with --max-bin 1024)
    "dig": _p(enc="ord", te_cols=ALL_COLS, te_nested=True, ce_cols=ALL_COLS,
              te_smoothings=(20.0,), digits=True),
    # everything
    "tte_sk_dig": _p(
        enc="ord", te_cols=ALL_COLS, te_nested=True, ce_cols=ALL_COLS,
        te_smoothings=("auto", 10.0, 100.0), smooth_keys=True, digits=True,
    ),
    # --- 2026-09-21: 誤差として見送った「符号がプラス」の施策の再検証 ---
    # One-Hot は単体で +0.00005 だった。本番構成(Triple TE + digit)の上で測り直す
    "tte_sk_dig_ohe": _p(
        enc="ohe", te_cols=ALL_COLS, te_nested=True, ce_cols=ALL_COLS,
        te_smoothings=("auto", 10.0, 100.0), smooth_keys=True, digits=True,
    ),
    # 四則演算は単体で +0.00004 だった。同上
    "tte_sk_dig_arith": _p(
        enc="ord", te_cols=ALL_COLS, te_nested=True, ce_cols=ALL_COLS,
        te_smoothings=("auto", 10.0, 100.0), smooth_keys=True, digits=True,
        arith="meaningful",
    ),
    # no count encoding (re-ablation once max_bin is raised)
    "tte_sk_dig_noce": _p(
        enc="ord", te_cols=ALL_COLS, te_nested=True,
        te_smoothings=("auto", 10.0, 100.0), smooth_keys=True, digits=True,
    ),
}


# ---------------------------------------------------------------------------
# feature building
# ---------------------------------------------------------------------------


def build_static(train_raw, test_raw, cfg):
    """Build the fold-independent part of the feature matrix."""
    cat_cols = list(fe.CATEGORICAL_COLS)
    tr = train_raw[fe.NUMERIC_COLS + cat_cols].copy()
    te = test_raw[fe.NUMERIC_COLS + cat_cols].copy()

    if cfg["arith"] is not None:
        pairs = fe.MEANINGFUL_PAIRS if cfg["arith"] == "meaningful" else fe.all_numeric_pairs()
        tr, te = fe.add_arithmetic(tr, te, train_raw, test_raw, pairs)

    if cfg["row_agg"]:
        tr, te = fe.add_row_aggregates(tr, te, train_raw, test_raw)

    if cfg["digits"]:
        tr, te = fe.add_digit_features(tr, te, train_raw, test_raw)

    if cfg["ce_cols"]:
        tr, te = fe.add_count_encoding(tr, te, train_raw, test_raw, cfg["ce_cols"])

    if cfg["smooth_keys"] and cfg["smooth_keys_ce"]:
        ktr, kte = fe.make_smooth_keys(train_raw, test_raw)
        tr, te = fe.add_count_encoding(tr, te, ktr, kte, list(ktr.columns))

    if cfg["ce_inter"]:
        ktr, kte = fe.make_interaction_keys(train_raw, test_raw, cfg["ce_inter"])
        tr, te = fe.add_count_encoding(tr, te, ktr, kte, list(ktr.columns))

    # encoding style for the raw categorical columns
    if cfg["enc"] == "cat":
        tr, te = fe.as_native_category(tr, te, cat_cols)
    elif cfg["enc"] == "ord":
        tr, te = fe.as_ordinal(tr, te, cat_cols)
    elif cfg["enc"] == "ohe":
        tr, te = fe.as_onehot(tr, te, cat_cols)
    else:
        raise ValueError(cfg["enc"])

    if cfg["drop_raw_cat"] and cfg["enc"] != "ohe":
        tr = tr.drop(columns=cat_cols)
        te = te.drop(columns=cat_cols)

    return tr, te


def build_te_sources(train_raw, test_raw, cfg):
    """Raw frames (train, test) + column list used for fold-wise target encoding."""
    if not cfg["te_cols"] and not cfg["te_inter"] and not cfg["smooth_keys"]:
        return None, None, []
    parts_tr, parts_te, cols = [], [], []
    if cfg["smooth_keys"]:
        ktr, kte = fe.make_smooth_keys(train_raw, test_raw)
        parts_tr.append(ktr)
        parts_te.append(kte)
        cols += list(ktr.columns)
    if cfg["te_cols"]:
        parts_tr.append(train_raw[cfg["te_cols"]])
        parts_te.append(test_raw[cfg["te_cols"]])
        cols += list(cfg["te_cols"])
    if cfg["te_inter"]:
        ktr, kte = fe.make_interaction_keys(train_raw, test_raw, cfg["te_inter"])
        parts_tr.append(ktr)
        parts_te.append(kte)
        cols += list(ktr.columns)
    return (
        pd.concat(parts_tr, axis=1),
        pd.concat(parts_te, axis=1),
        cols,
    )


# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------


def run_cv(cfg, args):
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    if args.sample < 1.0:
        train = train.sample(frac=args.sample, random_state=42).reset_index(drop=True)

    y = (train[TARGET] == "Yes").astype(int)

    X, X_test = build_static(train, test, cfg)
    te_src_tr, te_src_te, te_cols = build_te_sources(train, test, cfg)
    te_codes = None
    if te_cols and cfg["te_smoothings"]:
        te_codes = fe.prepare_te_codes(te_src_tr, te_src_te, te_cols)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))
    if args.folds < 5:
        splits = splits[: args.folds]

    params = dict(random_state=42, tree_method="hist", n_jobs=args.n_jobs)
    if cfg["enc"] == "cat":
        params["enable_categorical"] = True
    if args.max_bin is not None:
        params["max_bin"] = args.max_bin
    for kv in args.set_param:
        k, v = kv.split("=", 1)
        try:
            v = int(v) if v.isdigit() else float(v)
        except ValueError:
            pass
        params[k] = v
    if args.n_estimators is not None:
        params["n_estimators"] = args.n_estimators
    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate
    if args.early_stopping:
        # NOTE: the stopping point is chosen on the validation fold, so the
        # resulting OOF AUC is mildly optimistic. Use `--n-estimators <fixed>`
        # without --early-stopping for a fully clean estimate.
        params["early_stopping_rounds"] = args.early_stopping
        params["eval_metric"] = "auc"

    n_te_out = len(te_cols) * (len(cfg["te_smoothings"]) if cfg["te_smoothings"] else 1)
    oof = np.full(len(train), np.nan)
    test_pred = np.zeros(len(test))
    importances = np.zeros(X.shape[1] + n_te_out)
    feat_names = None
    best_iters = []
    t0 = time.time()

    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        X_te = X_test

        if te_cols:
            if cfg["te_smoothings"]:
                if not cfg["te_nested"]:
                    raise ValueError("te_smoothings requires te_nested=True")
                c_tr, c_te, ncats = te_codes
                te_tr, te_va, te_te = fe.fit_apply_te_cv_nested_multi(
                    c_tr, c_te, ncats, y, te_cols, tr_idx, va_idx,
                    cfg["te_smoothings"], cfg["te_min_samples"],
                )
            else:
                te_fn = fe.fit_apply_te_cv_nested if cfg["te_nested"] else fe.fit_apply_te_cv
                te_tr, te_va, te_te = te_fn(
                    te_src_tr, te_src_te, y, te_cols, tr_idx, va_idx,
                    cfg["te_smoothing"], cfg["te_min_samples"],
                )
            X_tr = pd.concat([X_tr.reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1)
            X_va = pd.concat([X_va.reset_index(drop=True), te_va.reset_index(drop=True)], axis=1)
            X_te = pd.concat([X_test.reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)

        if args.dump_features:
            import feature_catalog
            feature_catalog.dump(
                f"xgb{args.out_suffix}", "XGBoost", X_tr.columns,
                cat_features=[c for c in X_tr.columns
                              if str(X_tr[c].dtype) == "category"],
                note=f"pattern={args.pattern}",
            )
            return

        model = XGBClassifier(**params)
        if args.early_stopping:
            model.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
            best_iters.append(int(model.best_iteration) + 1)
        else:
            model.fit(X_tr, y.iloc[tr_idx])

        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / len(splits)
        importances += model.feature_importances_ / len(splits)
        feat_names = list(X_tr.columns)
        bi = f" best_iter={best_iters[-1]}" if best_iters else ""
        print(f"  fold {fold} AUC: {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.5f}{bi}", flush=True)

    mask = ~np.isnan(oof)
    auc = roc_auc_score(y[mask], oof[mask])
    tag = (f" lr={params.get('learning_rate', 'default')}"
           f" n_est={params.get('n_estimators', 'default')}"
           f" max_bin={params.get('max_bin', 'default')}")
    if best_iters:
        tag += f" best_iter_mean={np.mean(best_iters):.0f} best_iters={best_iters}"
    print(f"PATTERN={args.pattern}  n_feat={len(feat_names)}  OOF AUC: {auc:.5f}  "
          f"({time.time() - t0:.0f}s, folds={len(splits)}, sample={args.sample}){tag}", flush=True)

    if args.save:
        save_artifacts(
            train, test, y, oof, test_pred, importances, feat_names, auc, args.out_suffix
        )
    return auc


def save_artifacts(train, test, y, oof, test_pred, importances, feat_names, auc, suffix=""):
    for d in ("submit", "oof", "importance"):
        os.makedirs(d, exist_ok=True)

    pd.DataFrame({"id": test["id"], TARGET: test_pred}).to_csv(
        f"submit/submission_xgb{suffix}.csv", index=False
    )
    np.save(f"oof/oof_xgb{suffix}.npy", oof)
    np.save(f"oof/pred_xgb{suffix}.npy", test_pred)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
    top = imp.head(30)[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(top))))
    ax.barh(top.index, top.values, color="#2b6cb0")
    ax.set_title(f"XGBoost feature importance (OOF AUC {auc:.5f})")
    ax.set_xlabel("gain-based importance (fold mean)")
    fig.tight_layout()
    fig.savefig(f"importance/importance_xgb{suffix}.png", dpi=130)
    plt.close(fig)

    print(f"saved: submit/submission_xgb{suffix}.csv, oof/oof_xgb{suffix}.npy, "
          f"oof/pred_xgb{suffix}.npy, importance/importance_xgb{suffix}.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="base")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sample", type=float, default=1.0)
    ap.add_argument("--n-estimators", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument(
        "--early-stopping",
        type=int,
        default=0,
        help="early_stopping_rounds on the validation fold (0 = off)",
    )
    ap.add_argument("--max-bin", type=int, default=None,
                    help="XGBoost max_bin (tree_method=hist). 1024 per reference_URL.md")
    ap.add_argument("--set-param", action="append", default=[], metavar="KEY=VALUE",
                    help="extra XGBClassifier param, repeatable")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--save", action="store_true")
    ap.add_argument(
        "--dump-features", action="store_true",
        help="学習せず、fold1 の特徴量の列名を docs/features_<tag>.json に書いて終了する",
    )
    ap.add_argument(
        "--out-suffix",
        default="",
        help="suffix for artifact filenames, e.g. '_lr05' -> oof/oof_xgb_lr05.npy",
    )
    args = ap.parse_args()

    if args.pattern not in PATTERNS:
        raise SystemExit(f"unknown pattern: {args.pattern}\navailable: {sorted(PATTERNS)}")
    run_cv(PATTERNS[args.pattern], args)


if __name__ == "__main__":
    main()
