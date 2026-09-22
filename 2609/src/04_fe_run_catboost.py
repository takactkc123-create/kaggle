"""CatBoost training / evaluation script for S6E9.

Imports the FE functions from `03_fe_catboost.py`, builds a feature set from the
components given with `--fe`, runs StratifiedKFold CV and (optionally) writes
the ensemble artifacts.

Examples
--------
  uv run 04_fe_run_catboost.py --fe base --folds 3 --iters 400 --lr 0.15
  uv run 04_fe_run_catboost.py --fe te_all,catify --save
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import importlib
fe = importlib.import_module("03_fe_catboost")

ALL_COMPONENTS = {
    "base",      # nothing extra
    "arith",     # arithmetic features
    "inter",     # interaction string keys used as cat_features
    "cnt",       # count encoding of the 13 base columns
    "cnt_ix",    # count encoding of the interaction keys
    "te_cat",    # target encoding of the 6 categorical columns
    "te_low",    # target encoding of low-cardinality numeric columns
    "te_all",    # exact-value target encoding of all 13 columns
    "te_ix",     # target encoding of interaction keys
    "catify",    # low-card numeric columns passed as cat_features (string)
    "drop_num",  # drop the raw low-card numerics when they are target encoded
    "digits",    # per-digit decomposition of every numeric column (10^-4..10^3)
    "skeys",     # multi-scale "smooth keys" added as TE keys
    "te3",       # triple target encoding (3 smoothing levels per key)
}

TRIPLE_SMOOTHS = [10.0, 20.0, 100.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fe", default="base", help="comma separated FE components")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--iters", type=int, default=0, help="0 = CatBoost default (1000)")
    p.add_argument("--lr", type=float, default=0.0, help="0 = CatBoost default")
    p.add_argument("--rows", type=int, default=0, help="subsample train rows (0 = all)")
    p.add_argument("--smooth", type=float, default=20.0, help="TE smoothing")
    p.add_argument("--one-hot", type=int, default=0, help="one_hot_max_size (0 = default)")
    p.add_argument("--depth", type=int, default=0)
    p.add_argument(
        "--fast",
        action="store_true",
        help="speed-oriented params (Plain boosting, max_ctr_complexity=1, border_count=64)",
    )
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--border", type=int, default=0, help="override border_count")
    p.add_argument(
        "--hc-border",
        type=int,
        default=0,
        help="per-feature border_count for the high-cardinality numerics "
        "(Annual_Income_USD / Daily_Commute_km and their TE columns). "
        "Much cheaper than raising --border globally.",
    )
    p.add_argument("--rsm", type=float, default=0.0, help="colsample per split (0=off)")
    p.add_argument(
        "--subsample",
        type=float,
        default=0.0,
        help="Bernoulli row subsample per tree (0 = CatBoost default bootstrap)",
    )
    p.add_argument("--save", action="store_true", help="write submission/oof/importance")
    p.add_argument(
        "--dump-features", action="store_true",
        help="学習せず、fold1 の特徴量の列名を docs/features_<tag>.json に書いて終了する",
    )
    p.add_argument("--tag", default="", help="label printed with the result")
    return p.parse_args()


def build_features(components: set[str], rows: int, seed: int = 42):
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    if rows and rows < len(train):
        train = train.sample(n=rows, random_state=seed).reset_index(drop=True)

    y = (train[fe.TARGET] == "Yes").astype(int)

    base_cols = fe.NUMERIC_COLS + fe.CATEGORICAL_COLS
    X = train[base_cols].copy()
    X_test = test[base_cols].copy()

    cat_features: list[str] = list(fe.CATEGORICAL_COLS)
    feature_cols: list[str] = list(base_cols)
    helper_cols: list[str] = []  # created but not fed to the model

    # --- digit features (must run before catify casts columns to str) -----
    if "digits" in components:
        digit_cols = fe.add_digits(X)
        fe.add_digits(X_test)
        digit_cols = fe.drop_constant([X, X_test], digit_cols)
        feature_cols += digit_cols

    # --- multi-scale smooth keys (TE keys only, not model features) -------
    sk_cols: list[str] = []
    if "skeys" in components:
        sk_cols = fe.add_smooth_keys(X)
        fe.add_smooth_keys(X_test)
        helper_cols += sk_cols

    # --- arithmetic -------------------------------------------------------
    if "arith" in components:
        new_tr = fe.add_arithmetic(X)
        fe.add_arithmetic(X_test)
        feature_cols += new_tr

    # --- interaction keys -------------------------------------------------
    need_ix = {"inter", "cnt_ix", "te_ix"} & components
    ix_cols: list[str] = []
    if need_ix:
        ix_cols = fe.add_interactions(X)
        fe.add_interactions(X_test)
        if "inter" in components:
            feature_cols += ix_cols
            cat_features += ix_cols
        else:
            helper_cols += ix_cols

    # --- count encoding ---------------------------------------------------
    if "cnt" in components:
        feature_cols += fe.add_count_encoding(X, X_test, base_cols)
    if "cnt_ix" in components:
        feature_cols += fe.add_count_encoding(X, X_test, ix_cols)

    # --- catify (low-card numerics as categorical) ------------------------
    if "catify" in components:
        fe.cast_to_str([X, X_test], fe.LOWCARD_NUM_COLS)
        cat_features += fe.LOWCARD_NUM_COLS

    # --- target encoding source columns (applied inside each fold) --------
    te_cols: list[str] = []
    if "te_all" in components:
        te_cols += base_cols
    else:
        if "te_cat" in components:
            te_cols += fe.CATEGORICAL_COLS
        if "te_low" in components:
            te_cols += fe.LOWCARD_NUM_COLS
    if "te_ix" in components:
        te_cols += ix_cols
    if te_cols and sk_cols:
        te_cols += sk_cols
    te_cols = list(dict.fromkeys(te_cols))

    drop_after_te: list[str] = []
    if "drop_num" in components:
        drop_after_te = [c for c in fe.LOWCARD_NUM_COLS if c in te_cols]

    keep_cols = list(dict.fromkeys(feature_cols + helper_cols + te_cols))
    X = X[keep_cols]
    X_test = X_test[keep_cols]

    return train, test, X, y, X_test, feature_cols, cat_features, te_cols, drop_after_te


def main() -> None:
    args = parse_args()
    components = {c.strip() for c in args.fe.split(",") if c.strip()}
    unknown = components - ALL_COMPONENTS
    if unknown:
        raise SystemExit(f"unknown FE components: {sorted(unknown)}")

    t0 = time.time()
    (
        train,
        test,
        X,
        y,
        X_test,
        feature_cols,
        cat_features,
        te_cols,
        drop_after_te,
    ) = build_features(components, args.rows)

    print(f"[{args.tag or args.fe}] rows={len(X)} te_cols={len(te_cols)}")

    params = dict(random_state=42, verbose=False, allow_writing_files=False)
    if args.iters:
        params["iterations"] = args.iters
    if args.lr:
        params["learning_rate"] = args.lr
    if args.one_hot:
        params["one_hot_max_size"] = args.one_hot
    if args.depth:
        params["depth"] = args.depth
    if args.fast:
        params["boosting_type"] = "Plain"
        params["max_ctr_complexity"] = 1
        params["border_count"] = 64
    if args.threads:
        params["thread_count"] = args.threads
    if args.border:
        params["border_count"] = args.border
    if args.rsm:
        params["rsm"] = args.rsm
    if args.subsample:
        params["bootstrap_type"] = "Bernoulli"
        params["subsample"] = args.subsample

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    importances = None
    used_features: list[str] = []
    used_cats: list[str] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        X_te = X_test.copy()
        y_tr = y.iloc[tr_idx].to_numpy()

        fold_features = list(feature_cols)
        fold_cats = list(cat_features)

        if te_cols:
            new_te = fe.target_encode(
                X_tr,
                X_va,
                X_te,
                y_tr,
                te_cols,
                smooth=args.smooth,
                smooths=TRIPLE_SMOOTHS if "te3" in components else None,
            )
            fold_features += new_te
            for col in drop_after_te:
                if col in fold_features:
                    fold_features.remove(col)
                if col in fold_cats:
                    fold_cats.remove(col)

        fold_cats = [c for c in fold_cats if c in fold_features]
        used_features, used_cats = fold_features, fold_cats

        fold_params = dict(params)
        if args.hc_border:
            targets = set(fe.HIGHCARD_NUM_COLS) | {
                f"te{tag}_{c}"
                for c in fe.HIGHCARD_NUM_COLS
                for tag in ("", "10", "20", "100")
            }
            fold_params["per_float_feature_quantization"] = [
                f"{i}:border_count={args.hc_border}"
                for i, name in enumerate(fold_features)
                if name in targets and name not in fold_cats
            ]

        if args.dump_features:
            import feature_dump
            feature_dump.dump(
                args.tag or "catboost", "CatBoost", fold_features,
                cat_features=fold_cats, note=f"fe={args.fe}",
            )
            return

        model = CatBoostClassifier(**fold_params)
        model.fit(X_tr[fold_features], y_tr, cat_features=fold_cats)

        oof_pred[va_idx] = model.predict_proba(X_va[fold_features])[:, 1]
        if args.save:
            test_pred += model.predict_proba(X_te[fold_features])[:, 1] / args.folds

        imp = model.get_feature_importance()
        importances = imp if importances is None else importances + imp
        print(
            f"  fold {fold} AUC: {roc_auc_score(y.iloc[va_idx], oof_pred[va_idx]):.5f}"
            f"  ({time.time() - t0:.0f}s)"
        )

    oof_auc = roc_auc_score(y, oof_pred)
    print(f"RESULT\t{args.tag or args.fe}\tOOF AUC: {oof_auc:.5f}\t"
          f"n_features={len(used_features)}\t{time.time() - t0:.0f}s")

    if args.save:
        os.makedirs("submit", exist_ok=True)
        os.makedirs("oof", exist_ok=True)
        os.makedirs("importance", exist_ok=True)

        pd.DataFrame({"id": test["id"], fe.TARGET: test_pred}).to_csv(
            "submit/submission_catboost.csv", index=False
        )
        np.save("oof/oof_catboost.npy", oof_pred)
        np.save("oof/pred_catboost.npy", test_pred)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        imp = importances / args.folds
        order = np.argsort(imp)
        names = [used_features[i] for i in order]
        fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(names))))
        ax.barh(range(len(names)), imp[order], color="#2a9d8f")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("CatBoost feature importance")
        ax.set_title(f"CatBoost importance (OOF AUC {oof_auc:.5f})")
        fig.tight_layout()
        fig.savefig("importance/importance_catboost.png", dpi=130)
        print("saved submit/oof/importance artifacts")


if __name__ == "__main__":
    main()
