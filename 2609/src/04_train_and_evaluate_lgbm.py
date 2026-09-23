"""LightGBM CV runner for S6E9.

Imports the FE functions from `03_feature_engineering_lgbm.py`, applies the requested FE patterns
(all target-based encoders are fitted strictly inside the outer fold), runs
StratifiedKFold(n_splits=5, shuffle=True, random_state=42) and - with --save -
writes submission / OOF / importance artefacts.

Examples
--------
    uv run 04_train_and_evaluate_lgbm.py --patterns base --folds 3            # screening
    uv run 04_train_and_evaluate_lgbm.py --patterns base,te1                  # full 5-fold
    uv run 04_train_and_evaluate_lgbm.py --patterns base,te1,te2 --save       # artefacts
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import importlib
fe = importlib.import_module("03_feature_engineering_lgbm")

N_SPLITS = 5  # fold definition is fixed across all three models - do not change
SEED = 42


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--patterns", default="base", help="comma separated FE patterns")
    p.add_argument("--folds", type=int, default=N_SPLITS, help="folds actually trained")
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--num_leaves", type=int, default=31)
    p.add_argument(
        "--feature_fraction", type=float, default=1.0,
        help="木ごとに使う列の割合 (colsample_bytree)。1.0 = 全列",
    )
    p.add_argument("--max_depth", type=int, default=0, help="木の深さ上限 (0 = 無制限)")
    p.add_argument("--sample", type=float, default=1.0, help="row subsample for screening")
    p.add_argument("--smooth", type=float, default=20.0, help="TE smoothing")
    p.add_argument(
        "--smooths",
        default="",
        help="comma separated TE smoothing levels, e.g. 'auto,10,100' "
        "(triple target encoding). Overrides --smooth when given.",
    )
    p.add_argument(
        "--max_bin",
        type=int,
        default=255,
        help="LightGBM max_bin. 255 = default. Raising it to 1024 exposes the "
        "value-level structure of Annual_Income_USD, but also raises the noise "
        "floor from 0.00015 to ~0.00033 (see reference_URL.md).",
    )
    p.add_argument("--inner", type=int, default=5, help="inner folds for TE")
    p.add_argument("--save", action="store_true", help="write submission/oof/importance")
    p.add_argument(
        "--dump-features", action="store_true",
        help="学習せず、fold1 の特徴量の列名を docs/features_<tag>.json に書いて終了する",
    )
    p.add_argument("--tag", default="lgbm")
    p.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="LightGBM threads. Lower it when other agents are competing for CPU.",
    )
    p.add_argument(
        "--early_stopping",
        type=int,
        default=0,
        help="early stopping rounds on the validation fold (0 = off). "
        "NOTE: the stop point is chosen on the validation fold, so the resulting "
        "OOF AUC carries a small optimistic bias; use --n_estimators with the "
        "reported median best_iteration for an unbiased number.",
    )
    return p


def main():
    args = build_parser().parse_args()
    pats = [s.strip() for s in args.patterns.split(",") if s.strip()]
    t0 = time.time()

    train, test = fe.load_data()
    y_full = (train[fe.TARGET] == "Yes").astype(int)

    if args.sample < 1.0:
        idx = (
            train.sample(frac=args.sample, random_state=SEED).index.sort_values()
        )
        train = train.loc[idx].reset_index(drop=True)
        y_full = y_full.loc[idx].reset_index(drop=True)

    train_c, test_c = fe.make_categorical(train, test)
    keys_tr, keys_te = fe.make_key_frame(train, test)

    if "sk" in pats:  # multi-scale smooth keys (income/10, /100, /1000, floor(km))
        keys_tr = fe.add_smooth_keys(keys_tr, train)
        keys_te = fe.add_smooth_keys(keys_te, test)

    # ---------------- static (non target-based) features ----------------
    X_parts_tr = [train_c[fe.NUMERIC_COLS + fe.CATEGORICAL_COLS]]
    X_parts_te = [test_c[fe.NUMERIC_COLS + fe.CATEGORICAL_COLS]]

    if "ordcat" in pats:
        # native category dtype をやめ、整数コード(ordinal)として渡す。
        # LightGBM のカテゴリ分割は値を任意のグループに分けられるため、TE を併用している
        # 状況では過剰適合しうる。外部の台帳では native category が -0.0027 と報告されている。
        X_parts_tr = [train_c[fe.NUMERIC_COLS].join(
            pd.DataFrame({c: train_c[c].cat.codes.astype("int16") for c in fe.CATEGORICAL_COLS},
                         index=train_c.index))]
        X_parts_te = [test_c[fe.NUMERIC_COLS].join(
            pd.DataFrame({c: test_c[c].cat.codes.astype("int16") for c in fe.CATEGORICAL_COLS},
                         index=test_c.index))]

    if "nocat" in pats:
        X_parts_tr = [train_c[fe.NUMERIC_COLS]]
        X_parts_te = [test_c[fe.NUMERIC_COLS]]
    if "nonum" in pats:
        X_parts_tr = [train_c[fe.CATEGORICAL_COLS]]
        X_parts_te = [test_c[fe.CATEGORICAL_COLS]]

    if "arith" in pats:
        X_parts_tr.append(fe.add_arithmetic_meaningful(train))
        X_parts_te.append(fe.add_arithmetic_meaningful(test))
    if "arith_all" in pats:
        X_parts_tr.append(fe.add_arithmetic_all_pairs(train))
        X_parts_te.append(fe.add_arithmetic_all_pairs(test))
    if "gmean" in pats:
        X_parts_tr.append(fe.add_group_means(train))
        X_parts_te.append(fe.add_group_means(test))
    if "digit" in pats:
        d_tr = fe.add_digit_features(train)
        X_parts_tr.append(d_tr)
        # force the identical column set on test (a digit may be constant in
        # one frame and not the other)
        X_parts_te.append(fe.add_digit_features(test, keep=list(d_tr.columns)))

    cnt_keys = []
    if "cnt1" in pats:
        cnt_keys += fe.single_keys("all")
    if "sk" in pats and "skcnt" in pats:
        cnt_keys += fe.SMOOTH_KEYS
    if "cnt2" in pats:
        cnt_keys += fe.pair_keys("cat")
    if "cnt2all" in pats:
        cnt_keys += fe.pair_keys("all")
    if "cntrow" in pats:
        cnt_keys += fe.all_columns_key()
    for fp in fe.FP_SETS:
        if fp in pats or f"{fp}cnt" in pats:
            cnt_keys += fe.fingerprint_key(fp)
    if cnt_keys:
        a, b = fe.count_encode(keys_tr, keys_te, cnt_keys)
        X_parts_tr.append(a)
        X_parts_te.append(b)

    X_static = pd.concat(X_parts_tr, axis=1)
    X_test_static = pd.concat(X_parts_te, axis=1)

    # ---------------- target-encoding key sets (fold-internal) ----------
    te_keys = []
    if "te1" in pats:
        te_keys += fe.single_keys("all")
    if "sk" in pats:
        te_keys += fe.SMOOTH_KEYS
    if "te1cat" in pats:
        te_keys += fe.single_keys("cat")
    if "te1num" in pats:
        te_keys += fe.single_keys("num")
    if "te2" in pats:
        te_keys += fe.pair_keys("cat")
    if "te2all" in pats:
        te_keys += fe.pair_keys("all")
    if "te2catnum" in pats:
        te_keys += fe.pair_keys("cat_lownum")
    if "te3" in pats:
        te_keys += fe.triple_keys("selected")
    if "te3cat" in pats:
        te_keys += fe.triple_keys("cat")
    if "terow" in pats:
        te_keys += fe.all_columns_key()
    for fp in fe.FP_SETS:
        if fp in pats or f"{fp}te" in pats:
            te_keys += fe.fingerprint_key(fp)
    # de-duplicate, keep order
    seen = set()
    te_keys = [k for k in te_keys if not (k in seen or seen.add(k))]

    if args.smooths.strip():
        smooths = [
            s.strip() if s.strip() == "auto" else float(s)
            for s in args.smooths.split(",")
            if s.strip()
        ]
    else:
        smooths = [args.smooth]

    y = y_full.to_numpy()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(X_static), np.nan)
    test_pred = np.zeros(len(X_test_static))
    n_used = 0
    importance = None
    feat_names = None
    best_iters: list[int] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_static, y)):
        if fold >= args.folds:
            break
        X_tr = X_static.iloc[tr_idx]
        X_va = X_static.iloc[va_idx]
        X_te = X_test_static
        y_tr, y_va = y[tr_idx], y[va_idx]

        if te_keys:
            te_tr, (te_va, te_te) = fe.target_encode_fold(
                keys_tr.iloc[tr_idx],
                y_tr,
                [keys_tr.iloc[va_idx], keys_te],
                te_keys,
                smooth=smooths,
                n_inner=args.inner,
                seed=SEED,
            )
            X_tr = pd.concat([X_tr.reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1)
            X_va = pd.concat([X_va.reset_index(drop=True), te_va.reset_index(drop=True)], axis=1)
            X_te = pd.concat([X_te.reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)

        if args.dump_features:
            import feature_catalog
            feature_catalog.dump(
                args.tag, "LightGBM", X_tr.columns,
                cat_features=[c for c in X_tr.columns
                              if str(X_tr[c].dtype) == "category"],
                note=f"patterns={args.patterns} smooths={args.smooths}",
            )
            return

        model = LGBMClassifier(
            random_state=SEED,
            verbosity=-1,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_bin=args.max_bin,
            n_jobs=args.n_jobs,
            **(
                {"colsample_bytree": args.feature_fraction}
                if args.feature_fraction < 1.0 else {}
            ),
            **({"max_depth": args.max_depth} if args.max_depth > 0 else {}),
        )
        if args.early_stopping > 0:
            from lightgbm import early_stopping, log_evaluation

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=[
                    early_stopping(args.early_stopping, verbose=False),
                    log_evaluation(0),
                ],
            )
            best_iters.append(int(model.best_iteration_ or args.n_estimators))
        else:
            model.fit(X_tr, y_tr)

        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1]
        n_used += 1

        if importance is None:
            importance = np.zeros(X_tr.shape[1])
            feat_names = list(X_tr.columns)
        importance += model.feature_importances_

        print(
            f"fold {fold} AUC: {roc_auc_score(y_va, oof[va_idx]):.5f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    mask = ~np.isnan(oof)
    auc = roc_auc_score(y[mask], oof[mask])
    test_pred /= max(n_used, 1)
    n_feat = len(feat_names) if feat_names else 0
    bi = ""
    if best_iters:
        bi = (
            f" best_iters={best_iters} median_best={int(np.median(best_iters))}"
        )
    print(
        f"PATTERNS={args.patterns} folds={n_used} n_feat={n_feat} "
        f"smooth={smooths} max_bin={args.max_bin} lr={args.learning_rate} "
        f"n_est={args.n_estimators} es={args.early_stopping} "
        f"OOF_AUC={auc:.5f} time={time.time() - t0:.0f}s{bi}",
        flush=True,
    )

    if args.save:
        tag = args.tag
        for d in ("submit", "oof", "importance"):
            os.makedirs(d, exist_ok=True)
        np.save(f"oof/oof_{tag}.npy", oof)
        np.save(f"oof/pred_{tag}.npy", test_pred)
        pd.DataFrame({"id": test["id"], fe.TARGET: test_pred}).to_csv(
            f"submit/submission_{tag}.csv", index=False
        )

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        imp = pd.Series(importance / n_used, index=feat_names).sort_values(
            ascending=True
        )
        top = imp.tail(40)
        fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(top))))
        ax.barh(top.index, top.to_numpy(), color="#2f7ed8")
        ax.set_xlabel("mean split importance")
        ax.set_title(f"LightGBM feature importance (OOF AUC {auc:.5f})")
        fig.tight_layout()
        fig.savefig(f"importance/importance_{tag}.png", dpi=130)
        plt.close(fig)

        imp.sort_values(ascending=False).to_csv(f"importance/importance_{tag}.csv")
        print(
            f"saved submit/submission_{tag}.csv, oof/oof_{tag}.npy, "
            f"oof/pred_{tag}.npy, importance/importance_{tag}.png"
        )


if __name__ == "__main__":
    main()
