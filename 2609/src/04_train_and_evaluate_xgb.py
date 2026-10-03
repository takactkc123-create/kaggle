"""XGBoost の学習・評価(S6E9)。

`03_feature_engineering_xgb.py` の関数で**本番で使う列だけ**を作り、
StratifiedKFold(n_splits=5, shuffle=True, random_state=42) で学習する。--save で提出ファイル・OOF・重要度を保存する。

作る列(69 列)
    生の 13 列(カテゴリは整数コード)
    + digit(fe.DIGIT_COLS)
    + Count(全 13 列)
    + Target Encoding(fe.te_plan(): Smooth Keys 4本・生の 13 列。fold 内で Out-of-Fold)

例
    uv run src/04_train_and_evaluate_xgb.py --max-bin 1024 --set-param colsample_bytree=0.3 --set-param max_depth=5 \\
        --learning-rate 0.03 --n-estimators 8000 --early-stopping 200 --n-jobs 7 --save
"""

from __future__ import annotations

import argparse
import importlib
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

fe = importlib.import_module("03_feature_engineering_xgb")

TARGET = "Will_Buy_EV"


# fold によらない部分(生の列・digit・Count)と、Target Encoding のキーの整数コードを作る
def prepare(train: pd.DataFrame, test: pd.DataFrame):
    cols = fe.ALL_COLS
    X, X_test = train[cols].copy(), test[cols].copy()
    X, X_test = fe.add_digit_features(X, X_test, train, test, cols=fe.DIGIT_COLS)
    X, X_test = fe.add_count_encoding(X, X_test, train, test, cols)
    X, X_test = fe.as_ordinal(X, X_test, fe.CATEGORICAL_COLS)

    sk_tr, sk_te = fe.make_smooth_keys(train, test)
    src_tr = pd.concat([sk_tr, train[cols]], axis=1)
    src_te = pd.concat([sk_te, test[cols]], axis=1)
    plan = fe.te_plan(list(sk_tr.columns))
    codes = fe.prepare_te_codes(src_tr, src_te, [c for c, _ in plan])
    return X, X_test, codes, plan


# 1 つの fold の学習行・検証行・test の行列を作る(Target Encoding は学習行だけで作る)
def fold_matrices(prep, y, tr_idx, va_idx):
    X, X_test, (c_tr, c_te, ncats), plan = prep
    te_tr, te_va, te_te = fe.fit_apply_te_cv_nested_plan(c_tr, c_te, ncats, y, plan, tr_idx, va_idx)
    X_tr = pd.concat([X.iloc[tr_idx].reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1)
    X_va = pd.concat([X.iloc[va_idx].reset_index(drop=True), te_va.reset_index(drop=True)], axis=1)
    X_te = pd.concat([X_test.reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)
    return X_tr, X_va, X_te


# XGBoost を 5-fold で学習し、OOF 予測と test 予測を返す
def run_cv(args):
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    if args.sample < 1.0:
        train = train.sample(frac=args.sample, random_state=42).reset_index(drop=True)
    y = (train[TARGET] == "Yes").astype(int)

    prep = prepare(train, test)
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(prep[0], y))[: args.folds]

    params = dict(random_state=42, tree_method="hist", n_jobs=args.n_jobs)
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
        # 止める本数は検証 fold で決まるので、OOF AUC はわずかに楽観的になる
        params["early_stopping_rounds"] = args.early_stopping
        params["eval_metric"] = "auc"

    oof = np.full(len(train), np.nan)
    test_pred = np.zeros(len(test))
    importances, feat_names, best_iters = None, None, []
    t0 = time.time()

    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_tr, X_va, X_te = fold_matrices(prep, y, tr_idx, va_idx)

        if args.dump_features:
            catalog = importlib.import_module("03_feature_engineering_AllCatalog")
            catalog.dump(f"xgb{args.out_suffix}", "XGBoost", X_tr.columns,
                                 cat_features=[c for c in X_tr.columns if str(X_tr[c].dtype) == "category"],
                                 note="本番の構成(fe.te_plan())")
            return None

        model = XGBClassifier(**params)
        if args.early_stopping:
            model.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
            best_iters.append(int(model.best_iteration) + 1)
        else:
            model.fit(X_tr, y.iloc[tr_idx])

        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / len(splits)
        if importances is None:
            importances = np.zeros(X_tr.shape[1])
        importances += model.feature_importances_ / len(splits)
        feat_names = list(X_tr.columns)
        bi = f" best_iter={best_iters[-1]}" if best_iters else ""
        print(f"  fold {fold} AUC: {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.5f}{bi}", flush=True)

    mask = ~np.isnan(oof)
    auc = roc_auc_score(y[mask], oof[mask])
    tag = (f" lr={params.get('learning_rate', 'default')} n_est={params.get('n_estimators', 'default')}"
           f" max_bin={params.get('max_bin', 'default')}")
    if best_iters:
        tag += f" best_iter_mean={np.mean(best_iters):.0f} best_iters={best_iters}"
    print(f"n_feat={len(feat_names)}  OOF AUC: {auc:.5f}  ({time.time() - t0:.0f}s, folds={len(splits)}, "
          f"sample={args.sample}){tag}", flush=True)

    if args.save:
        save_artifacts(test, oof, test_pred, importances, feat_names, auc, args.out_suffix)
    return auc


# OOF・test 予測・提出ファイル・重要度の図を保存する
def save_artifacts(test, oof, test_pred, importances, feat_names, auc, suffix=""):
    for d in ("submit", "oof", "importance"):
        os.makedirs(d, exist_ok=True)
    pd.DataFrame({"id": test["id"], TARGET: test_pred}).to_csv(f"submit/submission_xgb{suffix}.csv", index=False)
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


# 引数を読み、学習・評価・保存を行う
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5, help="実際に学習する fold 数(分け方は常に 5)")
    ap.add_argument("--sample", type=float, default=1.0, help="スクリーニング用に train を割合で間引く")
    ap.add_argument("--n-estimators", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--early-stopping", type=int, default=0,
                    help="検証 fold の AUC が指定本数改善しなければ止める(0 = 使わない)")
    ap.add_argument("--max-bin", type=int, default=None, help="XGBoost の max_bin(本番は 1024)")
    ap.add_argument("--set-param", action="append", default=[], metavar="KEY=VALUE",
                    help="XGBClassifier に渡す追加のパラメータ(複数回指定できる)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--save", action="store_true", help="提出ファイル・OOF・重要度を保存する")
    ap.add_argument("--dump-features", action="store_true",
                    help="学習せず、fold 1 の列名を docs/features_xgb<suffix>.json に書いて終了する")
    ap.add_argument("--out-suffix", default="", help="成果物のファイル名の接尾辞。例 '_lr05' -> oof/oof_xgb_lr05.npy")
    run_cv(ap.parse_args())


if __name__ == "__main__":
    main()
