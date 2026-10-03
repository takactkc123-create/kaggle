"""CatBoost の学習・評価(S6E9)。

`03_feature_engineering_catboost.py` の関数で**本番で使う列だけ**を作り、
StratifiedKFold(n_splits=5, shuffle=True, random_state=42) で学習する。--save で提出ファイル・OOF・重要度を保存する。

作る列(43 列)
    生の 13 列(値の種類（ユニーク値）が少ない数値列も文字列にして cat_features に渡す = catify)
    + digit(fe.DIGIT_COLS。catify の前に作る)
    + Target Encoding(fe.te_plan(): 数値7列・Smooth Keys 3本。fold 内で Out-of-Fold)

例
    uv run src/04_train_and_evaluate_catboost.py --iters 1000 --lr 0.06 --fast --border 64 --hc-border 1024 --threads 7 --save
"""

from __future__ import annotations

import argparse
import importlib
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

fe = importlib.import_module("03_feature_engineering_catboost")


# コマンドライン引数を読む
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--iters", type=int, default=0, help="0 = CatBoost の既定(1000)")
    p.add_argument("--lr", type=float, default=0.0, help="0 = CatBoost の既定")
    p.add_argument("--rows", type=int, default=0, help="スクリーニング用に train を行数で間引く(0 = すべて)")
    p.add_argument("--one-hot", type=int, default=0, help="one_hot_max_size (0 = 既定)")
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--fast", action="store_true",
                   help="速度優先の設定(Plain boosting・max_ctr_complexity=1・border_count=64)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--border", type=int, default=0, help="border_count を上書きする")
    p.add_argument("--hc-border", type=int, default=0,
                   help="年収・通勤距離とその Target Encoding の列だけ border_count をこの値にする")
    p.add_argument("--rsm", type=float, default=0.0, help="分割ごとに使う列の割合(0 = 使わない)")
    p.add_argument("--subsample", type=float, default=0.0, help="木ごとの行の間引き(0 = CatBoost の既定)")
    p.add_argument("--save", action="store_true", help="提出ファイル・OOF・重要度を保存する")
    p.add_argument("--dump-features", action="store_true",
                   help="学習せず、fold 1 の列名を docs/features_<tag>.json に書いて終了する")
    p.add_argument("--tag", default="", help="結果の表示に付けるラベル")
    p.add_argument("--out-suffix", default="",
                   help="成果物のファイル名の接尾辞。例 '_d5' -> oof/oof_catboost_d5.npy(既定は本番の名前)")
    return p.parse_args()


# fold によらない部分(生の列・digit・Smooth Keys・catify)を作る
def prepare(train: pd.DataFrame, test: pd.DataFrame):
    base_cols = fe.NUMERIC_COLS + fe.CATEGORICAL_COLS
    X, X_test = train[base_cols].copy(), test[base_cols].copy()

    digit_cols = fe.add_digits(X, cols=fe.DIGIT_COLS)          # catify より前に作る
    fe.add_digits(X_test, cols=fe.DIGIT_COLS)
    digit_cols = fe.drop_constant([X, X_test], digit_cols)

    sk_cols = fe.add_smooth_keys(X, fe.PROD_SMOOTH_KEY_SPECS)   # Target Encoding のキー専用(モデルには渡さない)
    fe.add_smooth_keys(X_test, fe.PROD_SMOOTH_KEY_SPECS)

    fe.cast_to_str([X, X_test], fe.LOWCARD_NUM_COLS)            # catify
    feature_cols = base_cols + digit_cols
    cat_features = fe.CATEGORICAL_COLS + fe.LOWCARD_NUM_COLS
    return X, X_test, feature_cols, cat_features, fe.te_plan(sk_cols)


# 1 つの fold の学習行・検証行・test の行列と、使う列・カテゴリ列を作る(Target Encoding は学習行だけで作る)
def fold_matrices(prep, y: np.ndarray, tr_idx, va_idx):
    X, X_test, feature_cols, cat_features, plan = prep
    X_tr, X_va, X_te = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy(), X_test.copy()
    new_te = fe.target_encode_plan(X_tr, X_va, X_te, y[tr_idx], plan)
    feats = feature_cols + new_te
    cats = [c for c in cat_features if c in feats]
    return X_tr[feats], X_va[feats], X_te[feats], feats, cats


# CatBoost を 5-fold で学習・評価し、成果物を保存する
def main() -> None:
    args = parse_args()
    t0 = time.time()
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    if args.rows and args.rows < len(train):
        train = train.sample(n=args.rows, random_state=42).reset_index(drop=True)
    y = (train[fe.TARGET] == "Yes").astype(int).to_numpy()

    prep = prepare(train, test)
    print(f"[{args.tag or 'catboost'}] rows={len(train)} te_keys={len(prep[4])}")

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
    oof_pred = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    importances, used_features = None, []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        X_tr, X_va, X_te, feats, cats = fold_matrices(prep, y, tr_idx, va_idx)
        used_features = feats

        fold_params = dict(params)
        if args.hc_border:
            # 年収・通勤距離とその Target Encoding の列だけ、ビン数を上げる(列の位置で指定する)
            targets = set(fe.HIGHCARD_NUM_COLS) | {f"te{tag}_{c}" for c in fe.HIGHCARD_NUM_COLS
                                                   for tag in ("", "10", "20", "100")}
            fold_params["per_float_feature_quantization"] = [
                f"{i}:border_count={args.hc_border}" for i, name in enumerate(feats)
                if name in targets and name not in cats]

        if args.dump_features:
            catalog = importlib.import_module("03_feature_engineering_AllCatalog")
            catalog.dump(args.tag or "catboost", "CatBoost", feats, cat_features=cats,
                                 note="本番の構成(fe.te_plan())")
            return

        model = CatBoostClassifier(**fold_params)
        model.fit(X_tr, y[tr_idx], cat_features=cats)

        oof_pred[va_idx] = model.predict_proba(X_va)[:, 1]
        if args.save:
            test_pred += model.predict_proba(X_te)[:, 1] / args.folds
        imp = model.get_feature_importance()
        importances = imp if importances is None else importances + imp
        print(f"  fold {fold} AUC: {roc_auc_score(y[va_idx], oof_pred[va_idx]):.5f}  ({time.time() - t0:.0f}s)")

    oof_auc = roc_auc_score(y, oof_pred)
    print(f"RESULT\t{args.tag or 'catboost'}\tOOF AUC: {oof_auc:.5f}\tn_features={len(used_features)}\t"
          f"{time.time() - t0:.0f}s")

    if args.save:
        for d in ("submit", "oof", "importance"):
            os.makedirs(d, exist_ok=True)
        pd.DataFrame({"id": test["id"], fe.TARGET: test_pred}).to_csv(
            f"submit/submission_catboost{args.out_suffix}.csv", index=False)
        np.save(f"oof/oof_catboost{args.out_suffix}.npy", oof_pred)
        np.save(f"oof/pred_catboost{args.out_suffix}.npy", test_pred)

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
        fig.savefig(f"importance/importance_catboost{args.out_suffix}.png", dpi=130)
        print("saved submit/oof/importance artifacts")


if __name__ == "__main__":
    main()
