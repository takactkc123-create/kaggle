"""Feature importance(特徴量の重要度)を確かめる補助スクリプト。

本番と同じ特徴量(04 の prepare / fold_matrices)とパラメータ(05 の FIXED)で **fold 1 だけ**を学習し直し、
重要度の上位の列を print する。本番の fold 1 のモデルと同じものができる(学習の手順・乱数の種が同じため)。

- 重要度は gain(その列での分割が、損失をどれだけ減らしたか)の割合(%)。CatBoost は同じ考え方の
  PredictionValuesChange(その列で予測がどれだけ動いたか)
- 本番の成果物(oof/・submit/)には書き込まない。全列の値は importance/feature_importance_<model>.csv に保存する
- RealMLP(ニューラルネットワーク)は、木のモデルのような重要度を持たないので対象外

使い方:
    uv run src/feature_importance.py                  # LightGBM・XGBoost・CatBoost(約 7 分)
    uv run src/feature_importance.py lgbm --top 20    # モデルと表示する列数を指定
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TARGET = "Will_Buy_EV"
# LightGBM の「eval_set 引数は将来廃止予定」という通知は動作に影響しないので表示しない
warnings.filterwarnings("ignore", message="The argument 'eval_set' is deprecated")
MODELS = ("lgbm", "xgb", "catboost")


# 05 の FIXED(本番の引数の並び)を {引数: 値} にする。値を取らない引数は True、--set-param は key=value をほどく
def fixed_args(model):
    args, out, i = importlib.import_module("05_hyperparameter_tuning").FIXED[model], {}, 0
    while i < len(args):
        flag = args[i]
        has_val = i + 1 < len(args) and not args[i + 1].startswith("--")
        val = args[i + 1] if has_val else True
        if flag == "--set-param":
            flag, val = val.split("=")
        out[flag] = val
        i += 2 if has_val else 1
    return out


# fold 1 の学習行・検証行の行列を、本番(04)と同じ関数で作る
def fold1_matrices(model, train, test, y, tr_idx, va_idx):
    m04 = importlib.import_module(f"04_train_and_evaluate_{model}")
    if model == "lgbm":
        X_tr, X_va, _ = m04.fold_matrices(m04.prepare(train, test), y, tr_idx, va_idx)
        return X_tr, X_va, None
    if model == "xgb":
        X_tr, X_va, _ = m04.fold_matrices(m04.prepare(train, test), pd.Series(y), tr_idx, va_idx)
        return X_tr, X_va, None
    X_tr, X_va, _, feats, cats = m04.fold_matrices(m04.prepare(train, test), y, tr_idx, va_idx)
    return X_tr, X_va, cats


# fold 1 を本番と同じパラメータで学習し、列ごとの重要度(割合)を返す
def importance_of(model, X_tr, y_tr, X_va, y_va, cats):
    f = fixed_args(model)
    if model == "lgbm":
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        clf = LGBMClassifier(random_state=42, verbosity=-1, n_estimators=int(f["--n_estimators"]),
                             learning_rate=float(f["--learning_rate"]), num_leaves=31, max_bin=int(f["--max_bin"]),
                             n_jobs=int(f["--n_jobs"]), colsample_bytree=float(f["--feature_fraction"]),
                             max_depth=int(f["--max_depth"]), importance_type="gain",
                             **({"interaction_constraints": importlib.import_module("04_train_and_evaluate_lgbm")
                                 .interaction_groups(list(X_tr.columns), f["--interaction"])} if "--interaction" in f else {}))
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
                callbacks=[early_stopping(int(f["--early_stopping"]), verbose=False), log_evaluation(0)])
        imp = clf.feature_importances_
    elif model == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(random_state=42, tree_method="hist", n_jobs=int(f["--n-jobs"]), max_bin=int(f["--max-bin"]),
                            colsample_bytree=float(f["colsample_bytree"]), max_depth=int(f["max_depth"]),
                            n_estimators=int(f["--n-estimators"]), learning_rate=float(f["--learning-rate"]),
                            early_stopping_rounds=int(f["--early-stopping"]), eval_metric="auc", importance_type="gain")
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        imp = clf.feature_importances_
    else:
        from catboost import CatBoostClassifier
        cb = importlib.import_module("03_feature_engineering_catboost")
        targets = set(cb.HIGHCARD_NUM_COLS) | {f"te{t}_{c}" for c in cb.HIGHCARD_NUM_COLS for t in ("", "10", "20", "100")}
        params = dict(random_state=42, verbose=False, allow_writing_files=False, iterations=int(f["--iters"]),
                      learning_rate=float(f["--lr"]), boosting_type="Plain", max_ctr_complexity=1,
                      border_count=int(f["--border"]), thread_count=int(f["--threads"]),
                      per_float_feature_quantization=[f"{i}:border_count={f['--hc-border']}"
                                                      for i, name in enumerate(X_tr.columns)
                                                      if name in targets and name not in cats])
        clf = CatBoostClassifier(**params)
        clf.fit(X_tr, y_tr, cat_features=cats)
        imp = clf.get_feature_importance()
    auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
    share = pd.Series(np.asarray(imp, dtype="float64"), index=list(X_tr.columns))
    return (share / share.sum() * 100).sort_values(ascending=False), auc


# 列の種類(元の列 / Target Encoding / Count / digit / Smooth Keys の Target Encoding など)を返す
def kind_of(col: str) -> str:
    if "_sk_" in col or col.startswith(("te_sk_", "inc_f", "commute_f")) or "_sk" in col:
        return "Smooth Keys の Target Encoding"
    if col.startswith("te") or col.endswith(("_tea", "_te10", "_te100")):
        return "Target Encoding"
    if col.startswith("cnt_") or col.endswith("_ce"):
        return "Count"
    if re.search(r"_(digit|d)-?\d+_?$", col):
        return "digit"
    return "元の列"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=list(MODELS), choices=MODELS)
    ap.add_argument("--top", type=int, default=10, help="表示する列の数")
    args = ap.parse_args()

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = (train[TARGET] == "Yes").astype(int).to_numpy()
    tr_idx, va_idx = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train, y))
    os.makedirs("importance", exist_ok=True)

    for model in args.models:
        t0 = time.time()
        X_tr, X_va, cats = fold1_matrices(model, train, test, y, tr_idx, va_idx)
        share, auc = importance_of(model, X_tr, y[tr_idx], X_va, y[va_idx], cats)
        share.rename("share_pct").to_csv(f"importance/feature_importance_{model}.csv")

        print(f"\n{model}: fold 1 の AUC {auc:.5f} / 全 {len(share)} 列 / {time.time() - t0:.0f} 秒")
        print(f"  順位  {'列':<46s} {'割合':>7s}  種類")
        for rank, (col, pct) in enumerate(share.head(args.top).items(), 1):
            print(f"  {rank:>3d}  {col:<46s} {pct:6.2f}%  {kind_of(col)}")
        print(f"  上位 {args.top} 列の合計 {share.head(args.top).sum():.1f}% / 残り {len(share) - args.top} 列で "
              f"{share.iloc[args.top:].sum():.1f}%")


if __name__ == "__main__":
    main()
