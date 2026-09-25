"""ベースライン(LightGBM)。

EDA の施策1「カテゴリ列を落とさずモデルに渡す」を、ここで検証する。
  1. 数値列のみ(7列)
  2. 数値列 + カテゴリ列(13列)。カテゴリはエンコードせず、LightGBM のネイティブなカテゴリ対応(category dtype)に渡す
どちらもデフォルトパラメータ・同じ fold 分割で学習し、OOF AUC を比べる。
"""
import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

target = "Will_Buy_EV"
numeric_cols = [c for c in train.select_dtypes(include=[np.number]).columns if c != "id"]
categorical_cols = [c for c in train.columns if c not in numeric_cols + ["id", target]]
print("numeric features:", numeric_cols)
print("categorical features:", categorical_cols)

# train/test で共通のカテゴリ集合を定義し、ライブラリのネイティブなカテゴリ対応に渡す
for c in categorical_cols:
    categories = pd.concat([train[c], test[c]]).astype("category").cat.categories
    train[c] = pd.Categorical(train[c], categories=categories)
    test[c] = pd.Categorical(test[c], categories=categories)

y = (train[target] == "Yes").astype(int)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def run_cv(feature_cols, cat_cols):
    """5-fold で学習し、OOF 予測と test 予測(fold 平均)を返す。"""
    X, X_test = train[feature_cols], test[feature_cols]
    oof_pred = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        model = LGBMClassifier(random_state=42, verbosity=-1)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_pred[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits
        print(f"  fold {fold} AUC: {roc_auc_score(y.iloc[valid_idx], oof_pred[valid_idx]):.5f}")
    return oof_pred, test_pred


print("\n[1] 数値列のみ")
oof_num, _ = run_cv(numeric_cols, [])
auc_num = roc_auc_score(y, oof_num)
print(f"OOF AUC: {auc_num:.5f}")

print("\n[2] 数値列 + カテゴリ列(施策1)")
oof_all, test_all = run_cv(numeric_cols + categorical_cols, categorical_cols)
auc_all = roc_auc_score(y, oof_all)
print(f"OOF AUC: {auc_all:.5f}")

print("\n=== LightGBM ベースラインの比較 ===")
print(f"数値列のみ           : {auc_num:.5f}")
print(f"数値列 + カテゴリ列  : {auc_all:.5f}  (差 {auc_all - auc_num:+.5f})")

# 本番モデルの提出ファイル(submission_<model>.csv)と名前がぶつからないよう baseline を付ける
os.makedirs("submit", exist_ok=True)
path = "submit/submission_baseline_lgbm.csv"
pd.DataFrame({"id": test["id"], target: test_all}).to_csv(path, index=False)
print(f"saved {path}")
