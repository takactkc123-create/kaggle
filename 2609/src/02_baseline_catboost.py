import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

target = "Will_Buy_EV"
numeric_cols = [
    c for c in train.select_dtypes(include=[np.number]).columns if c != "id"
]
categorical_cols = [
    c for c in train.select_dtypes(include=["object", "string"]).columns if c != target
]
feature_cols = numeric_cols + categorical_cols
print("numeric features:", numeric_cols)
print("categorical features:", categorical_cols)

X = train[feature_cols]
y = (train[target] == "Yes").astype(int)
X_test = test[feature_cols]

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_pred = np.zeros(len(train))
test_pred = np.zeros(len(test))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    model = CatBoostClassifier(random_state=42, verbose=False)
    model.fit(X_train, y_train, cat_features=categorical_cols)

    oof_pred[valid_idx] = model.predict_proba(X_valid)[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / n_splits

    fold_auc = roc_auc_score(y_valid, oof_pred[valid_idx])
    print(f"fold {fold} AUC: {fold_auc:.5f}")

oof_auc = roc_auc_score(y, oof_pred)
print(f"OOF AUC: {oof_auc:.5f}")

submit_dir = "submit"
import os

os.makedirs(submit_dir, exist_ok=True)

submission = pd.DataFrame({"id": test["id"], target: test_pred})
submission.to_csv(f"{submit_dir}/submission_catboost.csv", index=False)
print(f"saved {submit_dir}/submission_catboost.csv")
