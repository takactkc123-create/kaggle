"""RealMLP (NN) 用の特徴量エンジニアリング関数群.

方針:
- ベースは yekenot「PS|S6|E9: RealMLP · PyTorch」(Kaggle 公開ノートブック、Apache License 2.0)の前処理。
  https://www.kaggle.com/code/yekenot/ps-s6-e9-realmlp-pytorch
- GBDT 陣 (fe_lgbm / fe_xgb / fe_catboost) とは意図的に別系統の前処理を使い、
  OOF の非相関性 (多様性) を稼ぐことを目的とする。
- NN 向けなので「カテゴリは整数コード化して embedding / one-hot」「数値は robust scaling」
  という分業になる。数値列のスケーリングは学習側 (NumericalPreprocessor) が担当する。

重要な規約:
- ここには「関数」しか置かない (CLAUDE.md のファイル構成規約)。実行は 04_train_and_evaluate_realmlp.py。
- Target Encoding はリーク防止のため fold 内で fit する。ここでは行わない
  (04_train_and_evaluate_realmlp.py 側で sklearn TargetEncoder を fold 内適用)。
- ここで行う factorize / KBins / count encoding は「目的変数を使わない」変換なので
  train 全体で fit してよい (リークしない)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

TARGET = "Will_Buy_EV"
ID = "id"

# 交互作用キー (yekenot と同一)。fold 内 TE の対象にもなる。
IMPORTANT_COMBOS = [
    ("Annual_Income_USD", "Range_Anxiety_Level"),
    ("Age", "Range_Anxiety_Level"),
]

# KBinsDiscretizer(quantile) の設定 (yekenot と同一)
BIN_CONFIG = {"Annual_Income_USD": [400, 600, 800, 900, 1100]}


# RealMLP 用の特徴量(カテゴリ化・Smooth Keys・ビン・組み合わせキーなど)を作る
def build_features(
    df: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    category_map: dict,
    fit: bool,
    orig: pd.DataFrame | None = None,
    extra_combos: list | None = None,
):
    """yekenot RealMLP カーネルの feature_engineering を再実装した関数.

    Parameters
    ----------
    df : 変換対象 (id / target を落とした後の DataFrame)
    cat_cols, num_cols : 元データの文字列列 / 数値列
    category_map : fit=True で学習された変換器を貯めこむ dict (呼び出し側が保持)
    fit : True なら category_map を構築、False なら再利用
    orig : 元データ (EV_Adoption_and_Range_Anxiety_Dataset.csv)。None なら org_mean をスキップ
    extra_combos : 組み合わせキーに足す列の組(自宅充電の可否 × 自宅スタンド数など)

    Returns
    -------
    (df, new_cat_cols, new_num_cols, combo_names)
    """
    df = df.copy()

    # ── 欠損補完 (本データは欠損ゼロだが公開実装に合わせる) ──────────────────
    for col in cat_cols:
        df[col] = df[col].fillna("missing")
    for col in num_cols:
        df[col] = df[col].fillna(0.0)

    # ── 文字列カテゴリ → 整数コード ───────────────────────────────────────
    for col in cat_cols:
        if fit:
            codes, uniques = df[col].factorize()
            category_map[f"cat::{col}"] = uniques
        else:
            uniques = category_map[f"cat::{col}"]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = df[col].map(code_map).fillna(-1).astype("int32")
        df[col] = np.asarray(codes, dtype="int32")

    # ── 粗い解像度カテゴリ (Smooth Keys) ──────────────────────────────────
    df["Income_/_100_floor_"] = np.floor(df["Annual_Income_USD"] / 100.0).astype("int64")
    df["Income_/_1000_floor_"] = np.floor(df["Annual_Income_USD"] / 1000.0).astype("int64")
    df["Income_/_10000_floor_"] = np.floor(df["Annual_Income_USD"] / 10000.0).astype("int64")

    # ── 数値列の floor をカテゴリ化 (厳密値に近い高解像度キー) ─────────────
    for col in num_cols:
        cat_name = f"{col}_cat_"
        floored = np.floor(df[col])
        if fit:
            codes, uniques = pd.factorize(floored)
            category_map[f"numcat::{col}"] = uniques
        else:
            uniques = category_map[f"numcat::{col}"]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = pd.Series(floored).map(code_map).fillna(-1).astype("int32")
        df[cat_name] = np.asarray(codes, dtype="int32")

    # ── 小数部 / 桁の特徴量 ──────────────────────────────────────────────
    for col in ["Daily_Commute_km"]:
        df[f"_{col}_decimal"] = (df[col] % 1).round(2).astype("float32")
    df["Annual_Income_USD_is_multiple_10_"] = (
        np.floor(df["Annual_Income_USD"]) % 10 == 0
    ).astype("int64")

    # ── 元データ由来の target mean (org_mean) ────────────────────────────
    #    元データはコンペ train/test と別データなのでリークにならない。
    if orig is not None:
        for col in ["Annual_Income_USD"]:
            df[f"_{col}_mean_target_orig"] = (
                df[col]
                .map(orig.groupby(col)[TARGET].mean())
                .fillna(orig[TARGET].mean())
                .astype("float32")
            )

    # ── Count encoding (train で作った頻度表を test に適用) ────────────────
    for col in ["Annual_Income_USD"]:
        count_name = f"_{col}_count"
        if fit:
            count_map = df[col].value_counts()
            category_map[f"count::{col}"] = count_map
        else:
            count_map = category_map[f"count::{col}"]
        df[count_name] = (
            df[col].astype(object).map(count_map).fillna(0).astype("int32")
        )

    # ── KBinsDiscretizer (quantile) による多解像度ビン ─────────────────────
    for col, bins_list in BIN_CONFIG.items():
        for n_bins in bins_list:
            bin_name = f"{col}_{n_bins}_quantile_bin_"
            if fit:
                kb = KBinsDiscretizer(
                    n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
                )
                binned = kb.fit_transform(df[[col]]).ravel().astype("int32")
                category_map[f"kbins::{bin_name}"] = kb
            else:
                kb = category_map[f"kbins::{bin_name}"]
                binned = kb.transform(df[[col]]).ravel().astype("int32")
            df[bin_name] = binned

    # ── 交互作用カテゴリ (fold 内 TE の対象) ──────────────────────────────
    combo_names = []
    for cols in IMPORTANT_COMBOS + list(extra_combos or []):
        combo_name = "_".join(cols) + "_"
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + "_" + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[f"combo::{combo_name}"] = uniques
        else:
            uniques = category_map[f"combo::{combo_name}"]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype("int32")
        df[combo_name] = np.asarray(codes, dtype="int32")

    new_cat_cols = [c for c in df.columns if c.endswith("_")]
    new_num_cols = [c for c in df.columns if c.startswith("_")]
    return df, new_cat_cols, new_num_cols, combo_names




# 元データを読み込み、目的変数を 0/1 にして返す(なければ None)
def load_orig(path: str) -> pd.DataFrame | None:
    """元データを読み込み target を 0/1 化して返す。無ければ None。"""
    import os

    if not os.path.exists(path):
        return None
    orig = pd.read_csv(path)
    if not pd.api.types.is_numeric_dtype(orig[TARGET]):
        orig[TARGET] = (orig[TARGET] == "Yes").astype(int)
    return orig
