"""RealMLP (NN) 用の特徴量エンジニアリング関数群.

方針:
- ベースは公開カーネル yekenot/ps-s6-e9-realmlp-pytorch (kernels/yekenot_realmlp/) の前処理。
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

# digit features の対象。低カーデ列は既に厳密値の embedding を持つので高カーデ2列に限る
DIGIT_COLS = ["Annual_Income_USD", "Daily_Commute_km"]

# KBinsDiscretizer(quantile) の設定 (yekenot と同一)
BIN_CONFIG = {"Annual_Income_USD": [400, 600, 800, 900, 1100]}


def build_features(
    df: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    category_map: dict,
    fit: bool,
    orig: pd.DataFrame | None = None,
    digits: bool = False,
):
    """yekenot RealMLP カーネルの feature_engineering を再実装した関数.

    Parameters
    ----------
    df : 変換対象 (id / target を落とした後の DataFrame)
    cat_cols, num_cols : 元データの文字列列 / 数値列
    category_map : fit=True で学習された変換器を貯めこむ dict (呼び出し側が保持)
    fit : True なら category_map を構築、False なら再利用
    orig : 元データ (EV_Adoption_and_Range_Anxiety_Dataset.csv)。None なら org_mean をスキップ

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

    # ── 四則演算 / 粗い解像度カテゴリ (Smooth Keys) ───────────────────────
    df["_Daily_Commute_km_/_Age"] = (
        df["Daily_Commute_km"] / (df["Age"] + 1e-6)
    ).astype("float32")
    df["Income_/_100_floor_"] = np.floor(df["Annual_Income_USD"] / 100.0).astype("int64")
    df["Income_/_1000_floor_"] = np.floor(df["Annual_Income_USD"] / 1000.0).astype("int64")
    df["Income_/_10000_floor_"] = np.floor(df["Annual_Income_USD"] / 10000.0).astype("int64")
    df["Daily_km_/_5_floor_"] = np.floor(df["Daily_Commute_km"] / 5.0).astype("int64")

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

    # ── digit features (高カーデ2列の各桁をカテゴリとして embedding に渡す) ──
    #    全列が小数1桁なので10倍して整数化し、浮動小数の丸め誤差を避ける。
    #    定数になる桁は fit 時に落とし、test でも同じ列集合を使う。
    if digits:
        if fit:
            keep = []
            for col in DIGIT_COLS:
                scaled = np.round(df[col] * 10).astype("int64")
                for p in range(7):
                    if ((scaled // 10**p) % 10).nunique() > 1:
                        keep.append((col, p))
            category_map["digits"] = keep
        for col, p in category_map["digits"]:
            scaled = np.round(df[col] * 10).astype("int64")
            df[f"{col}_d{p - 1}_"] = ((scaled // 10**p) % 10).astype("int32")

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
    for cols in IMPORTANT_COMBOS:
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


# ══════════════════════════════════════════════════════════════════════════════
# 厳密値 Target Encoding(高カーデ2列に絞る版, 2026-09-18)
#
# 背景: GBDT 3種は「13列の厳密値TE + Smooth Keys の Triple TE」で各 +0.003 前後の
# 改善を得ているが、RealMLP には combo TE (income×RangeAnxiety, age×RangeAnxiety の
# 2列のみ, 04_train_and_evaluate_realmlp.py 側で適用) しか入っていなかった。
# fe_results_all.md の分析により、54列一括投入はCPU競合で完走せずコストも高いため、
# 効果源が確実な高カーディナリティ2列 + その Smooth Keys 3本 = 5キー に絞り込む。
# smooth を auto/10/100 の Triple で同時投入 -> 5キー × 3 smooth = 15列。
#
# リーク対策: 入れ子CV (inner StratifiedKFold(5)) を outer fold 内で回し、学習行には
# inner-OOF 値、valid/test には学習fold全体の統計を当てる (fe_lgbm/fe_all と同方式)。
# ══════════════════════════════════════════════════════════════════════════════
HIGHCARD_TE_COLS = ["Annual_Income_USD", "Daily_Commute_km"]
SMOOTH_KEY_SCALES_TE = (10, 100, 1000)
EXACT_TE_SMOOTHS = ("auto", 10.0, 100.0)


def build_te_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    """厳密値2列 + income の Smooth Keys(/10,/100,/1000) = 5キーのフレームを返す.

    厳密値キーは小数1桁を ×10 して整数化し、float 等価判定の揺れを排除する
    (fe_lgbm.make_key_frame と同方式)。教師変数は使わないので train/test それぞれに
    直接適用してよい (リークしない)。
    """
    keys = pd.DataFrame(index=df.index)
    for c in HIGHCARD_TE_COLS:
        keys[c] = np.rint(df[c].to_numpy(dtype="float64") * 10).astype("int64")
    inc = df["Annual_Income_USD"].to_numpy(dtype="float64")
    for s in SMOOTH_KEY_SCALES_TE:
        keys[f"sk_inc{s}"] = np.floor(inc / s).astype("int64")
    return keys


def _te_agg(arr, y):
    return pd.DataFrame({"k": arr, "y": y}).groupby("k", observed=True)["y"].agg(
        ["sum", "count"]
    )


def _te_map(agg, prior, smooth):
    cnt = agg["count"].to_numpy(dtype="float64")
    s = agg["sum"].to_numpy(dtype="float64")
    if isinstance(smooth, str):  # "auto" = sklearn TargetEncoder の経験ベイズ則
        p_i = s / cnt
        m = (p_i * (1.0 - p_i)) / (prior * (1.0 - prior))
    else:
        m = float(smooth)
    return pd.Series((s + prior * m) / (cnt + m), index=agg.index)


def _smooth_tag(sm):
    if isinstance(sm, str):
        return sm
    f = float(sm)
    return str(int(f)) if f == int(f) else str(f).replace(".", "p")


def target_encode_highcard(
    keys_fit: pd.DataFrame,
    y_fit,
    other_frames: list[pd.DataFrame],
    cols,
    smooths=EXACT_TE_SMOOTHS,
    n_inner: int = 5,
    seed: int = 42,
):
    """リークフリー入れ子 TE。戻り値 (te_fit, [te_other, ...]).

    keys_fit     : 現在の outer fold の学習行のキーフレーム
    other_frames : 同じ統計を当てるフレーム (通常 [valid_keys, test_keys])
    smooths      : float または "auto" のリスト。複数指定で Triple TE。
    """
    from sklearn.model_selection import StratifiedKFold

    smooths = list(smooths) if isinstance(smooths, (list, tuple)) else [smooths]
    multi = len(smooths) > 1
    y_fit = np.asarray(y_fit)
    prior = float(y_fit.mean())
    te_fit = pd.DataFrame(index=keys_fit.index)
    te_others = [pd.DataFrame(index=f.index) for f in other_frames]

    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    inner_splits = list(inner.split(np.zeros(len(y_fit)), y_fit))

    for c in cols:
        arr = keys_fit[c].to_numpy()
        names = {
            sm: (f"te_{c}_s{_smooth_tag(sm)}" if multi else f"te_{c}") for sm in smooths
        }

        vals = {sm: np.full(len(arr), prior, dtype="float32") for sm in smooths}
        for in_idx, out_idx in inner_splits:
            agg = _te_agg(arr[in_idx], y_fit[in_idx])
            s_out = pd.Series(arr[out_idx])
            for sm in smooths:
                vals[sm][out_idx] = (
                    s_out.map(_te_map(agg, prior, sm)).fillna(prior).to_numpy(dtype="float32")
                )
        for sm in smooths:
            te_fit[names[sm]] = vals[sm]

        agg_full = _te_agg(arr, y_fit)
        for sm in smooths:
            m_full = _te_map(agg_full, prior, sm)
            for f, out in zip(other_frames, te_others):
                out[names[sm]] = (
                    f[c].map(m_full).fillna(prior).to_numpy(dtype="float32")
                )
    return te_fit, te_others


def load_orig(path: str) -> pd.DataFrame | None:
    """元データを読み込み target を 0/1 化して返す。無ければ None。"""
    import os

    if not os.path.exists(path):
        return None
    orig = pd.read_csv(path)
    if not pd.api.types.is_numeric_dtype(orig[TARGET]):
        orig[TARGET] = (orig[TARGET] == "Yes").astype(int)
    return orig
