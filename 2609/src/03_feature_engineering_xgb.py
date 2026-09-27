"""Feature engineering functions for XGBoost (S6E9).

This module defines FE *functions only*. No execution code.
`04_train_and_evaluate_xgb.py` imports these and runs CV.

Two kinds of functions:
  1. Fold-independent transforms (arithmetic, count/frequency encoding,
     categorical interaction keys, encoding-style conversion).
     These are safe to compute on the concatenation of train+test
     because they never touch the target.
  2. Fold-dependent transforms (Target Encoding).
     `fit_target_encoding` MUST be fitted on the training fold only and
     applied to the validation fold / test. Leakage is strictly forbidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Will_Buy_EV"

NUMERIC_COLS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

CATEGORICAL_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

# Low-cardinality numeric columns: exact-value TE is especially promising here.
LOW_CARD_NUMERIC = [
    "Age",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

# ---------------------------------------------------------------------------
# 0. base
# ---------------------------------------------------------------------------


# カテゴリ列を整数コードにする
def as_ordinal(tr: pd.DataFrame, te: pd.DataFrame, cols):
    """Ordinal (label) encoding -> plain int codes. XGBoost treats them as numeric."""
    tr = tr.copy()
    te = te.copy()
    for c in cols:
        cats = pd.concat([tr[c].astype(str), te[c].astype(str)]).astype("category").cat.categories
        mapping = {v: i for i, v in enumerate(cats)}
        tr[c] = tr[c].astype(str).map(mapping).astype("int16")
        te[c] = te[c].astype(str).map(mapping).astype("int16")
    return tr, te


# ---------------------------------------------------------------------------
# 2. count / frequency encoding  (target-free -> fit on train+test)
# ---------------------------------------------------------------------------


# 値ごとの出現回数の列を追加する(train と test をまとめて数える)
def add_count_encoding(tr, te, src_tr, src_te, cols, as_frequency=True, suffix="_ce"):
    """Count (or frequency) encoding computed over train+test combined."""
    tr = tr.copy()
    te = te.copy()
    n_total = len(src_tr) + len(src_te)
    for c in cols:
        both = pd.concat([src_tr[c].astype(str), src_te[c].astype(str)], ignore_index=True)
        counts = both.value_counts()
        if as_frequency:
            counts = counts / n_total
        tr[f"{c}{suffix}"] = src_tr[c].astype(str).map(counts).astype("float32")
        te[f"{c}{suffix}"] = src_te[c].astype(str).map(counts).astype("float32")
    return tr, te


# ---------------------------------------------------------------------------
# 2b. digit features  (target-free)
# ---------------------------------------------------------------------------

DIGIT_KS = tuple(range(-4, 4))  # 10^-4 .. 10^3  -> 8 digits per column


# 1 列を桁ごとの列にばらす
def digit_block(df: pd.DataFrame, cols, ks=DIGIT_KS) -> pd.DataFrame:
    """Per-column decimal digits: ``(x // 10**k) % 10`` for k in ``ks``.

    Implemented in fixed point (4 decimals) so that no float rounding noise can
    flip a digit (e.g. 12.3 * 10 == 122.99999999999999 in binary floating point).
    All columns in this dataset have at most 1 decimal, so 4 decimals is exact.
    """
    out = {}
    for c in cols:
        iv = np.rint(df[c].fillna(0).astype("float64").values * 10_000).astype("int64")
        for k in ks:
            out[f"{c}_d{k}"] = (iv // (10 ** (k + 4))) % 10
    return pd.DataFrame(out, index=df.index).astype("int8")


# 数値列を桁ごとの列(digit features)にばらして追加する
def add_digit_features(tr, te, src_tr, src_te, cols=None, ks=DIGIT_KS, drop_constant=True):
    """Append digit features built from the raw numeric columns.

    Columns that are constant across train+test (e.g. the decimal digits of an
    integer-valued column) carry no information and are dropped by default.
    """
    cols = NUMERIC_COLS if cols is None else cols
    dtr = digit_block(src_tr, cols, ks)
    dte = digit_block(src_te, cols, ks)
    if drop_constant:
        keep = [c for c in dtr.columns if dtr[c].nunique() > 1 or dte[c].nunique() > 1]
        dtr, dte = dtr[keep], dte[keep]
    return (
        pd.concat([tr.reset_index(drop=True), dtr.reset_index(drop=True)], axis=1),
        pd.concat([te.reset_index(drop=True), dte.reset_index(drop=True)], axis=1),
    )


# ---------------------------------------------------------------------------
# 2c. multi-scale "smooth keys"  (coarser resolutions of the high-card numerics)
# ---------------------------------------------------------------------------

# NOTE: the reference kernels also list ``floor(income)``, but in this dataset
# Annual_Income_USD is integer-valued, so floor(income) == income and the key
# would be a literal duplicate of the exact-value TE we already have.
SMOOTH_KEY_SPECS = (
    # (output name, source column, divisor)
    ("inc_f100", "Annual_Income_USD", 100),
    ("inc_f1000", "Annual_Income_USD", 1000),
    ("inc_f10000", "Annual_Income_USD", 10000),
    ("commute_f1", "Daily_Commute_km", 1),  # floor(commute)
)


# 1 列を指定の刻みで丸めたキーを作る
def _smooth_key_block(src: pd.DataFrame, specs) -> pd.DataFrame:
    out = {}
    for name, col, div in specs:
        v = src[col].astype("float64").values
        out[name] = np.floor(v / div).astype("int64")
    return pd.DataFrame(out, index=src.index)


# 年収・通勤距離を粗く丸めたキー(Smooth Keys)を作る
def make_smooth_keys(src_tr, src_te, specs=SMOOTH_KEY_SPECS):
    """Coarse-resolution keys of the high-cardinality numeric columns.

    Returned as *extra raw frames* so they can be fed to target / count
    encoding without being added to the model as raw numeric columns
    (the exact values are already there).
    """
    return _smooth_key_block(src_tr, specs), _smooth_key_block(src_te, specs)


# ---------------------------------------------------------------------------
# 4b. Triple Target Encoding: several smoothing strengths as separate columns
# ---------------------------------------------------------------------------


# 平滑化の強さを列名用の文字列にする
def _smooth_tag(sm):
    if sm == "auto":
        return "a"
    f = float(sm)
    return str(int(f)) if f == int(f) else str(f).replace(".", "p")


# Target Encoding のキー列を train・test 共通の整数コードにする
def prepare_te_codes(src_tr, src_te, cols):
    """Factorize every TE key column over train+test ONCE -> int32 codes.

    Doing this up front turns the per-fold encoding into pure numpy
    (bincount + fancy indexing), which is ~30x faster than repeated
    `groupby` / `Series.map` on 500k+ rows and makes Triple TE affordable.
    Factorizing over train+test is target-free, so it introduces no leakage.
    """
    n_tr = len(src_tr)
    codes_tr, codes_te, ncats = {}, {}, {}
    for c in cols:
        both = pd.concat([src_tr[c], src_te[c]], ignore_index=True)
        codes, uniques = pd.factorize(both, sort=False)
        codes = codes.astype("int32")
        codes_tr[c] = codes[:n_tr]
        codes_te[c] = codes[n_tr:]
        ncats[c] = len(uniques)
    return codes_tr, codes_te, ncats


# 1 列について、平滑化ごとの購入率の配列を作る
def _encode_one_column(codes, y, ncat, smoothings, prior, y_var, min_samples):
    """Return one float32 encoding array (indexed by code) per smoothing."""
    cnt = np.bincount(codes, minlength=ncat).astype("float64")
    s = np.bincount(codes, weights=y, minlength=ncat)
    safe_cnt = np.maximum(cnt, 1.0)
    mean_cat = s / safe_cnt
    seen = cnt > 0
    out = []
    for sm in smoothings:
        if sm == "auto":
            var_cat = mean_cat * (1.0 - mean_cat)
            denom = y_var * cnt + var_cat
            lam = np.where(denom > 0, (y_var * cnt) / np.where(denom > 0, denom, 1.0), 0.0)
            enc = lam * mean_cat + (1.0 - lam) * prior
        else:
            m = float(sm)
            enc = (s + prior * m) / (cnt + m)
        enc = np.where(seen, enc, prior)
        if min_samples > 1:
            enc = np.where(cnt >= min_samples, enc, prior)
        out.append(enc.astype("float32"))
    return out


# fold 内で Out-of-Fold の Target Encoding を平滑化 3 種で作る
def fit_apply_te_cv_nested_multi(
    codes_tr,
    codes_te,
    ncats,
    y,
    cols,
    train_idx,
    valid_idx,
    smoothings=(20.0,),
    min_samples=1,
    n_inner=5,
    seed=42,
):
    """Nested (inner-OOF) target encoding with multiple smoothing strengths.

    Same leakage discipline as `fit_apply_te_cv_nested`: everything is fitted
    strictly inside the outer training fold, training rows receive inner
    out-of-fold values, validation/test rows receive the outer-fold statistic.
    Operates on the integer code arrays from `prepare_te_codes`.
    """
    from sklearn.model_selection import StratifiedKFold

    y_arr = np.asarray(y, dtype="float64")
    fit_y = y_arr[train_idx]
    n_fit = len(train_idx)
    n_out = len(cols) * len(smoothings)
    out_names = [f"{c}_te{_smooth_tag(sm)}" for c in cols for sm in smoothings]

    fit_codes = {c: codes_tr[c][train_idx] for c in cols}
    va_codes = {c: codes_tr[c][valid_idx] for c in cols}

    te_train = np.zeros((n_fit, n_out), dtype="float32")
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for in_tr, in_va in inner.split(np.zeros((n_fit, 1)), fit_y):
        y_in = fit_y[in_tr]
        prior = float(y_in.mean())
        y_var = float(y_in.var())
        j = 0
        for c in cols:
            cc = fit_codes[c]
            for arr in _encode_one_column(
                cc[in_tr], y_in, ncats[c], smoothings, prior, y_var, min_samples
            ):
                te_train[in_va, j] = arr[cc[in_va]]
                j += 1

    prior = float(fit_y.mean())
    y_var = float(fit_y.var())
    te_valid = np.zeros((len(valid_idx), n_out), dtype="float32")
    te_test = np.zeros((len(codes_te[cols[0]]), n_out), dtype="float32")
    j = 0
    for c in cols:
        for arr in _encode_one_column(
            fit_codes[c], fit_y, ncats[c], smoothings, prior, y_var, min_samples
        ):
            te_valid[:, j] = arr[va_codes[c]]
            te_test[:, j] = arr[codes_te[c]]
            j += 1

    return (
        pd.DataFrame(te_train, columns=out_names),
        pd.DataFrame(te_valid, columns=out_names),
        pd.DataFrame(te_test, columns=out_names),
    )


# ---------------------------------------------------------------------------
# 本番の構成(作る列の設計)
# ---------------------------------------------------------------------------
ALL_COLS = NUMERIC_COLS + CATEGORICAL_COLS
# 値の種類（ユニーク値）が多い数値列。平滑化3種の Target Encoding はこの2列と Smooth Keys だけに作る
HIGH_CARD_NUMERIC = ["Annual_Income_USD", "Daily_Commute_km"]
TRIPLE_SMOOTHS = ("auto", 10.0, 100.0)
# digit を作る列。保有台数・環境意識は値が 0〜9 なので 1 の位が値そのものになり、作らない
DIGIT_COLS = [c for c in NUMERIC_COLS if c not in ("Number_of_Cars_Owned", "Environmental_Concern_Level")]


# 本番で作る Target Encoding の (キー, 平滑化) の一覧を返す。この順番がそのまま列の並び順になる
def te_plan(smooth_key_names):
    """Smooth Keys 4本(3種)→ 生の 13 列(値の種類が多い2列は3種、少ない11列は auto だけ)。

    カテゴリ列にも作る(XGBoost はカテゴリを整数コードで渡すので、購入率の列が役に立つ)。
    列の並び順は学習結果に影響するので変えないこと。
    """
    plan = [(k, TRIPLE_SMOOTHS) for k in smooth_key_names]
    plan += [(c, TRIPLE_SMOOTHS if c in HIGH_CARD_NUMERIC else ("auto",)) for c in ALL_COLS]
    return plan


# キーごとに平滑化を変えて、fold 内で Out-of-Fold の Target Encoding を作る(plan の順に列を並べる)
def fit_apply_te_cv_nested_plan(codes_tr, codes_te, ncats, y, plan, train_idx, valid_idx, n_inner=5, seed=42):
    """te_plan() のとおりに Target Encoding を作る。値は fit_apply_te_cv_nested_multi と同じ(キーごとに独立に計算するため)。"""
    parts = [fit_apply_te_cv_nested_multi(codes_tr, codes_te, ncats, y, [c], train_idx, valid_idx, tuple(sm),
                                          n_inner=n_inner, seed=seed) for c, sm in plan]
    return tuple(pd.concat([p[i] for p in parts], axis=1) for i in range(3))
