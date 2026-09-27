"""Feature engineering functions for the CatBoost model (S6E9).

This module contains ONLY function/constant definitions. No execution code.
`04_train_and_evaluate_catboost.py` imports it and runs the CV.

Design notes
------------
* Target Encoding is always fitted INSIDE a fold:
  - valid/test get a mapping fitted on the fold's training part
  - the training part itself gets an inner-KFold out-of-fold mapping
  so no row ever sees its own label. (leak-free)
* "Exact value TE" = target encoding applied to numeric columns using their
  raw (exact) values as categories. This was the S6E8 breakthrough.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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

# Numeric columns with low cardinality -> candidates for cat_features / TE
LOWCARD_NUM_COLS = [
    "Age",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

HIGHCARD_NUM_COLS = ["Annual_Income_USD", "Daily_Commute_km"]

# ---------------------------------------------------------------------------
# 2b. Digit features  (reference_URL.md S-1)
# ---------------------------------------------------------------------------

DIGIT_KS = list(range(-4, 4))  # 10^-4 .. 10^3


# 数値列を桁ごとの列(digit features)にばらして追加する
def add_digits(
    df: pd.DataFrame, cols: list[str] | None = None, ks: list[int] | None = None
) -> list[str]:
    """Add `(x // 10**k) % 10` digit columns (int8) for each numeric column.

    The digit is computed on an integer scaled by 1e4 so that float artefacts
    (e.g. 23.4 // 1e-4 == 233999) never leak into the low-order digits.
    Must be called BEFORE `cast_to_str` (catify).
    """
    cols = NUMERIC_COLS if cols is None else cols
    ks = DIGIT_KS if ks is None else ks
    new: list[str] = []
    for col in cols:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        scaled = np.rint(x * 10_000).astype(np.int64)  # 4 decimals of headroom
        for k in ks:
            name = f"{col}_d{k}"
            df[name] = ((scaled // (10 ** (k + 4))) % 10).astype("int8")
            new.append(name)
    return new


# すべてのフレームで値が一定の列を落とし、残った列名を返す
def drop_constant(frames: list[pd.DataFrame], cols: list[str]) -> list[str]:
    """Drop columns that are constant across all given frames. Returns kept."""
    kept: list[str] = []
    for col in cols:
        if any(frame[col].nunique(dropna=False) > 1 for frame in frames):
            kept.append(col)
        else:
            for frame in frames:
                frame.drop(columns=[col], inplace=True)
    return kept


# ---------------------------------------------------------------------------
# 2c. Multi-scale "Smooth Keys"  (reference_URL.md S-3)
# ---------------------------------------------------------------------------

SMOOTH_KEY_SPECS = [
    ("sk_inc_1", "Annual_Income_USD", 1.0),
    ("sk_inc_100", "Annual_Income_USD", 100.0),
    ("sk_inc_1000", "Annual_Income_USD", 1000.0),
    ("sk_com_1", "Daily_Commute_km", 1.0),
]


# 年収・通勤距離を粗く丸めたキー(Smooth Keys)を追加する
def add_smooth_keys(df: pd.DataFrame, specs=None) -> list[str]:
    """Add coarse floor(x / scale) keys used as *additional* TE keys."""
    new: list[str] = []
    for name, src, scale in (SMOOTH_KEY_SPECS if specs is None else specs):
        x = pd.to_numeric(df[src], errors="coerce").fillna(0.0)
        df[name] = np.floor(x / scale).astype("int64")
        new.append(name)
    return new


# ---------------------------------------------------------------------------
# 4. Target Encoding (fold-internal, leak free)
# ---------------------------------------------------------------------------

# キーごとの購入者数と行数を集計する
def _te_agg(keys: pd.Series, y: np.ndarray) -> pd.DataFrame:
    """sum / count of the target per key (smoothing-independent part)."""
    return pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(
        ["sum", "count"]
    )


# 集計から平滑化した購入率の対応表を作る
def _te_mapping(keys: pd.Series, y: np.ndarray, prior: float, smooth: float) -> pd.Series:
    """Smoothed target mean per key."""
    agg = _te_agg(keys, y)
    return (agg["sum"] + prior * smooth) / (agg["count"] + smooth)


# 平滑化の強さを列名用の文字列にする
def _smooth_tag(smooth: float) -> str:
    return f"{smooth:g}".replace(".", "p")


# fold 内で Out-of-Fold の Target Encoding を作って追加する
def target_encode(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    te: pd.DataFrame,
    y_tr: np.ndarray,
    cols: list[str],
    smooth: float = 20.0,
    smooths: list[float] | None = None,
    n_inner: int = 5,
    seed: int = 42,
    drop_source: bool = False,
    always_tag: bool = False,
) -> list[str]:
    """Leak-free target encoding.

    `tr` gets inner out-of-fold values, `va`/`te` get values fitted on all of `tr`.
    When `smooths` holds several values, one column per smoothing level is
    produced (Triple TE, reference_URL.md S-2); the sum/count aggregation is
    shared across levels so the extra levels are nearly free.
    All three frames are modified in place. Returns created column names.
    """
    levels = [smooth] if not smooths else list(smooths)
    prior = float(y_tr.mean())
    new: list[str] = []

    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    inner_splits = list(inner.split(np.zeros(len(tr)), y_tr))

    for col in cols:
        keys_tr = tr[col].astype(str)
        keys_va = va[col].astype(str)
        keys_te = te[col].astype(str)
        names = [
            f"te_{col}" if len(levels) == 1 and not always_tag else f"te{_smooth_tag(s)}_{col}"
            for s in levels
        ]

        # --- valid / test: fit on the whole training part
        full_agg = _te_agg(keys_tr, y_tr)
        for s, name in zip(levels, names):
            m = (full_agg["sum"] + prior * s) / (full_agg["count"] + s)
            va[name] = keys_va.map(m).astype("float64").fillna(prior)
            te[name] = keys_te.map(m).astype("float64").fillna(prior)

        # --- train: inner out-of-fold
        oof = {name: np.full(len(tr), prior, dtype="float64") for name in names}
        for in_idx, out_idx in inner_splits:
            agg = _te_agg(keys_tr.iloc[in_idx], y_tr[in_idx])
            keys_out = keys_tr.iloc[out_idx]
            for s, name in zip(levels, names):
                m = (agg["sum"] + prior * s) / (agg["count"] + s)
                vals = keys_out.map(m).astype("float64").to_numpy()
                oof[name][out_idx] = np.where(np.isnan(vals), prior, vals)
        for name in names:
            tr[name] = oof[name]
        new += names

    if drop_source:
        for col in cols:
            for frame in (tr, va, te):
                frame.drop(columns=[col], inplace=True)

    return new


# ---------------------------------------------------------------------------
# 5. CatBoost-specific: cast low cardinality numerics to string categories
# ---------------------------------------------------------------------------

# 列を文字列にして cat_features に渡せる形にする(catify)
def cast_to_str(frames: list[pd.DataFrame], cols: list[str]) -> None:
    """Cast columns to string so they can be passed as cat_features."""
    for frame in frames:
        for col in cols:
            frame[col] = frame[col].astype(str)


# ---------------------------------------------------------------------------
# 本番の構成(作る列の設計)
# ---------------------------------------------------------------------------
TRIPLE_SMOOTHS = [10.0, 20.0, 100.0]
# digit を作る列。保有台数・環境意識は 1 の位だけになる(catify した元の列を、数値として見る唯一の列)
DIGIT_COLS = HIGHCARD_NUM_COLS + ["Number_of_Cars_Owned", "Environmental_Concern_Level"]
# Smooth Keys。年収 /1 は年収そのものと同じキーになるので作らない
PROD_SMOOTH_KEY_SPECS = [spec for spec in SMOOTH_KEY_SPECS if spec[0] != "sk_inc_1"]


# 本番で作る Target Encoding の (キー, 平滑化) の一覧を返す。この順番がそのまま列の並び順になる
def te_plan(smooth_key_names):
    """数値列の順(値の種類が多い2列は平滑化3種、少ない5列は 20 だけ)→ Smooth Keys 3本(3種)。

    カテゴリ列には作らない(CatBoost は cat_features の購入率を内部で計算している)。
    列の並び順は学習結果に影響する(列ごとのビン数の指定が列の位置で決まる)ので変えないこと。
    """
    plan = [(c, TRIPLE_SMOOTHS if c in HIGHCARD_NUM_COLS else [20.0]) for c in NUMERIC_COLS]
    plan += [(k, TRIPLE_SMOOTHS) for k in smooth_key_names]
    return plan


# キーごとに平滑化を変えて、fold 内で Out-of-Fold の Target Encoding を作って追加する(plan の順)
def target_encode_plan(tr, va, te, y_tr, plan, n_inner: int = 5, seed: int = 42) -> list[str]:
    """te_plan() のとおりに Target Encoding を作る。値は target_encode と同じ(キーごとに独立に計算するため)。"""
    new: list[str] = []
    for col, smooths in plan:
        new += target_encode(tr, va, te, y_tr, [col], smooths=list(smooths), n_inner=n_inner, seed=seed,
                             always_tag=True)
    return new
