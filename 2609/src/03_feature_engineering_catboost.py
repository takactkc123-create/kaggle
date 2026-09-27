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
# 1. Arithmetic features
# ---------------------------------------------------------------------------

# 【不採用】数値列の四則演算の列を追加する
def add_arithmetic(df: pd.DataFrame) -> list[str]:
    """Add meaningful arithmetic (diff/ratio/sum/avg) features in place.

    Returns the list of created column names.
    """
    new: list[str] = []

    # 列を追加し、作った列名を記録する
    def put(name: str, values) -> None:
        df[name] = values
        new.append(name)

    eps = 1e-6
    home = df["Charging_Stations_Near_Home"]
    work = df["Charging_Stations_Near_Work"]
    age = df["Age"]
    inc = df["Annual_Income_USD"]
    com = df["Daily_Commute_km"]
    cars = df["Number_of_Cars_Owned"]
    env = df["Environmental_Concern_Level"]

    # charging station combinations
    put("cs_sum", home + work)
    put("cs_diff", home - work)
    put("cs_avg", (home + work) / 2.0)
    put("cs_ratio", home / (work + eps))

    # income based
    put("inc_per_km", inc / (com + eps))
    put("inc_per_age", inc / (age + eps))
    put("inc_per_car", inc / (cars + eps))
    put("inc_x_env", inc * env)

    # commute based
    put("km_per_car", com / (cars + eps))
    put("km_x_age", com * age)
    put("km_per_station", com / (home + work + eps))

    # environment / misc
    put("env_x_cs", env * (home + work))
    put("age_x_env", age * env)
    put("cars_per_age", cars / (age + eps))

    return new


# ---------------------------------------------------------------------------
# 2. Interaction keys (string concatenation of 2 columns)
# ---------------------------------------------------------------------------

INTERACTION_PAIRS = [
    ("City_Type", "Home_Charging_Possible"),
    ("City_Type", "Current_Car_Type"),
    ("Current_Car_Type", "Range_Anxiety_Level"),
    ("Home_Charging_Possible", "Subsidy_Available"),
    ("Range_Anxiety_Level", "Subsidy_Available"),
    ("Gender", "City_Type"),
    ("Charging_Stations_Near_Home", "Charging_Stations_Near_Work"),
    ("Environmental_Concern_Level", "Range_Anxiety_Level"),
    ("Age", "Environmental_Concern_Level"),
    ("City_Type", "Environmental_Concern_Level"),
]


# 【不採用】2 列を連結した交互作用の列を追加する
def add_interactions(
    df: pd.DataFrame, pairs: list[tuple[str, str]] | None = None
) -> list[str]:
    """Add string-concatenated 2-column interaction keys in place."""
    pairs = INTERACTION_PAIRS if pairs is None else pairs
    new: list[str] = []
    for a, b in pairs:
        name = f"ix_{a}_{b}"
        df[name] = df[a].astype(str) + "_" + df[b].astype(str)
        new.append(name)
    return new


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
def add_smooth_keys(df: pd.DataFrame) -> list[str]:
    """Add coarse floor(x / scale) keys used as *additional* TE keys."""
    new: list[str] = []
    for name, src, scale in SMOOTH_KEY_SPECS:
        x = pd.to_numeric(df[src], errors="coerce").fillna(0.0)
        df[name] = np.floor(x / scale).astype("int64")
        new.append(name)
    return new


# ---------------------------------------------------------------------------
# 3. Count / Frequency encoding (unsupervised -> no leak, fit on train+test)
# ---------------------------------------------------------------------------

# 【不採用】値ごとの出現回数の列を追加する
def add_count_encoding(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str], freq: bool = False
) -> list[str]:
    """Add count (or frequency) encoding for `cols`, fitted on train+test."""
    new: list[str] = []
    n_total = len(train) + len(test)
    for col in cols:
        combined = pd.concat([train[col], test[col]], ignore_index=True)
        counts = combined.value_counts()
        name = f"cnt_{col}"
        mapped_tr = train[col].map(counts).astype("float64")
        mapped_te = test[col].map(counts).astype("float64")
        if freq:
            mapped_tr /= n_total
            mapped_te /= n_total
        train[name] = mapped_tr
        test[name] = mapped_te
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
            f"te_{col}" if len(levels) == 1 else f"te{_smooth_tag(s)}_{col}"
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
# 重複している列(同じ情報を形だけ変えて持っている列)の一覧
# ---------------------------------------------------------------------------
# 他の列と同じ情報しか持たない列の名前を返す(--dedup)
def dedup_columns() -> list[str]:
    """他の列と同じ情報しか持たない列の名前を返す(--dedup で学習から外す)。

    - 値の種類（ユニーク値）が少ない11列の TE は、平滑化 10 / 20 / 100 の順位相関が 0.9999 以上になる
      (1値あたり数万行あり、平滑化の強さが効かない)。20 だけ残し 10 / 100 を外す
    - Smooth Key の sk_inc_1 = floor(年収) は、年収が整数なので年収そのものと同じキー。
      その TE 3列は年収の TE と値まで同じ
    - 値が 0〜9 に収まる列の 1 の位(保有台数・環境意識)は外さない。CatBoost では元の列を
      catify でカテゴリにしているので、この2列が「数値として見る」唯一の列になっている
    """
    lowcard = CATEGORICAL_COLS + LOWCARD_NUM_COLS
    te_dups = [f"te{s}_{c}" for c in lowcard for s in ("10", "100")]
    sk_dups = [f"te{s}_sk_inc_1" for s in ("10", "20", "100")]
    return te_dups + sk_dups


# エンコーディングを絞るときに外す列の名前を返す(--lean)
def lean_columns() -> list[str]:
    """エンコーディングを絞るときに外す列(--lean。dedup_columns() の後に適用する)。

    LightGBM と同じ検証(2026-09-27)で、CatBoost でも外して悪化しなかった(単体 +0.000036, z=+1.97)。
    - 値の種類（ユニーク値）が少ない数値列(年齢・スタンド数2列)の digit。保有台数・環境意識の 1 の位は、catify した列を
      数値として見る唯一の列なので残す
    - カテゴリ列の TE。CatBoost は cat_features の購入率を内部で自動計算している
    """
    digits = [f"{c}_d{k}" for c in ("Age", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work")
              for k in (0, 1)]
    return digits + [f"te20_{c}" for c in CATEGORICAL_COLS]
