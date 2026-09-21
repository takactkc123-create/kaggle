"""Feature engineering functions for the CatBoost model (S6E9).

This module contains ONLY function/constant definitions. No execution code.
`04_fe_run_catboost.py` imports it and runs the CV.

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

def add_arithmetic(df: pd.DataFrame) -> list[str]:
    """Add meaningful arithmetic (diff/ratio/sum/avg) features in place.

    Returns the list of created column names.
    """
    new: list[str] = []

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

def _te_agg(keys: pd.Series, y: np.ndarray) -> pd.DataFrame:
    """sum / count of the target per key (smoothing-independent part)."""
    return pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(
        ["sum", "count"]
    )


def _te_mapping(keys: pd.Series, y: np.ndarray, prior: float, smooth: float) -> pd.Series:
    """Smoothed target mean per key."""
    agg = _te_agg(keys, y)
    return (agg["sum"] + prior * smooth) / (agg["count"] + smooth)


def _smooth_tag(smooth: float) -> str:
    return f"{smooth:g}".replace(".", "p")


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

def cast_to_str(frames: list[pd.DataFrame], cols: list[str]) -> None:
    """Cast columns to string so they can be passed as cat_features."""
    for frame in frames:
        for col in cols:
            frame[col] = frame[col].astype(str)
