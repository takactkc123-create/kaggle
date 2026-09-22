"""Feature engineering functions for XGBoost (S6E9).

This module defines FE *functions only*. No execution code.
`04_fe_run_xgb.py` imports these and runs CV.

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


def make_base(train: pd.DataFrame, test: pd.DataFrame):
    """Baseline feature frames: numeric as-is + categorical as pandas Categorical.

    Mirrors 02_bl_xgb.py (enable_categorical=True path).
    """
    tr = train[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    te = test[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    return as_native_category(tr, te, CATEGORICAL_COLS)


def as_native_category(tr: pd.DataFrame, te: pd.DataFrame, cols):
    """Align category sets across train/test and cast to pandas Categorical."""
    tr = tr.copy()
    te = te.copy()
    for c in cols:
        cats = pd.concat([tr[c].astype(str), te[c].astype(str)]).astype("category").cat.categories
        tr[c] = pd.Categorical(tr[c].astype(str), categories=cats)
        te[c] = pd.Categorical(te[c].astype(str), categories=cats)
    return tr, te


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


def as_onehot(tr: pd.DataFrame, te: pd.DataFrame, cols):
    """One-hot encoding of the given columns (drop original)."""
    n_tr = len(tr)
    both = pd.concat([tr, te], axis=0, ignore_index=True)
    for c in cols:
        both[c] = both[c].astype(str)
    both = pd.get_dummies(both, columns=list(cols), dtype="int8")
    return both.iloc[:n_tr].reset_index(drop=True), both.iloc[n_tr:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. arithmetic features
# ---------------------------------------------------------------------------

# Semantically meaningful pairs (first pass).
MEANINGFUL_PAIRS = [
    ("Charging_Stations_Near_Home", "Charging_Stations_Near_Work"),
    ("Annual_Income_USD", "Daily_Commute_km"),
    ("Age", "Annual_Income_USD"),
    ("Annual_Income_USD", "Number_of_Cars_Owned"),
    ("Daily_Commute_km", "Charging_Stations_Near_Work"),
    ("Environmental_Concern_Level", "Annual_Income_USD"),
    ("Age", "Number_of_Cars_Owned"),
]

_EPS = 1e-6


def _arith_block(df: pd.DataFrame, pairs, ops=("diff", "ratio", "sum", "avg")):
    out = {}
    for a, b in pairs:
        va = df[a].astype("float32")
        vb = df[b].astype("float32")
        if "diff" in ops:
            out[f"{a}_minus_{b}"] = va - vb
        if "sum" in ops:
            out[f"{a}_plus_{b}"] = va + vb
        if "avg" in ops:
            out[f"{a}_avg_{b}"] = (va + vb) / 2.0
        if "ratio" in ops:
            out[f"{a}_div_{b}"] = va / (vb + _EPS)
    return pd.DataFrame(out, index=df.index)


def add_arithmetic(tr, te, src_tr, src_te, pairs=None, ops=("diff", "ratio", "sum", "avg")):
    """Append arithmetic combinations of numeric columns.

    `src_*` are the raw frames holding the numeric columns.
    """
    pairs = MEANINGFUL_PAIRS if pairs is None else pairs
    return (
        pd.concat([tr, _arith_block(src_tr, pairs, ops)], axis=1),
        pd.concat([te, _arith_block(src_te, pairs, ops)], axis=1),
    )


def all_numeric_pairs():
    """【打ち止め】数値列の全2列ペアを列挙する (四則演算用)."""
    pairs = []
    for i, a in enumerate(NUMERIC_COLS):
        for b in NUMERIC_COLS[i + 1 :]:
            pairs.append((a, b))
    return pairs


# ---------------------------------------------------------------------------
# 2. count / frequency encoding  (target-free -> fit on train+test)
# ---------------------------------------------------------------------------


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


def _smooth_key_block(src: pd.DataFrame, specs) -> pd.DataFrame:
    out = {}
    for name, col, div in specs:
        v = src[col].astype("float64").values
        out[name] = np.floor(v / div).astype("int64")
    return pd.DataFrame(out, index=src.index)


def make_smooth_keys(src_tr, src_te, specs=SMOOTH_KEY_SPECS):
    """Coarse-resolution keys of the high-cardinality numeric columns.

    Returned as *extra raw frames* so they can be fed to target / count
    encoding without being added to the model as raw numeric columns
    (the exact values are already there).
    """
    return _smooth_key_block(src_tr, specs), _smooth_key_block(src_te, specs)


# ---------------------------------------------------------------------------
# 3. categorical / value interaction keys
# ---------------------------------------------------------------------------


def make_interaction_keys(src_tr, src_te, pairs):
    """Build string keys for column pairs -> returned as extra raw frames."""
    ktr = pd.DataFrame(index=src_tr.index)
    kte = pd.DataFrame(index=src_te.index)
    for a, b in pairs:
        name = f"{a}__x__{b}"
        ktr[name] = src_tr[a].astype(str) + "|" + src_tr[b].astype(str)
        kte[name] = src_te[a].astype(str) + "|" + src_te[b].astype(str)
    return ktr, kte


def cat_pairs(cols=None):
    """【打ち止め】カテゴリ列の2列ペアを列挙する (交互作用TE用)."""
    cols = CATEGORICAL_COLS if cols is None else cols
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            out.append((a, b))
    return out


# ---------------------------------------------------------------------------
# 4. Target Encoding (fold-safe)
# ---------------------------------------------------------------------------


def fit_target_encoding(src_fit: pd.DataFrame, y_fit, cols, smoothing=20.0, min_samples=1):
    """Fit smoothed target encoding maps on the *training fold only*.

    Returns a dict {col: (mapping Series, prior)}.
    """
    y_fit = pd.Series(np.asarray(y_fit), index=src_fit.index)
    prior = float(y_fit.mean())
    maps = {}
    for c in cols:
        key = src_fit[c]
        grp = y_fit.groupby(key, observed=True)
        agg = grp.agg(["sum", "count"])
        smooth = (agg["sum"] + prior * smoothing) / (agg["count"] + smoothing)
        if min_samples > 1:
            smooth = smooth.where(agg["count"] >= min_samples, prior)
        maps[c] = (smooth.astype("float32"), prior)
    return maps


def apply_target_encoding(src: pd.DataFrame, maps, suffix="_te"):
    """Apply fitted TE maps. Unseen values fall back to the fold prior."""
    out = {}
    for c, (mapping, prior) in maps.items():
        out[f"{c}{suffix}"] = src[c].map(mapping).astype("float32").fillna(np.float32(prior))
    return pd.DataFrame(out, index=src.index)


def fit_apply_te_cv(src_tr, src_te, y, cols, train_idx, valid_idx, smoothing=20.0, min_samples=1):
    """Convenience: fit on train_idx rows, return (te_train, te_valid, te_test).

    Simple variant: the training rows receive the statistic computed from the
    whole training fold (including themselves). Safe w.r.t. the validation fold,
    but the training rows see a slightly optimistic encoding.
    """
    maps = fit_target_encoding(
        src_tr.iloc[train_idx], np.asarray(y)[train_idx], cols, smoothing, min_samples
    )
    return (
        apply_target_encoding(src_tr.iloc[train_idx], maps),
        apply_target_encoding(src_tr.iloc[valid_idx], maps),
        apply_target_encoding(src_te, maps),
    )


def fit_apply_te_cv_nested(
    src_tr,
    src_te,
    y,
    cols,
    train_idx,
    valid_idx,
    smoothing=20.0,
    min_samples=1,
    n_inner=5,
    seed=42,
):
    """Double-protected target encoding.

    - Everything is fitted strictly inside the OUTER training fold.
    - TRAINING rows get inner out-of-fold values (inner StratifiedKFold), so a
      row never sees its own label through the encoding.
    - VALIDATION and TEST rows get the statistic of the whole outer training fold.

    This removes the optimistic bias on the training rows and is what makes the
    exact-value TE actually pay off.
    """
    from sklearn.model_selection import StratifiedKFold

    y_arr = np.asarray(y)
    fit_src = src_tr.iloc[train_idx].reset_index(drop=True)
    fit_y = y_arr[train_idx]

    # training rows: inner OOF encoding
    te_train = pd.DataFrame(
        np.zeros((len(fit_src), len(cols)), dtype="float32"),
        columns=[f"{c}_te" for c in cols],
    )
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for in_tr, in_va in inner.split(fit_src, fit_y):
        maps = fit_target_encoding(
            fit_src.iloc[in_tr], fit_y[in_tr], cols, smoothing, min_samples
        )
        te_train.iloc[in_va] = apply_target_encoding(fit_src.iloc[in_va], maps).values

    # validation / test: statistic of the full outer training fold
    full_maps = fit_target_encoding(fit_src, fit_y, cols, smoothing, min_samples)
    return (
        te_train,
        apply_target_encoding(src_tr.iloc[valid_idx], full_maps),
        apply_target_encoding(src_te, full_maps),
    )


# ---------------------------------------------------------------------------
# 4b. Triple Target Encoding: several smoothing strengths as separate columns
# ---------------------------------------------------------------------------


def _smooth_tag(sm):
    if sm == "auto":
        return "a"
    f = float(sm)
    return str(int(f)) if f == int(f) else str(f).replace(".", "p")


def fit_target_encoding_multi(src_fit, y_fit, cols, smoothings=(20.0,), min_samples=1):
    """Fit TE maps for *several* smoothing strengths at once.

    The per-column groupby is done once and reused for every smoothing value,
    so k smoothings cost far less than k separate passes.

    ``smoothings`` accepts floats (m-estimate: ``(sum + prior*m)/(count + m)``)
    and the string ``"auto"``, which reproduces sklearn's empirical-Bayes
    shrinkage ``lambda = var(y)*n / (var(y)*n + var_within)``. For a binary
    target ``var_within = p_i*(1-p_i)``, so no extra aggregation is needed.

    Returns {out_col_name: (src_col, mapping Series, prior)}.
    """
    y_ser = pd.Series(np.asarray(y_fit, dtype="float64"), index=src_fit.index)
    prior = float(y_ser.mean())
    y_var = float(y_ser.var(ddof=0))
    maps = {}
    for c in cols:
        agg = y_ser.groupby(src_fit[c], observed=True).agg(["sum", "count"])
        s = agg["sum"].to_numpy()
        n = agg["count"].to_numpy()
        mean_cat = s / n
        for sm in smoothings:
            if sm == "auto":
                var_cat = mean_cat * (1.0 - mean_cat)
                denom = y_var * n + var_cat
                lam = np.where(denom > 0, (y_var * n) / np.where(denom > 0, denom, 1.0), 1.0)
                enc = lam * mean_cat + (1.0 - lam) * prior
            else:
                m = float(sm)
                enc = (s + prior * m) / (n + m)
            if min_samples > 1:
                enc = np.where(n >= min_samples, enc, prior)
            name = f"{c}_te{_smooth_tag(sm)}"
            maps[name] = (c, pd.Series(enc.astype("float32"), index=agg.index), prior)
    return maps


def apply_target_encoding_multi(src: pd.DataFrame, maps):
    """複数 smooth の TE マップをまとめて適用する."""
    out = {}
    for name, (col, mapping, prior) in maps.items():
        out[name] = src[col].map(mapping).astype("float32").fillna(np.float32(prior))
    return pd.DataFrame(out, index=src.index)


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
# 5. misc
# ---------------------------------------------------------------------------


def add_row_aggregates(tr, te, src_tr, src_te):
    """Simple row-wise aggregates over the charging-station / concern block."""
    tr = tr.copy()
    te = te.copy()
    for frame, src in ((tr, src_tr), (te, src_te)):
        home = src["Charging_Stations_Near_Home"].astype("float32")
        work = src["Charging_Stations_Near_Work"].astype("float32")
        frame["charge_total"] = home + work
        frame["charge_min"] = np.minimum(home, work)
        frame["charge_max"] = np.maximum(home, work)
    return tr, te
