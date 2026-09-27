"""Feature engineering functions for the LightGBM line (S6E9).

This module defines FE **functions only** - no execution code.
`04_train_and_evaluate_lgbm.py` imports these and runs the CV.

Design notes
------------
* Leak-free by construction: every target-based encoder is fitted **inside**
  an outer CV fold (`target_encode_fold`), and the training rows themselves get
  an *inner* out-of-fold encoding so the model never sees its own label.
* Count / Frequency encoding is unsupervised -> it can be fitted once on
  train+test concatenated (no label involved, no leakage).
* "Key frames": categorical columns are converted to integer codes so that
  group keys are cheap and dtype-safe. LightGBM still receives the original
  `category` dtype columns.
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

# low-cardinality numeric columns (safe to use as interaction keys)
LOW_CARD_NUMERIC = [
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
    "Age",
]


# --------------------------------------------------------------------------
# base loading / dtype handling
# --------------------------------------------------------------------------
# train.csv と test.csv を読み込む
def load_data(data_dir: str = "data"):
    """Same read_csv flow as 02_baseline_lgbm.py."""
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    return train, test


# カテゴリ列を train・test 共通の水準で category 型にする
def make_categorical(train: pd.DataFrame, test: pd.DataFrame, cols=None):
    """Give train/test a shared category set (LightGBM native categorical)."""
    cols = CATEGORICAL_COLS if cols is None else cols
    train = train.copy()
    test = test.copy()
    for c in cols:
        categories = pd.concat([train[c], test[c]]).astype("category").cat.categories
        train[c] = pd.Categorical(train[c], categories=categories)
        test[c] = pd.Categorical(test[c], categories=categories)
    return train, test


# 13 列すべてを値のまま整数キーにしたフレームを作る
def make_key_frame(train: pd.DataFrame, test: pd.DataFrame):
    """Integer-coded copies of all 13 raw columns, used as encoder group keys.

    Numeric columns keep their *exact* values (the "exact value TE" idea from
    S6E8); categorical columns become int8 codes.
    """
    keys_tr = pd.DataFrame(index=train.index)
    keys_te = pd.DataFrame(index=test.index)
    for c in NUMERIC_COLS:
        keys_tr[c] = train[c].to_numpy()
        keys_te[c] = test[c].to_numpy()
    for c in CATEGORICAL_COLS:
        cats = pd.concat([train[c], test[c]]).astype("category").cat.categories
        keys_tr[c] = pd.Categorical(train[c], categories=cats).codes.astype("int16")
        keys_te[c] = pd.Categorical(test[c], categories=cats).codes.astype("int16")
    return keys_tr, keys_te


# --------------------------------------------------------------------------
# 1b. digit features  (round 3 - from reference_URL.md S-1)
# --------------------------------------------------------------------------
DIGIT_K = list(range(-4, 4))  # 10^-4 .. 10^3


# 数値列を桁ごとの列(digit features)にばらす
def add_digit_features(
    df: pd.DataFrame, cols=None, ks=None, keep: list | None = None
) -> pd.DataFrame:
    """`(x // 10**k) % 10` as an int8 column, for every numeric column and k.

    Why this works (reference_URL.md): with the default ``max_bin=255`` the
    13k distinct values of ``Annual_Income_USD`` are squeezed into 255 bins
    (51.8 values per bin), so the value-level structure is invisible to the
    histogram splitter.  Exposing each decimal digit as its own column gives
    the tree a second, exact route to the same information.

    Constant columns (e.g. the 10^-4 digit of an integer-valued column) are
    dropped - they only slow the fit down.  Pass ``keep`` to force an explicit
    column list so that train and test always end up identical.
    """
    cols = NUMERIC_COLS if cols is None else cols
    ks = DIGIT_K if ks is None else ks
    out = {}
    for c in cols:
        x = df[c].fillna(0).to_numpy(dtype="float64")
        for k in ks:
            name = f"{c}_digit{k}"
            if keep is not None and name not in keep:
                continue
            scaled = x * (10.0 ** (-k))
            arr = np.mod(np.floor(scaled + 1e-9), 10.0).astype("int8")
            if keep is None and arr.min() == arr.max():  # constant -> useless
                continue
            out[name] = arr
    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------
# 1c. multi-scale "smooth keys"  (reference_URL.md S-3)
# --------------------------------------------------------------------------
# 年収・通勤距離を粗く丸めたキー(Smooth Keys)を追加する
def add_smooth_keys(keys: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Coarser resolutions of the two high-cardinality numeric columns.

    The exact-value keys stay untouched; these are *added* on top so that TE /
    count encoding also see a coarse, better-populated view of income and
    commute distance.

    ``floor(Annual_Income_USD)`` is intentionally omitted: the column is
    integer valued, so it would be a bit-identical duplicate of the existing
    exact-value key.  ``floor(income/10)`` is used in its place.
    """
    keys = keys.copy()
    inc = df["Annual_Income_USD"].fillna(0).to_numpy(dtype="float64")
    km = df["Daily_Commute_km"].fillna(0).to_numpy(dtype="float64")
    keys["sk_inc10"] = np.floor(inc / 10.0).astype("int32")
    keys["sk_inc100"] = np.floor(inc / 100.0).astype("int32")
    keys["sk_inc1000"] = np.floor(inc / 1000.0).astype("int32")
    keys["sk_commute"] = np.floor(km).astype("int32")
    return keys


SMOOTH_KEYS = ["sk_inc10", "sk_inc100", "sk_inc1000", "sk_commute"]


# --------------------------------------------------------------------------
# 2. count / frequency encoding (unsupervised -> fit on train+test)
# --------------------------------------------------------------------------
# 1 列または複数列のキーを 1 本の系列にする
def _key_series(frame: pd.DataFrame, key):
    if isinstance(key, str):
        return frame[key]
    codes = None
    for c in key:
        v = frame[c]
        codes = v.astype(str) if codes is None else codes + "_" + v.astype(str)
    return codes


# キーの値ごとの出現回数を列にする(train と test をまとめて数える)
def count_encode(keys_tr: pd.DataFrame, keys_te: pd.DataFrame, keys, freq: bool = True):
    """Count / frequency encoding fitted on train+test concatenated.

    Returns (DataFrame_train, DataFrame_test).
    """
    out_tr = pd.DataFrame(index=keys_tr.index)
    out_te = pd.DataFrame(index=keys_te.index)
    n_total = len(keys_tr) + len(keys_te)
    for key in keys:
        name = key if isinstance(key, str) else "X".join(key)
        s_tr = _key_series(keys_tr, key)
        s_te = _key_series(keys_te, key)
        vc = pd.concat([s_tr, s_te], ignore_index=True).value_counts()
        col = f"cnt_{name}"
        a = s_tr.map(vc).astype("float32")
        b = s_te.map(vc).astype("float32")
        if freq:
            a = a / n_total
            b = b / n_total
        out_tr[col] = a.to_numpy()
        out_te[col] = b.to_numpy()
    return out_tr, out_te


# --------------------------------------------------------------------------
# 3. target encoding (fold-internal, leak free)
# --------------------------------------------------------------------------
# キーごとの購入者数と行数を集計する
def _te_agg(key_s: pd.Series, y: np.ndarray):
    """(sum, count) per key - the sufficient statistics shared by all smooths."""
    g = pd.DataFrame({"k": key_s.to_numpy(), "y": y}).groupby("k", observed=True)["y"]
    return g.agg(["sum", "count"])


# 集計から平滑化した購入率の対応表を作る
def _te_map_from_agg(agg: pd.DataFrame, prior: float, smooth):
    """Shrunken mean per key.

    ``smooth`` is either a float (classic m-estimate) or ``"auto"``, which
    reproduces sklearn's ``TargetEncoder(smooth="auto")`` empirical-Bayes rule:
    ``m_i = var_within_i / var_total`` which for a binary target is
    ``p_i(1-p_i) / (p(1-p))`` - i.e. an almost unsmoothed encoding for pure
    categories and a strongly shrunken one for 50/50 categories.
    """
    cnt = agg["count"].to_numpy(dtype="float64")
    s = agg["sum"].to_numpy(dtype="float64")
    if isinstance(smooth, str):  # "auto"
        p_i = s / cnt
        var_i = p_i * (1.0 - p_i)
        var_y = prior * (1.0 - prior)
        m = var_i / var_y
    else:
        m = float(smooth)
    vals = (s + prior * m) / (cnt + m)
    return pd.Series(vals, index=agg.index)


# キーと目的変数から、平滑化した購入率の対応表を作る
def _fit_te_map(key_s: pd.Series, y: np.ndarray, prior: float, smooth):
    return _te_map_from_agg(_te_agg(key_s, y), prior, smooth)


# 平滑化の強さを列名用の文字列にする
def _smooth_tag(smooth) -> str:
    if isinstance(smooth, str):
        return smooth
    f = float(smooth)
    return str(int(f)) if f == int(f) else str(f)


# fold 内で Out-of-Fold の Target Encoding を作る
def target_encode_fold(
    keys_train_fold: pd.DataFrame,
    y_train_fold: np.ndarray,
    other_frames: list[pd.DataFrame],
    keys,
    smooth=20.0,
    n_inner: int = 5,
    seed: int = 42,
    suffix: bool | None = None,
):
    """Leak-free target encoding.

    Parameters
    ----------
    keys_train_fold : key frame of the *training* rows of the current outer fold
    y_train_fold    : labels of those rows
    other_frames    : key frames to transform with the full-fold statistics
                      (typically [valid_keys, test_keys])
    keys            : list of column names (str) or tuples (interaction keys)

    Returns
    -------
    (te_train, [te_other, ...]) as DataFrames aligned to the inputs.
    """
    smooths = smooth if isinstance(smooth, (list, tuple)) else [smooth]
    # 列名に平滑化の接尾辞(_sauto など)を付けるか。既定は平滑化が複数のときだけ
    multi = len(smooths) > 1 if suffix is None else suffix
    prior = float(y_train_fold.mean())
    te_tr = pd.DataFrame(index=keys_train_fold.index)
    te_others = [pd.DataFrame(index=f.index) for f in other_frames]

    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    inner_splits = list(inner.split(np.zeros(len(y_train_fold)), y_train_fold))

    for key in keys:
        name = key if isinstance(key, str) else "X".join(key)
        s_fit = _key_series(keys_train_fold, key)
        arr = s_fit.to_numpy()
        cols = {
            sm: (f"te_{name}_s{_smooth_tag(sm)}" if multi else f"te_{name}")
            for sm in smooths
        }

        # --- training rows: inner out-of-fold encoding -------------------
        # the (sum, count) aggregation is the expensive part, so it is done
        # once per inner fold and shared by every smoothing level.
        vals = {sm: np.full(len(arr), prior, dtype="float32") for sm in smooths}
        for in_idx, out_idx in inner_splits:
            agg = _te_agg(pd.Series(arr[in_idx]), y_train_fold[in_idx])
            s_out = pd.Series(arr[out_idx])
            for sm in smooths:
                m = _te_map_from_agg(agg, prior, sm)
                vals[sm][out_idx] = (
                    s_out.map(m).fillna(prior).to_numpy(dtype="float32")
                )
        for sm in smooths:
            te_tr[cols[sm]] = vals[sm]

        # --- valid / test rows: statistics of the whole training fold ----
        agg_full = _te_agg(s_fit, y_train_fold)
        others_s = [_key_series(frame, key) for frame in other_frames]
        for sm in smooths:
            m_full = _te_map_from_agg(agg_full, prior, sm)
            for s, out in zip(others_s, te_others):
                out[cols[sm]] = s.map(m_full).fillna(prior).to_numpy(dtype="float32")

    return te_tr, te_others


# --------------------------------------------------------------------------
# key set builders
# --------------------------------------------------------------------------
# Target Encoding・Count に使う単独列のキーの一覧を返す
def single_keys(kind: str = "all"):
    """単独列の TE / Count キー一覧. kind='all'/'cat'/'num' で絞る."""
    if kind == "all":
        return NUMERIC_COLS + CATEGORICAL_COLS
    if kind == "cat":
        return list(CATEGORICAL_COLS)
    if kind == "num":
        return list(NUMERIC_COLS)
    raise ValueError(kind)


# --------------------------------------------------------------------------
# 本番の構成(作る列の設計)
# --------------------------------------------------------------------------
# 値の種類（ユニーク値）が多い数値列。digit・Count・平滑化3種の Target Encoding はこの2列だけに作る
HIGH_CARD_NUMERIC = ["Annual_Income_USD", "Daily_Commute_km"]
TRIPLE_SMOOTHS = ["auto", 10.0, 100.0]
# EDA で見つかった唯一の交互作用: 自宅充電の可否 × 自宅近くの充電スタンド数
HOME_PAIR = ("Home_Charging_Possible", "Charging_Stations_Near_Home")


# 本番で作る Target Encoding の (キー, 平滑化) の一覧を返す。この順番がそのまま列の並び順になる
def te_plan():
    """数値列の順(値の種類が多い2列は平滑化3種、少ない5列は auto だけ)→ Smooth Keys 4本(3種)→ 組み合わせ1組(auto)。

    カテゴリ列には作らない(LightGBM はカテゴリのまま好きな組に分けられる)。
    列の並び順は学習結果に影響する(列サンプリングが列の位置で決まる)ので変えないこと。
    """
    plan = [(c, TRIPLE_SMOOTHS if c in HIGH_CARD_NUMERIC else ["auto"]) for c in NUMERIC_COLS]
    plan += [(k, TRIPLE_SMOOTHS) for k in SMOOTH_KEYS]
    plan.append((HOME_PAIR, ["auto"]))
    return plan


# キーごとに平滑化を変えて、fold 内で Out-of-Fold の Target Encoding を作る(plan の順に列を並べる)
def target_encode_plan(keys_train_fold, y_train_fold, other_frames, plan, n_inner: int = 5, seed: int = 42):
    """te_plan() のとおりに Target Encoding を作る。値は target_encode_fold と同じ(キーごとに独立に計算するため)。"""
    parts_fit, parts_other = [], [[] for _ in other_frames]
    for key, smooths in plan:
        a, bs = target_encode_fold(keys_train_fold, y_train_fold, other_frames, [key], smooth=list(smooths),
                                   n_inner=n_inner, seed=seed, suffix=True)
        parts_fit.append(a)
        for lst, b in zip(parts_other, bs):
            lst.append(b)
    return pd.concat(parts_fit, axis=1), [pd.concat(lst, axis=1) for lst in parts_other]
