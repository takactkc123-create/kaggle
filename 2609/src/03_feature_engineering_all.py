"""S6E9 統合 FE カタログ (FE Lead 管轄).

4本の `fe_<model>.py` に散らばっていた FE 関数を、**モデル非依存の統一インターフェース**
として1箇所にまとめたもの。既存の `03_feature_engineering_lgbm.py` / `03_feature_engineering_xgb.py` / `03_feature_engineering_catboost.py` /
`03_feature_engineering_realmlp.py` は一切変更していない。このファイルは独立した集約版カタログであり、
横展開テスト (`tools/crosstest_gbdt.py`) の入力になる。

各関数の docstring に **実装差分** (どのモデル版と何が違うか) を明記している。
差分サマリは `fe_results_all.md` を参照。

規約:
- 関数は「特徴量を作る」ことだけを行う。cat_features 指定 / category dtype 化 /
  標準化などモデル固有の処理は呼び出し側の責務。
- 目的変数を使う変換 (TE) は必ず fold 内 fit。学習行には inner-OOF を当てる。
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

TARGET = "Will_Buy_EV"
ID = "id"

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

# 低カーディナリティ数値列 (catify / 交互作用キーの候補)
# fe_lgbm.LOW_CARD_NUMERIC と fe_catboost.LOWCARD_NUM_COLS は同一内容 (順序のみ違う)
LOWCARD_NUM_COLS = [
    "Age",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

HIGHCARD_NUM_COLS = ["Annual_Income_USD", "Daily_Commute_km"]

ALL_COLS = NUMERIC_COLS + CATEGORICAL_COLS


# ==========================================================================
# 0. データ読み込み (baseline_*.py と同一フロー)
# ==========================================================================
def load_data(data_dir: str = "data"):
    """data/train.csv と data/test.csv を読む (02_baseline_*.py と同じフロー)."""
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    return train, test


def get_y(train: pd.DataFrame) -> np.ndarray:
    """目的変数を 0/1 の ndarray にする."""
    return (train[TARGET] == "Yes").astype(int).to_numpy()


# ==========================================================================
# 1. エンコーディング方式 (モデル固有の入力形式)
#    出典: fe_xgb.as_native_category / as_ordinal / as_onehot
#    実測: 3方式とも単体では同点 (±0.0001)。多様性用に散らす軸として使う。
# ==========================================================================
def as_native_category(tr, te, cols=None):
    """train/test 共通のカテゴリ集合で category dtype 化 (LightGBM / XGB enable_categorical)."""
    cols = CATEGORICAL_COLS if cols is None else cols
    tr, te = tr.copy(), te.copy()
    for c in cols:
        cats = pd.concat([tr[c], te[c]]).astype("category").cat.categories
        tr[c] = pd.Categorical(tr[c], categories=cats)
        te[c] = pd.Categorical(te[c], categories=cats)
    return tr, te


def as_ordinal(tr, te, cols=None):
    """整数コード化 (XGBoost の最終採用方式)."""
    cols = CATEGORICAL_COLS if cols is None else cols
    tr, te = tr.copy(), te.copy()
    for c in cols:
        cats = pd.concat([tr[c], te[c]]).astype("category").cat.categories
        tr[c] = pd.Categorical(tr[c], categories=cats).codes.astype("int16")
        te[c] = pd.Categorical(te[c], categories=cats).codes.astype("int16")
    return tr, te


def as_str(frames, cols) -> None:
    """文字列化 (CatBoost cat_features / catify 用). in-place."""
    for f in frames:
        for c in cols:
            f[c] = f[c].astype(str)


# ==========================================================================
# 2. キーフレーム (厳密値 TE / Count のグループキー)
#    出典: fe_lgbm.make_key_frame
#    差分: fe_xgb / fe_catboost は生データフレームを直接キーに使い astype(str) で
#          キー化している。数値的には等価だが str 化は遅い。ここでは int キー方式
#          (fe_lgbm 版) を採る。さらに小数1桁を ×10 して整数化し float 等価判定の
#          揺れを完全に排除している (fe_lgbm 版は float のまま groupby)。
# ==========================================================================
def make_key_frame(train: pd.DataFrame, test: pd.DataFrame):
    """13列すべてを「厳密値のまま」整数キー化したフレームを返す (S6E8 のブレークスルー)."""
    keys_tr = pd.DataFrame(index=train.index)
    keys_te = pd.DataFrame(index=test.index)
    for c in NUMERIC_COLS:
        keys_tr[c] = np.rint(train[c].to_numpy(dtype="float64") * 10).astype("int64")
        keys_te[c] = np.rint(test[c].to_numpy(dtype="float64") * 10).astype("int64")
    for c in CATEGORICAL_COLS:
        cats = pd.concat([train[c], test[c]]).astype("category").cat.categories
        keys_tr[c] = pd.Categorical(train[c], categories=cats).codes.astype("int16")
        keys_te[c] = pd.Categorical(test[c], categories=cats).codes.astype("int16")
    return keys_tr, keys_te


# ==========================================================================
# 3. Smooth Keys (高カーデ数値列の粗い解像度キー)  reference_URL.md S-3
#    ★ 4モデルで実装が食い違っている箇所 ★
#      fe_lgbm     : inc/10,  inc/100,  inc/1000,  floor(km)
#      fe_xgb      : inc/100, inc/1000, inc/10000, floor(km)
#      fe_catboost : floor(inc) <- 厳密値キーと**ビット同一の重複**, inc/100, inc/1000, floor(km)
#      fe_realmlp  : inc/100, inc/1000, inc/10000, floor(km/5)  ※TEキーでなくcat特徴として
#    Annual_Income_USD は整数値なので floor(inc) == inc であり CatBoost の sk_inc_1 は
#    無駄列。ここでは和集合 {10, 100, 1000, 10000} を既定にし scales で選択可能にする。
# ==========================================================================
SMOOTH_KEY_SCALES = (10, 100, 1000, 10000)


def add_smooth_keys(keys: pd.DataFrame, df: pd.DataFrame, scales=SMOOTH_KEY_SCALES,
                    commute: bool = True) -> pd.DataFrame:
    """年収・通勤距離を粗く丸めたキーを keys に追加して返す."""
    keys = keys.copy()
    inc = df["Annual_Income_USD"].to_numpy(dtype="float64")
    for s in scales:
        keys[f"sk_inc{s}"] = np.floor(inc / s).astype("int64")
    if commute:
        km = df["Daily_Commute_km"].to_numpy(dtype="float64")
        keys["sk_commute"] = np.floor(km).astype("int64")
    return keys


def smooth_key_names(scales=SMOOTH_KEY_SCALES, commute: bool = True):
    """add_smooth_keys が作るキー名の一覧を返す."""
    out = [f"sk_inc{s}" for s in scales]
    return out + (["sk_commute"] if commute else [])


# ==========================================================================
# 4. digit features  ((x // 10**k) % 10)  reference_URL.md S-1
#    出典: fe_catboost.add_digits (整数演算版) を採用。
#    差分: fe_lgbm / fe_xgb は float 演算 (x * 10**-k を floor) で、23.4 のような値で
#          丸め誤差が下位桁に漏れうる。整数版の方が安全。
#    ★重要★ 単独では効かず Triple TE との併用で初めて効く。
#            さらに「1値あたりの行数」が前提なのでサブサンプル検証は原理的に無効。
# ==========================================================================
DIGIT_KS = list(range(-4, 4))


def add_digit_features(df: pd.DataFrame, cols=None, ks=None) -> pd.DataFrame:
    """数値列を桁ごとにばらした int8 列を作る ((x // 10**k) % 10)."""
    cols = NUMERIC_COLS if cols is None else cols
    ks = DIGIT_KS if ks is None else ks
    out = {}
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy()
        scaled = np.rint(x * 10_000).astype(np.int64)  # 小数4桁分の余裕
        for k in ks:
            out[f"{c}_d{k}"] = ((scaled // (10 ** (k + 4))) % 10).astype("int8")
    return pd.DataFrame(out, index=df.index)


def drop_constant_cols(frames, cols):
    """全フレームで定数の列を落とす (学習を遅くするだけなので)。残った列名を返す."""
    kept = []
    for c in cols:
        if any(f[c].nunique(dropna=False) > 1 for f in frames):
            kept.append(c)
        else:
            for f in frames:
                f.drop(columns=[c], inplace=True)
    return kept


# ==========================================================================
# 5. Count / Frequency Encoding (教師なし -> train+test 結合で fit、リークなし)
#    出典: fe_lgbm.count_encode
#    差分: fe_catboost は freq=False (生カウント)、fe_lgbm/fe_xgb は freq=True。
#          木にとっては単調変換で等価だが、**NN ではスケールが効くので freq 推奨**。
#          fe_realmlp は Annual_Income_USD 1列のみ、しかも train だけで fit している
#          (= test の頻度情報を捨てている)。
#    実測: LGBM +0.00083 / XGB +0.00049 / CatBoost -0.00017(無効) / RealMLP 未検証。
# ==========================================================================
def count_encode(keys_tr: pd.DataFrame, keys_te: pd.DataFrame, cols, freq: bool = True):
    """出現頻度を列にする. 目的変数を使わないので train+test でまとめて数える."""
    out_tr = pd.DataFrame(index=keys_tr.index)
    out_te = pd.DataFrame(index=keys_te.index)
    n_total = len(keys_tr) + len(keys_te)
    for c in cols:
        vc = pd.concat([keys_tr[c], keys_te[c]], ignore_index=True).value_counts()
        a = keys_tr[c].map(vc).astype("float32").to_numpy()
        b = keys_te[c].map(vc).astype("float32").to_numpy()
        if freq:
            a, b = a / n_total, b / n_total
        out_tr[f"cnt_{c}"] = a
        out_te[f"cnt_{c}"] = b
    return out_tr, out_te


# ==========================================================================
# 6. Target Encoding (fold 内 fit + 学習行は inner-OOF = 入れ子TE)
#    出典: fe_lgbm.target_encode_fold (XGB で +0.00108 を出した実装形)
#    差分: 「単純 fold 内 fit」版 (学習行にも fold 全体の統計を当てる) は学習行に
#          楽観バイアスが乗り、入れ子版より **-0.00108** 劣る。
#          fe_catboost.target_encode / fe_xgb.fit_apply_te_cv_nested_multi も同じ
#          入れ子方式。ここでは (sum,count) 集計を smooth 間で共有する fe_lgbm 版を
#          採用 (Triple TE の追加コストがほぼゼロになる)。
#    Triple TE = smooth を auto/10/100 の3系統「同時投入」 (選ぶのではない)。
# ==========================================================================
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


def target_encode_fold(keys_fit: pd.DataFrame, y_fit, other_frames, cols,
                       smooths=(20.0,), n_inner: int = 5, seed: int = 42):
    """リークフリー TE。戻り値 (te_fit, [te_other, ...])。

    keys_fit     : 現在の outer fold の**学習行**のキーフレーム
    other_frames : 同じ統計を当てるフレーム (通常 [valid_keys, test_keys])
    smooths      : float または "auto" のリスト。複数指定で Triple TE。
    """
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
        names = {sm: (f"te_{c}_s{_smooth_tag(sm)}" if multi else f"te_{c}")
                 for sm in smooths}

        vals = {sm: np.full(len(arr), prior, dtype="float32") for sm in smooths}
        for in_idx, out_idx in inner_splits:
            agg = _te_agg(arr[in_idx], y_fit[in_idx])
            s_out = pd.Series(arr[out_idx])
            for sm in smooths:
                vals[sm][out_idx] = (
                    s_out.map(_te_map(agg, prior, sm)).fillna(prior)
                    .to_numpy(dtype="float32")
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


# ==========================================================================
# 7. catify (低カーデ数値列をカテゴリとして扱う)
#    出典: fe_catboost.cast_to_str (+0.00170)
#    LightGBM では category dtype、XGBoost では enable_categorical、
#    CatBoost では cat_features として渡す。RealMLP は build_features 内の
#    `{col}_cat_` (floor 値の factorize) が実質これに相当し**適用済み**。
# ==========================================================================
def catify(tr: pd.DataFrame, te: pd.DataFrame, cols=None, mode: str = "category"):
    """低カーデ数値列をカテゴリ扱いに変換した (tr, te) を返す."""
    cols = LOWCARD_NUM_COLS if cols is None else cols
    tr, te = tr.copy(), te.copy()
    for c in cols:
        if mode == "str":
            tr[c] = tr[c].astype(str)
            te[c] = te[c].astype(str)
        else:
            # XGBoost は float dtype のカテゴリを受け付けないので、必ず整数コードに
            # 変換してから category dtype にする (LightGBM もこれで問題ない)。
            cats = pd.Index(sorted(set(tr[c].unique()) | set(te[c].unique())))
            codes_tr = pd.Categorical(tr[c], categories=cats).codes.astype("int16")
            codes_te = pd.Categorical(te[c], categories=cats).codes.astype("int16")
            allc = pd.Index(range(len(cats)))
            tr[c] = pd.Categorical(codes_tr, categories=allc)
            te[c] = pd.Categorical(codes_te, categories=allc)
    return tr, te


# ==========================================================================
# 8. 打ち止め済み (再検証不要) — 参照用に残すが既定では使わない
#    - 四則演算 diff/ratio/sum/avg : 3モデルすべてで無効〜悪化 (CatBoost -0.00099)
#    - 交互作用 TE (2/3/6/10/13列) : 全滅
#    - 行フィンガープリント        : train 全行ユニークで原理的に機能しない
#    - 元データ concat / buy_score : 実測 -0.00002
# ==========================================================================
def arithmetic_meaningful(df: pd.DataFrame) -> pd.DataFrame:
    """【打ち止め】意味ベースの四則演算。再検証不要 (LGBM -0.00014 / CatBoost -0.00099)."""
    out = pd.DataFrame(index=df.index)
    home, work = df["Charging_Stations_Near_Home"], df["Charging_Stations_Near_Work"]
    inc, km, age = df["Annual_Income_USD"], df["Daily_Commute_km"], df["Age"]
    cars, env = df["Number_of_Cars_Owned"], df["Environmental_Concern_Level"]
    out["ar_charge_sum"] = home + work
    out["ar_charge_diff"] = home - work
    out["ar_income_per_km"] = inc / (km + 1.0)
    out["ar_income_per_age"] = inc / age
    out["ar_km_per_charge"] = km / (home + work + 1.0)
    out["ar_age_x_env"] = age * env
    out["ar_charge_per_car"] = (home + work) / (cars + 1.0)
    return out


def interaction_keys(tr, te, pairs):
    """【打ち止め】2列連結キー。TE/Count いずれも全滅。"""
    ktr, kte = pd.DataFrame(index=tr.index), pd.DataFrame(index=te.index)
    for a, b in pairs:
        n = f"{a}__x__{b}"
        ktr[n] = tr[a].astype(str) + "|" + tr[b].astype(str)
        kte[n] = te[a].astype(str) + "|" + te[b].astype(str)
    return ktr, kte


def cat_pairs(cols=None):
    """【打ち止め】カテゴリ列の2列ペアを列挙する (交互作用TE用)."""
    cols = CATEGORICAL_COLS if cols is None else cols
    return [tuple(p) for p in itertools.combinations(cols, 2)]


# ==========================================================================
# 6. 年収の近傍統計(近くの値の購入率・傾き・曲率)
#    出典: jazivxt/single-model-zoom-zoom の局所ビン統計、blamerx の window encodings
#    厳密値TEは「年収がちょうどこの値」の購入率(1値あたり約50行)。
#    ここでは「この値を中心に ±r ドル」の購入率と、左右の購入率の差(カーブの向き)を渡す。
# ==========================================================================
NEIGHBOR_RADII = (10, 50, 250, 1000)   # 窓に入る行数は中央値で約 400 / 1,300 / 4,600 / 16,000 行


def _window_stats(fit_values, fit_y, query_values, radius):
    """query の各値について、fit 側の3つの窓の (購入者数, 件数) を返す。

    fit を値の順に並べて累積和を作り、窓の端を searchsorted で探す。
    窓の中の合計は「右端までの累積 - 左端の手前までの累積」。行ごとのループがないので速い。

      center: [x - r, x + r]   left: [x - r, x)   right: (x, x + r]
    """
    order = np.argsort(fit_values, kind="stable")
    sorted_values = np.asarray(fit_values, dtype=np.float64)[order]
    cum_pos = np.concatenate([[0.0], np.cumsum(np.asarray(fit_y, dtype=np.float64)[order])])
    q = np.asarray(query_values, dtype=np.float64)

    lo = np.searchsorted(sorted_values, q - radius, side="left")
    mid_lo = np.searchsorted(sorted_values, q, side="left")
    mid_hi = np.searchsorted(sorted_values, q, side="right")
    hi = np.searchsorted(sorted_values, q + radius, side="right")

    def count(a, b):
        return cum_pos[b] - cum_pos[a], (b - a).astype(np.float64)

    return {"center": count(lo, hi), "left": count(lo, mid_lo), "right": count(mid_hi, hi)}


def _neighborhood_frame(fit_values, fit_y, query_values, radii, smooth, prior, with_slope):
    """1組の (fit, query) について、半径ごとの購入率・傾き・曲率の列を作る。"""
    columns = {}
    for r in radii:
        stats = _window_stats(fit_values, fit_y, query_values, r)
        rate = {k: (pos + smooth * prior) / (cnt + smooth) for k, (pos, cnt) in stats.items()}
        columns[f"inc_nbr_r{r}"] = rate["center"]
        if with_slope:
            columns[f"inc_slope_r{r}"] = rate["right"] - rate["left"]
            columns[f"inc_curve_r{r}"] = rate["center"] - 0.5 * (rate["left"] + rate["right"])
    return pd.DataFrame(columns).astype("float32")


def add_income_neighborhood(fit_income, fit_y, other_incomes, radii=NEIGHBOR_RADII,
                            smooth=10.0, with_slope=True, n_inner=5, seed=42):
    """年収の近傍統計を、target_encode_fold と同じ作法でリークなく作る。

    - fit 側の行: 内側 n_inner-fold の out-of-fold 値(自分の正解を含まない)
    - other(valid / test): fit 全体の統計
    購入率は smooth 件ぶん全体平均に寄せる: (購入者 + smooth * 平均) / (件数 + smooth)

    返り値: (fit 用の DataFrame, [other ごとの DataFrame])
    列: 半径ごとに inc_nbr_r{r}、with_slope なら inc_slope_r{r} / inc_curve_r{r} も
    """
    fit_income = np.asarray(fit_income, dtype=np.float64)
    fit_y = np.asarray(fit_y)
    prior = float(fit_y.mean())

    fit_frame = None
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for inner_fit, inner_valid in inner.split(fit_income, fit_y):
        block = _neighborhood_frame(fit_income[inner_fit], fit_y[inner_fit], fit_income[inner_valid],
                                    radii, smooth, float(fit_y[inner_fit].mean()), with_slope)
        if fit_frame is None:
            fit_frame = pd.DataFrame(np.nan, index=range(len(fit_income)),
                                     columns=block.columns, dtype="float32")
        fit_frame.iloc[inner_valid] = block.to_numpy()

    others = [_neighborhood_frame(fit_income, fit_y, np.asarray(o, dtype=np.float64),
                                  radii, smooth, prior, with_slope)
              for o in other_incomes]
    return fit_frame, others
