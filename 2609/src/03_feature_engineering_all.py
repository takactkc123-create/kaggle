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

# 値の種類（ユニーク値）が少ない数値列 (catify / 交互作用キーの候補)
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
# train.csv と test.csv を読み込む
def load_data(data_dir: str = "data"):
    """data/train.csv と data/test.csv を読む (02_baseline_*.py と同じフロー)."""
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    return train, test


# 目的変数を 0/1 の配列にする
def get_y(train: pd.DataFrame) -> np.ndarray:
    """目的変数を 0/1 の ndarray にする."""
    return (train[TARGET] == "Yes").astype(int).to_numpy()


# ==========================================================================
# 1. エンコーディング方式 (モデル固有の入力形式)
#    出典: fe_xgb.as_native_category / as_ordinal / as_onehot
#    実測: 3方式とも単体では同点 (±0.0001)。多様性用に散らす軸として使う。
# ==========================================================================
# カテゴリ列を train・test 共通の水準で category 型にする
def as_native_category(tr, te, cols=None):
    """train/test 共通のカテゴリ集合で category dtype 化 (LightGBM / XGB enable_categorical)."""
    cols = CATEGORICAL_COLS if cols is None else cols
    tr, te = tr.copy(), te.copy()
    for c in cols:
        cats = pd.concat([tr[c], te[c]]).astype("category").cat.categories
        tr[c] = pd.Categorical(tr[c], categories=cats)
        te[c] = pd.Categorical(te[c], categories=cats)
    return tr, te


# カテゴリ列を整数コードにする
def as_ordinal(tr, te, cols=None):
    """整数コード化 (XGBoost の最終採用方式)."""
    cols = CATEGORICAL_COLS if cols is None else cols
    tr, te = tr.copy(), te.copy()
    for c in cols:
        cats = pd.concat([tr[c], te[c]]).astype("category").cat.categories
        tr[c] = pd.Categorical(tr[c], categories=cats).codes.astype("int16")
        te[c] = pd.Categorical(te[c], categories=cats).codes.astype("int16")
    return tr, te


# 列を文字列にする(CatBoost の cat_features 用)
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
# 13 列すべてを値のまま整数キーにしたフレームを作る
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
# 3. Smooth Keys (値の種類（ユニーク値）が多い数値列の粗い解像度キー)  reference_URL.md S-3
#    ★ 4モデルで実装が食い違っている箇所 ★
#      fe_lgbm     : inc/10,  inc/100,  inc/1000,  floor(km)
#      fe_xgb      : inc/100, inc/1000, inc/10000, floor(km)
#      fe_catboost : floor(inc) <- 厳密値キーと**ビット同一の重複**, inc/100, inc/1000, floor(km)
#      fe_realmlp  : inc/100, inc/1000, inc/10000, floor(km/5)  ※TEキーでなくcat特徴として
#    Annual_Income_USD は整数値なので floor(inc) == inc であり CatBoost の sk_inc_1 は
#    無駄列。ここでは和集合 {10, 100, 1000, 10000} を既定にし scales で選択可能にする。
# ==========================================================================
SMOOTH_KEY_SCALES = (10, 100, 1000, 10000)


# 年収・通勤距離を粗く丸めたキー(Smooth Keys)を追加する
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


# add_smooth_keys が作るキー名の一覧を返す
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


# 数値列を桁ごとの列(digit features)にばらす
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


# すべてのフレームで値が一定の列を落とし、残った列名を返す
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
# キーの値ごとの出現回数を列にする(train と test をまとめて数える)
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
# 6. Target Encoding (fold 内 fit + 学習行は inner-OOF = Out-of-Fold TE)
#    出典: fe_lgbm.target_encode_fold (XGB で +0.00108 を出した実装形)
#    差分: 「単純 fold 内 fit」版 (学習行にも fold 全体の統計を当てる) は学習行に
#          楽観バイアスが乗り、Out-of-Fold 版より **-0.00108** 劣る。
#          fe_catboost.target_encode / fe_xgb.fit_apply_te_cv_nested_multi も同じ
#          Out-of-Fold 方式。ここでは (sum,count) 集計を smooth 間で共有する fe_lgbm 版を
#          採用 (Triple TE の追加コストがほぼゼロになる)。
#    Triple TE = smooth を auto/10/100 の3系統「同時投入」 (選ぶのではない)。
# ==========================================================================
# キーごとの購入者数と行数を集計する
def _te_agg(arr, y):
    return pd.DataFrame({"k": arr, "y": y}).groupby("k", observed=True)["y"].agg(
        ["sum", "count"]
    )


# 集計から平滑化した購入率の対応表を作る
def _te_map(agg, prior, smooth):
    cnt = agg["count"].to_numpy(dtype="float64")
    s = agg["sum"].to_numpy(dtype="float64")
    if isinstance(smooth, str):  # "auto" = sklearn TargetEncoder の経験ベイズ則
        p_i = s / cnt
        m = (p_i * (1.0 - p_i)) / (prior * (1.0 - prior))
    else:
        m = float(smooth)
    return pd.Series((s + prior * m) / (cnt + m), index=agg.index)


# 平滑化の強さを列名用の文字列にする
def _smooth_tag(sm):
    if isinstance(sm, str):
        return sm
    f = float(sm)
    return str(int(f)) if f == int(f) else str(f).replace(".", "p")


# fold 内で Out-of-Fold の Target Encoding を作る
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
# 7. catify (値の種類（ユニーク値）が少ない数値列をカテゴリとして扱う)
#    出典: fe_catboost.cast_to_str (+0.00170)
#    LightGBM では category dtype、XGBoost では enable_categorical、
#    CatBoost では cat_features として渡す。RealMLP は build_features 内の
#    `{col}_cat_` (floor 値の factorize) が実質これに相当し**適用済み**。
# ==========================================================================
# 値の種類が少ない数値列をカテゴリとして扱える形にする
def catify(tr: pd.DataFrame, te: pd.DataFrame, cols=None, mode: str = "category"):
    """値の種類（ユニーク値）が少ない数値列をカテゴリ扱いに変換した (tr, te) を返す."""
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
# 【不採用】意味で選んだ数値列の四則演算の列を作る
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


# 【不採用】2 列を連結した交互作用のキーを作る
def interaction_keys(tr, te, pairs):
    """【打ち止め】2列連結キー。TE/Count いずれも全滅。"""
    ktr, kte = pd.DataFrame(index=tr.index), pd.DataFrame(index=te.index)
    for a, b in pairs:
        n = f"{a}__x__{b}"
        ktr[n] = tr[a].astype(str) + "|" + tr[b].astype(str)
        kte[n] = te[a].astype(str) + "|" + te[b].astype(str)
    return ktr, kte


# 【不採用】カテゴリ列の 2 列の組をすべて列挙する
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


# 年収の各値について、前後の窓に入る購入者数と行数を数える
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

    # 累積和から、区間の購入者数と行数を取り出す
    def count(a, b):
        return cum_pos[b] - cum_pos[a], (b - a).astype(np.float64)

    return {"center": count(lo, hi), "left": count(lo, mid_lo), "right": count(mid_hi, hi)}


# 半径ごとの近傍の購入率・傾き・曲率の列を作る
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


# 【不採用】年収の近傍統計をリークなく作る
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


# ============================================================================
# 不採用(記録)— 各モデルの 03_feature_engineering_<model>.py から移した関数
#   本番では使わない。再検証しないための記録として残す。名前の末尾はもとのモデル。
#   根拠は src/feature_catalog.py の FUNC_STATUS と、notebooks/03_feature_engineering.ipynb の 9 章。
#   移す前のコードは git のタグ best-20260927-d にある。
# ============================================================================


# 移した関数が参照する定数
LOW_CARD_NUMERIC_LGBM = [
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
    "Age",
]
_EPS_XGB = 1e-6
INTERACTION_PAIRS_CATBOOST = [
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


# ---- LightGBM(もとは 03_feature_engineering_lgbm.py) --------------------

# 【不採用】意味で選んだ比・差・和の列を作る
def add_arithmetic_meaningful_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-meaningful ratio / diff / sum features."""
    out = pd.DataFrame(index=df.index)
    home = df["Charging_Stations_Near_Home"]
    work = df["Charging_Stations_Near_Work"]
    inc = df["Annual_Income_USD"]
    km = df["Daily_Commute_km"]
    age = df["Age"]
    cars = df["Number_of_Cars_Owned"]
    env = df["Environmental_Concern_Level"]

    out["ar_charge_sum"] = home + work
    out["ar_charge_diff"] = home - work
    out["ar_charge_ratio"] = home / (work + 1.0)
    out["ar_income_per_km"] = inc / (km + 1.0)
    out["ar_income_per_age"] = inc / age
    out["ar_income_per_car"] = inc / (cars + 1.0)
    out["ar_km_per_charge"] = km / (home + work + 1.0)
    out["ar_income_x_env"] = inc * env
    out["ar_env_per_km"] = env / (km + 1.0)
    out["ar_charge_per_car"] = (home + work) / (cars + 1.0)
    out["ar_age_x_env"] = age * env
    out["ar_km_per_car"] = km / (cars + 1.0)
    return out


# 【不採用】数値列の全ペアの差・比・和の列を作る
def add_arithmetic_all_pairs_lgbm(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """diff / ratio / sum over every numeric pair.

    `avg` for a pair is a strictly monotonic transform of `sum` (sum / 2), so
    tree models cannot distinguish them -> only `sum` is materialised here and
    `avg` is covered by the multi-column group means below.
    """
    cols = NUMERIC_COLS if cols is None else cols
    out = pd.DataFrame(index=df.index)
    for a, b in itertools.combinations(cols, 2):
        va = df[a].astype("float32")
        vb = df[b].astype("float32")
        out[f"p_{a}_m_{b}"] = va - vb
        out[f"p_{a}_p_{b}"] = va + vb
        out[f"p_{a}_d_{b}"] = va / (vb + 1.0)
    return out


# 【不採用】標準化した数値列の行ごとの平均・標準偏差を作る
def add_group_means_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    """avg-style aggregates over standardised numeric columns."""
    out = pd.DataFrame(index=df.index)
    z = (df[NUMERIC_COLS] - df[NUMERIC_COLS].mean()) / df[NUMERIC_COLS].std()
    out["g_avg_all"] = z.mean(axis=1).astype("float32")
    out["g_std_all"] = z.std(axis=1).astype("float32")
    out["g_avg_charge"] = df[
        ["Charging_Stations_Near_Home", "Charging_Stations_Near_Work"]
    ].mean(axis=1).astype("float32")
    return out


# 【不採用】補助金(0/1)と 3 列との積を作る
def add_subsidy_products_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    """補助金(0/1)と、補助金がないと効きが横ばいになる3列との積。

    EDA で、補助金なしの群では環境意識・収入・航続距離への不安のどれを動かしても
    購入率がほぼ床に張り付いていた。補助金ありのときだけ値が残る形で渡す。
    """
    subsidy = (df["Subsidy_Available"] == "Yes").astype("float32")
    anxiety = df["Range_Anxiety_Level"].map({"Low": 0, "Medium": 1, "High": 2}).astype("float32")
    return pd.DataFrame({
        "subsidy_x_env": subsidy * df["Environmental_Concern_Level"].astype("float32"),
        "subsidy_x_income": subsidy * df["Annual_Income_USD"].astype("float32"),
        "subsidy_x_anxiety": subsidy * anxiety,
    }, index=df.index)


# 【不採用】2 列の交互作用キーの一覧を返す
def pair_keys_lgbm(kind: str = "cat"):
    """2-way interaction keys."""
    if kind == "cat":
        return [tuple(p) for p in itertools.combinations(CATEGORICAL_COLS, 2)]
    if kind == "cat_lownum":
        return [
            (a, b) for a in CATEGORICAL_COLS for b in LOW_CARD_NUMERIC_LGBM
        ]
    if kind == "lownum":
        return [tuple(p) for p in itertools.combinations(LOW_CARD_NUMERIC_LGBM, 2)]
    if kind == "all":
        cols = CATEGORICAL_COLS + LOW_CARD_NUMERIC_LGBM
        return [tuple(p) for p in itertools.combinations(cols, 2)]
    raise ValueError(kind)


# 【不採用】3 列の交互作用キーの一覧を返す
def triple_keys_lgbm(kind: str = "cat"):
    """【打ち止め】3列を連結した交互作用キー."""
    if kind == "cat":
        return [tuple(p) for p in itertools.combinations(CATEGORICAL_COLS, 3)]
    if kind == "selected":
        return [
            ("Home_Charging_Possible", "Range_Anxiety_Level", "Subsidy_Available"),
            ("City_Type", "Current_Car_Type", "Range_Anxiety_Level"),
            ("Home_Charging_Possible", "Charging_Stations_Near_Home", "Range_Anxiety_Level"),
            ("Environmental_Concern_Level", "Range_Anxiety_Level", "Home_Charging_Possible"),
            ("City_Type", "Home_Charging_Possible", "Environmental_Concern_Level"),
        ]
    raise ValueError(kind)


# 【不採用】全列を連結したキー(行フィンガープリント)を作る
def all_columns_key_lgbm():
    """One key made of every raw column (row fingerprint).

    WARNING (measured 2026-09-12): the full 13-column key is **unique for every
    single train row** (668,665 distinct keys / 668,665 rows, 100% singletons),
    so both TE and Count degenerate to a constant. Kept only for reference -
    use `fingerprint_key_lgbm()` subsets instead.
    """
    return [tuple(NUMERIC_COLS + CATEGORICAL_COLS)]


# fingerprint subsets that actually have repeated rows
FP_SETS_LGBM = {
    # 142,801 keys / median count 10 / only 8.8% singletons  <- the sweet spot
    "fp1": CATEGORICAL_COLS
    + [
        "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
        "Environmental_Concern_Level",
    ],
    # 313 keys / median count 13,652 -> full 6-way categorical interaction
    "fp2": list(CATEGORICAL_COLS),
    # 150,264 keys / median count 8
    "fp3": list(LOW_CARD_NUMERIC_LGBM),
    # cat6 + the two binary-ish charging signals only
    "fp4": CATEGORICAL_COLS
    + ["Charging_Stations_Near_Home", "Environmental_Concern_Level"],
}


# 【不採用】列の部分集合を連結したキーを作る
def fingerprint_key_lgbm(name: str):
    """Single multi-column key for the named fingerprint subset."""
    return [tuple(FP_SETS_LGBM[name])]


# ---- XGBoost(もとは 03_feature_engineering_xgb.py) --------------------

# 平滑化の強さを列名用の文字列にする
def _smooth_tag_xgb(sm):
    if sm == "auto":
        return "a"
    f = float(sm)
    return str(int(f)) if f == int(f) else str(f).replace(".", "p")


# 数値列はそのまま、カテゴリ列は category 型にした基本のフレームを作る
def make_base_xgb(train: pd.DataFrame, test: pd.DataFrame):
    """Baseline feature frames: numeric as-is + categorical as pandas Categorical.

    Mirrors 02_baseline_xgb.py (enable_categorical=True path).
    """
    tr = train[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    te = test[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    return as_native_category_xgb(tr, te, CATEGORICAL_COLS)


# カテゴリ列を train・test 共通の水準で category 型にする
def as_native_category_xgb(tr: pd.DataFrame, te: pd.DataFrame, cols):
    """Align category sets across train/test and cast to pandas Categorical."""
    tr = tr.copy()
    te = te.copy()
    for c in cols:
        cats = pd.concat([tr[c].astype(str), te[c].astype(str)]).astype("category").cat.categories
        tr[c] = pd.Categorical(tr[c].astype(str), categories=cats)
        te[c] = pd.Categorical(te[c].astype(str), categories=cats)
    return tr, te


# 【不採用】カテゴリ列を One-Hot Encoding する
def as_onehot_xgb(tr: pd.DataFrame, te: pd.DataFrame, cols):
    """One-hot encoding of the given columns (drop original)."""
    n_tr = len(tr)
    both = pd.concat([tr, te], axis=0, ignore_index=True)
    for c in cols:
        both[c] = both[c].astype(str)
    both = pd.get_dummies(both, columns=list(cols), dtype="int8")
    return both.iloc[:n_tr].reset_index(drop=True), both.iloc[n_tr:].reset_index(drop=True)


# Semantically meaningful pairs (first pass).
MEANINGFUL_PAIRS_XGB = [
    ("Charging_Stations_Near_Home", "Charging_Stations_Near_Work"),
    ("Annual_Income_USD", "Daily_Commute_km"),
    ("Age", "Annual_Income_USD"),
    ("Annual_Income_USD", "Number_of_Cars_Owned"),
    ("Daily_Commute_km", "Charging_Stations_Near_Work"),
    ("Environmental_Concern_Level", "Annual_Income_USD"),
    ("Age", "Number_of_Cars_Owned"),
]


# 2 列の和・差・比・平均の列をまとめて作る
def _arith_block_xgb(df: pd.DataFrame, pairs, ops=("diff", "ratio", "sum", "avg")):
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
            out[f"{a}_div_{b}"] = va / (vb + _EPS_XGB)
    return pd.DataFrame(out, index=df.index)


# 【不採用】数値列の四則演算の列を追加する
def add_arithmetic_xgb(tr, te, src_tr, src_te, pairs=None, ops=("diff", "ratio", "sum", "avg")):
    """Append arithmetic combinations of numeric columns.

    `src_*` are the raw frames holding the numeric columns.
    """
    pairs = MEANINGFUL_PAIRS_XGB if pairs is None else pairs
    return (
        pd.concat([tr, _arith_block_xgb(src_tr, pairs, ops)], axis=1),
        pd.concat([te, _arith_block_xgb(src_te, pairs, ops)], axis=1),
    )


# 【不採用】数値列の 2 列の組をすべて列挙する
def all_numeric_pairs_xgb():
    """【打ち止め】数値列の全2列ペアを列挙する (四則演算用)."""
    pairs = []
    for i, a in enumerate(NUMERIC_COLS):
        for b in NUMERIC_COLS[i + 1 :]:
            pairs.append((a, b))
    return pairs


# 【不採用】2 列を連結した交互作用のキーを作る
def make_interaction_keys_xgb(src_tr, src_te, pairs):
    """Build string keys for column pairs -> returned as extra raw frames."""
    ktr = pd.DataFrame(index=src_tr.index)
    kte = pd.DataFrame(index=src_te.index)
    for a, b in pairs:
        name = f"{a}__x__{b}"
        ktr[name] = src_tr[a].astype(str) + "|" + src_tr[b].astype(str)
        kte[name] = src_te[a].astype(str) + "|" + src_te[b].astype(str)
    return ktr, kte


# 【不採用】カテゴリ列の 2 列の組をすべて列挙する
def cat_pairs_xgb(cols=None):
    """【打ち止め】カテゴリ列の2列ペアを列挙する (交互作用TE用)."""
    cols = CATEGORICAL_COLS if cols is None else cols
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            out.append((a, b))
    return out


# 【不採用】学習行だけで Target Encoding の対応表を作る(Out-of-Fold なし)
def fit_target_encoding_xgb(src_fit: pd.DataFrame, y_fit, cols, smoothing=20.0, min_samples=1):
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


# 【不採用】Target Encoding の対応表を当てる
def apply_target_encoding_xgb(src: pd.DataFrame, maps, suffix="_te"):
    """Apply fitted TE maps. Unseen values fall back to the fold prior."""
    out = {}
    for c, (mapping, prior) in maps.items():
        out[f"{c}{suffix}"] = src[c].map(mapping).astype("float32").fillna(np.float32(prior))
    return pd.DataFrame(out, index=src.index)


# 【不採用】学習行で作った Target Encoding を学習・検証・test に当てる
def fit_apply_te_cv_xgb(src_tr, src_te, y, cols, train_idx, valid_idx, smoothing=20.0, min_samples=1):
    """Convenience: fit on train_idx rows, return (te_train, te_valid, te_test).

    Simple variant: the training rows receive the statistic computed from the
    whole training fold (including themselves). Safe w.r.t. the validation fold,
    but the training rows see a slightly optimistic encoding.
    """
    maps = fit_target_encoding_xgb(
        src_tr.iloc[train_idx], np.asarray(y)[train_idx], cols, smoothing, min_samples
    )
    return (
        apply_target_encoding_xgb(src_tr.iloc[train_idx], maps),
        apply_target_encoding_xgb(src_tr.iloc[valid_idx], maps),
        apply_target_encoding_xgb(src_te, maps),
    )


# 【不採用】Out-of-Fold の Target Encoding(平滑化 1 種類)を作る
def fit_apply_te_cv_nested_xgb(
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
        maps = fit_target_encoding_xgb(
            fit_src.iloc[in_tr], fit_y[in_tr], cols, smoothing, min_samples
        )
        te_train.iloc[in_va] = apply_target_encoding_xgb(fit_src.iloc[in_va], maps).values

    # validation / test: statistic of the full outer training fold
    full_maps = fit_target_encoding_xgb(fit_src, fit_y, cols, smoothing, min_samples)
    return (
        te_train,
        apply_target_encoding_xgb(src_tr.iloc[valid_idx], full_maps),
        apply_target_encoding_xgb(src_te, full_maps),
    )


# 【不採用】平滑化を複数並べた Target Encoding の対応表を作る
def fit_target_encoding_multi_xgb(src_fit, y_fit, cols, smoothings=(20.0,), min_samples=1):
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
            name = f"{c}_te{_smooth_tag_xgb(sm)}"
            maps[name] = (c, pd.Series(enc.astype("float32"), index=agg.index), prior)
    return maps


# 【不採用】平滑化を複数並べた対応表をまとめて当てる
def apply_target_encoding_multi_xgb(src: pd.DataFrame, maps):
    """複数 smooth の TE マップをまとめて適用する."""
    out = {}
    for name, (col, mapping, prior) in maps.items():
        out[name] = src[col].map(mapping).astype("float32").fillna(np.float32(prior))
    return pd.DataFrame(out, index=src.index)


# 【不採用】スタンド数などの行ごとの最小・最大・合計を作る
def add_row_aggregates_xgb(tr, te, src_tr, src_te):
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


# ---- CatBoost(もとは 03_feature_engineering_catboost.py) --------------------

# 【不採用】数値列の四則演算の列を追加する
def add_arithmetic_catboost(df: pd.DataFrame) -> list[str]:
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


# 【不採用】2 列を連結した交互作用の列を追加する
def add_interactions_catboost(
    df: pd.DataFrame, pairs: list[tuple[str, str]] | None = None
) -> list[str]:
    """Add string-concatenated 2-column interaction keys in place."""
    pairs = INTERACTION_PAIRS_CATBOOST if pairs is None else pairs
    new: list[str] = []
    for a, b in pairs:
        name = f"ix_{a}_{b}"
        df[name] = df[a].astype(str) + "_" + df[b].astype(str)
        new.append(name)
    return new


# 【不採用】値ごとの出現回数の列を追加する
def add_count_encoding_catboost(
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


# ---- RealMLP(もとは 03_feature_engineering_realmlp.py。--exact-te 用) --------------------

HIGHCARD_TE_COLS_REALMLP = ["Annual_Income_USD", "Daily_Commute_km"]


SMOOTH_KEY_SCALES_TE_REALMLP = (10, 100, 1000)


EXACT_TE_SMOOTHS_REALMLP = ("auto", 10.0, 100.0)


# 【不採用】厳密値 Target Encoding 用のキー 5 本のフレームを作る
def build_te_key_frame_realmlp(df: pd.DataFrame) -> pd.DataFrame:
    """厳密値2列 + income の Smooth Keys(/10,/100,/1000) = 5キーのフレームを返す.

    厳密値キーは小数1桁を ×10 して整数化し、float 等価判定の揺れを排除する
    (fe_lgbm.make_key_frame と同方式)。教師変数は使わないので train/test それぞれに
    直接適用してよい (リークしない)。
    """
    keys = pd.DataFrame(index=df.index)
    for c in HIGHCARD_TE_COLS_REALMLP:
        keys[c] = np.rint(df[c].to_numpy(dtype="float64") * 10).astype("int64")
    inc = df["Annual_Income_USD"].to_numpy(dtype="float64")
    for s in SMOOTH_KEY_SCALES_TE_REALMLP:
        keys[f"sk_inc{s}"] = np.floor(inc / s).astype("int64")
    return keys


# キーごとの購入者数と行数を集計する
def _te_agg_realmlp(arr, y):
    return pd.DataFrame({"k": arr, "y": y}).groupby("k", observed=True)["y"].agg(
        ["sum", "count"]
    )


# 集計から平滑化した購入率の対応表を作る
def _te_map_realmlp(agg, prior, smooth):
    cnt = agg["count"].to_numpy(dtype="float64")
    s = agg["sum"].to_numpy(dtype="float64")
    if isinstance(smooth, str):  # "auto" = sklearn TargetEncoder の経験ベイズ則
        p_i = s / cnt
        m = (p_i * (1.0 - p_i)) / (prior * (1.0 - prior))
    else:
        m = float(smooth)
    return pd.Series((s + prior * m) / (cnt + m), index=agg.index)


# 平滑化の強さを列名用の文字列にする
def _smooth_tag_realmlp(sm):
    if isinstance(sm, str):
        return sm
    f = float(sm)
    return str(int(f)) if f == int(f) else str(f).replace(".", "p")


# 【不採用】値の種類が多い列の Out-of-Fold Target Encoding を作る
def target_encode_highcard_realmlp(
    keys_fit: pd.DataFrame,
    y_fit,
    other_frames: list[pd.DataFrame],
    cols,
    smooths=EXACT_TE_SMOOTHS_REALMLP,
    n_inner: int = 5,
    seed: int = 42,
):
    """リークフリーの Out-of-Fold TE。戻り値 (te_fit, [te_other, ...]).

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
            sm: (f"te_{c}_s{_smooth_tag_realmlp(sm)}" if multi else f"te_{c}") for sm in smooths
        }

        vals = {sm: np.full(len(arr), prior, dtype="float32") for sm in smooths}
        for in_idx, out_idx in inner_splits:
            agg = _te_agg_realmlp(arr[in_idx], y_fit[in_idx])
            s_out = pd.Series(arr[out_idx])
            for sm in smooths:
                vals[sm][out_idx] = (
                    s_out.map(_te_map_realmlp(agg, prior, sm)).fillna(prior).to_numpy(dtype="float32")
                )
        for sm in smooths:
            te_fit[names[sm]] = vals[sm]

        agg_full = _te_agg_realmlp(arr, y_fit)
        for sm in smooths:
            m_full = _te_map_realmlp(agg_full, prior, sm)
            for f, out in zip(other_frames, te_others):
                out[names[sm]] = (
                    f[c].map(m_full).fillna(prior).to_numpy(dtype="float32")
                )
    return te_fit, te_others


# 【不採用】RealMLP の build_features から外した3つの特徴量(digit・通勤距離 ÷ 年齢・通勤距離 /5)を作る
def realmlp_rejected_extras(df: pd.DataFrame, category_map: dict, fit: bool) -> pd.DataFrame:
    """RealMLP の build_features にあった、既定で無効の3つの分岐をまとめたもの。

    - digit(--digits): 年収・通勤距離の各桁をカテゴリとして embedding に渡す。fold 1 で -0.00002
    - 通勤距離 ÷ 年齢(ratio): 2026-09-27 に削除。外しても単体 -0.000011(z=-0.75)
    - 通勤距離 /5 のキー(km5): 2026-09-27 に削除。外しても単体 -0.000008(z=-0.51)
    """
    df = df.copy()
    df["_Daily_Commute_km_/_Age"] = (df["Daily_Commute_km"] / (df["Age"] + 1e-6)).astype("float32")
    df["Daily_km_/_5_floor_"] = np.floor(df["Daily_Commute_km"] / 5.0).astype("int64")
    DIGIT_COLS = ["Annual_Income_USD", "Daily_Commute_km"]
    # ── digit features (値の種類（ユニーク値）が多い2列の各桁をカテゴリとして embedding に渡す) ──
    #    全列が小数1桁なので10倍して整数化し、浮動小数の丸め誤差を避ける。
    #    定数になる桁は fit 時に落とし、test でも同じ列集合を使う。
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
    return df
