"""生成された特徴量の列名をダンプする共通ヘルパー。

各 `04_train_and_evaluate_<model>.py` は `--dump-features` を受け取ると、**fold 1 の学習行列を
組み上げた直後**に本モジュールを呼び、列名一覧を `docs/features_<tag>.json` に
書いて終了する(学習はしない)。本番と同じコードパスを通るので、列の取りこぼしがない。

`notebooks/03_feature_engineering.ipynb` はこの JSON を読んで表示する。分類規則(`classify`)も
ここに置き、ノートブックと実行スクリプトで同じ定義を共有する。

再生成:
    uv run src/04_train_and_evaluate_lgbm.py --patterns base,te1,cnt1,digit,sk,te2home --smooths auto,10,100 --dedup --lean \
      --sample 0.01 --folds 1 --dump-features --tag lgbm
"""

from __future__ import annotations

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs")

# 生データの13列(id / 目的変数を除く)
RAW_COLS = [
    "Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
    "Gender", "City_Type", "Current_Car_Type",
    "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level",
]

# 判定は上から順に適用する(先に当たったものを採用)。順序に意味がある:
# 例 `Income_/_100_floor_` は「_/_」を含むが四則演算ではなく Smooth Key なので先に拾う。
_RULES = [
    ("元データ由来",            re.compile(r"orig$|org_mean", re.I)),
    ("Target Encoding",        re.compile(r"(^te\d*_)|(_te$)|(_te(a|\d+)$)|(TE$)|(_target)")),
    ("Count / Frequency",      re.compile(r"(^cnt_)|(_ce$)|(^freq_)|(_count$)", re.I)),
    ("digit / 小数の分解",       re.compile(r"(^d-?\d+_)|(_d-?\d+$)|digit|(_decimal$)", re.I)),
    ("Smooth Keys(粗い解像度)", re.compile(r"(_floor_?$)|(_sk\d*$)|(^sk_)|(^inc_f\d+$)|(^commute_f\d+$)")),
    ("catify(数値→カテゴリ)",    re.compile(r"_cat_?$")),
    ("ビン分割",                re.compile(r"_bin_?$", re.I)),
    ("フラグ",                  re.compile(r"(^|_)is_", re.I)),
    ("四則演算",                re.compile(r"_(diff|ratio|sum|avg|prod)_|_/_|_[-+*]_", re.I)),
]


# 列名を特徴量の種類に振り分ける
def classify(name: str) -> str:
    """列名を FE の種類に振り分ける。未知のものは『その他の生成列』。"""
    if name in RAW_COLS:
        return "① 生の列"
    for label, pat in _RULES:
        if pat.search(name):
            return label
    # 生の列名が 2 つ以上埋まっていれば連結キー(交互作用)とみなす
    if sum(1 for c in RAW_COLS if c in name) >= 2:
        return "交互作用キー"
    return "その他の生成列"


# Target Encoding の列名から元のキー名を取り出す
def te_key_of(name: str) -> str:
    """TE 列から元になったキー名を取り出す。

    smooth の付き方がモデルごとに違うので、3通りとも剥がして正規化する::

        LightGBM  te_Age_sauto   te_sk_inc1000_s100
        XGBoost   Age_tea        inc_f1000_te100
        CatBoost  te10_Age       te100_sk_inc_1000
    """
    s = re.sub(r"^te(a|\d+)?_", "", name)      # CatBoost / LightGBM の接頭辞
    s = re.sub(r"_te(a|\d+)?$", "", s)         # XGBoost の接尾辞
    s = re.sub(r"_s(auto|\d+)$", "", s)        # LightGBM の smooth タグ
    s = re.sub(r"_TE$", "", s)                 # RealMLP
    return s


# 列名の一覧を docs/features_<tag>.json に書き出す
def dump(tag: str, model: str, columns, cat_features=None, note: str = "") -> str:
    """列名一覧を docs/features_<tag>.json に書き出してパスを返す。"""
    cols = [str(c) for c in columns]
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"features_{tag}.json")
    payload = {
        "tag": tag,
        "model": model,
        "note": note,
        "n_features": len(cols),
        "cat_features": sorted(str(c) for c in (cat_features or [])),
        "columns": cols,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"[dump-features] {model}: {len(cols)} 列 -> docs/features_{tag}.json", flush=True)
    return path


# 書き出した列名の JSON を読む
def load(tag: str) -> dict:
    """ダンプ済み JSON を読む(ノートブック用)。"""
    with open(os.path.join(OUT_DIR, f"features_{tag}.json"), encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# FE 関数の採否
# ---------------------------------------------------------------------------
# `03_feature_engineering_<model>.py` には本番で使う関数だけを置き、検証して捨てた施策は
# `03_feature_engineering_all.py` の「不採用(記録)」に、名前の末尾にモデル名を付けて移してある(2026-09-28)。
# どれが本番で生きているのかを、ここに一覧で持つ。
#
#   ADOPTED  = 本番の構成で実際に呼ばれている
#   REJECTED = 検証したうえで不採用。根拠を併記する(再検証不要)
#   SUPPORT  = 補助・基盤。単体で採否を論じるものではない
#
# **この表は手で保つ。** ただし列を作る関数については `verify_status()` が
# `docs/features_<tag>.json` と突き合わせて矛盾を検出できる。

ADOPTED, REJECTED, SUPPORT = "〇", "✖", "—"

# (モジュール, 関数名) -> (記号, 根拠)
FUNC_STATUS = {
    # ---- 03_feature_engineering_lgbm.py (本番: 04 が te_plan() どおりに必要な列だけ作る) ----
    ("03_feature_engineering_lgbm", "load_data"):               (SUPPORT,  "読み込み"),
    ("03_feature_engineering_lgbm", "make_categorical"):        (ADOPTED,  "native category として渡す"),
    ("03_feature_engineering_lgbm", "make_key_frame"):          (ADOPTED,  "TE / Count のキー生成"),
    ("03_feature_engineering_lgbm", "add_digit_features"):      (ADOPTED,  "15列(うち元の列と同じ2列は dedup で除外)"),
    ("03_feature_engineering_lgbm", "add_smooth_keys"):         (ADOPTED,  "TEキー4本を追加 (sk)"),
    ("03_feature_engineering_lgbm", "count_encode"):            (ADOPTED,  "+0.00083"),
    ("03_feature_engineering_lgbm", "target_encode_fold"):      (ADOPTED,  "最大の改善要因"),
    ("03_feature_engineering_lgbm", "single_keys"):             (ADOPTED,  "te1 / cnt1 のキー集合"),
    ("03_feature_engineering_lgbm", "te_plan"):               (ADOPTED,  "本番の Target Encoding の設計(キーと平滑化)"),
    ("03_feature_engineering_lgbm", "target_encode_plan"):    (ADOPTED,  "te_plan() のとおりに Target Encoding を作る"),

    # ---- 03_feature_engineering_xgb.py (本番: 04 が te_plan() どおりに必要な列だけ作る) ----
    ("03_feature_engineering_xgb", "as_ordinal"):               (ADOPTED,  "XGBoost の最終採用方式"),
    ("03_feature_engineering_xgb", "add_count_encoding"):       (ADOPTED,  "+0.00049"),
    ("03_feature_engineering_xgb", "digit_block"):              (SUPPORT,  "add_digit_features の内部"),
    ("03_feature_engineering_xgb", "add_digit_features"):       (ADOPTED,  "16列 (小数第1位を含む。うち2列は dedup で除外)"),
    ("03_feature_engineering_xgb", "make_smooth_keys"):         (ADOPTED,  "TEキー4本を追加"),
    ("03_feature_engineering_xgb", "prepare_te_codes"):         (SUPPORT,  "TEキーの整数コード化(高速化)"),
    ("03_feature_engineering_xgb", "fit_apply_te_cv_nested_multi"): (ADOPTED, "本番の Out-of-Fold Triple TE"),
    ("03_feature_engineering_xgb", "te_plan"):                (ADOPTED,  "本番の Target Encoding の設計(キーと平滑化)"),
    ("03_feature_engineering_xgb", "fit_apply_te_cv_nested_plan"): (ADOPTED,  "te_plan() のとおりに Target Encoding を作る"),

    # ---- 03_feature_engineering_catboost.py (本番: 04 が te_plan() どおりに必要な列だけ作る) ----
    ("03_feature_engineering_catboost", "add_digits"):          (ADOPTED,  "+0.00061。既定ビン64が粗いため効いた"),
    ("03_feature_engineering_catboost", "drop_constant"):       (SUPPORT,  "定数列の除去"),
    ("03_feature_engineering_catboost", "add_smooth_keys"):     (ADOPTED,  "TEキー4本を追加 (skeys)"),
    ("03_feature_engineering_catboost", "target_encode"):       (ADOPTED,  "te_all + te3 (Triple smooth)"),
    ("03_feature_engineering_catboost", "cast_to_str"):         (ADOPTED,  "catify +0.00170。CatBoost 単体最大"),
    ("03_feature_engineering_catboost", "te_plan"):           (ADOPTED,  "本番の Target Encoding の設計(キーと平滑化)"),
    ("03_feature_engineering_catboost", "target_encode_plan"): (ADOPTED,  "te_plan() のとおりに Target Encoding を作る"),

    # ---- 03_feature_engineering_realmlp.py (本番: 追加フラグなし) ----
    ("03_feature_engineering_realmlp", "build_features"):       (ADOPTED,  "catify / ビン分割 / Smooth Keys を一括生成"),
    ("03_feature_engineering_realmlp", "load_orig"):            (ADOPTED,  "元データ由来の org_mean。アンサンブル +0.000022"),

    # ---- 03_feature_engineering_all.py (横断カタログ。どのモデルの本番でも import されない) ----
    ("03_feature_engineering_all", "load_data"):                (SUPPORT,  "読み込み"),
    ("03_feature_engineering_all", "get_y"):                    (SUPPORT,  "目的変数の0/1化"),
    ("03_feature_engineering_all", "as_native_category"):       (ADOPTED,  "LightGBM が採用"),
    ("03_feature_engineering_all", "as_ordinal"):               (ADOPTED,  "XGBoost が採用"),
    ("03_feature_engineering_all", "as_str"):                   (ADOPTED,  "CatBoost の catify"),
    ("03_feature_engineering_all", "make_key_frame"):           (ADOPTED,  "厳密値キー。全モデル"),
    ("03_feature_engineering_all", "add_smooth_keys"):          (ADOPTED,  "全モデル"),
    ("03_feature_engineering_all", "smooth_key_names"):         (SUPPORT,  "キー名の生成"),
    ("03_feature_engineering_all", "add_digit_features"):       (ADOPTED,  "全モデル"),
    ("03_feature_engineering_all", "drop_constant_cols"):       (SUPPORT,  "定数列の除去"),
    ("03_feature_engineering_all", "count_encode"):             (ADOPTED,  "LightGBM / XGBoost のみ"),
    ("03_feature_engineering_all", "target_encode_fold"):       (ADOPTED,  "GBDT3種"),
    ("03_feature_engineering_all", "catify"):                   (ADOPTED,  "CatBoost のみ。他モデルでは逆効果"),
    ("03_feature_engineering_all", "arithmetic_meaningful"):    (REJECTED, "四則演算。打ち止め"),
    ("03_feature_engineering_all", "interaction_keys"):         (REJECTED, "交互作用キー。全滅"),
    ("03_feature_engineering_all", "cat_pairs"):                (REJECTED, "同上のペア列挙"),
    ("03_feature_engineering_all", "add_income_neighborhood"): (REJECTED, "年収の近傍統計。GBDT は誤差、RealMLP は単体 +0.000077 だがアンサンブル ±0"),

    # ---- 03_feature_engineering_all.py の「不採用(記録)」(各モデルから移した関数。名前の末尾がもとのモデル) ----
    ("03_feature_engineering_all", "add_arithmetic_meaningful_lgbm"): (REJECTED, "四則演算 -0.00014。打ち止め"),
    ("03_feature_engineering_all", "add_arithmetic_all_pairs_lgbm"): (REJECTED, "全ペア四則演算。同上"),
    ("03_feature_engineering_all", "add_group_means_lgbm"):         (REJECTED, "行方向の平均。効果なし"),
    ("03_feature_engineering_all", "add_subsidy_products_lgbm"): (REJECTED, "補助金との積。+0.000009 (z=+0.65) で誤差"),
    ("03_feature_engineering_all", "pair_keys_lgbm"):               (REJECTED, "2列交互作用TE(全ペア)。全滅。自宅充電×自宅スタンド数の1組だけは te2home で最終構成に採用"),
    ("03_feature_engineering_all", "triple_keys_lgbm"):             (REJECTED, "3列交互作用TE。全滅"),
    ("03_feature_engineering_all", "all_columns_key_lgbm"):         (REJECTED, "行フィンガープリント。全行ユニークで原理的に不可"),
    ("03_feature_engineering_all", "fingerprint_key_lgbm"):         (REJECTED, "同上(部分集合版)"),
    ("03_feature_engineering_all", "make_base_xgb"):                (REJECTED, "native category の素のフレーム。ordinal を採用したので使っていない"),
    ("03_feature_engineering_all", "as_native_category_xgb"):       (REJECTED, "ordinal と差なし。非相関性を狙い ordinal を採用"),
    ("03_feature_engineering_all", "as_onehot_xgb"):                (REJECTED, "列が増えるだけで効果なし"),
    ("03_feature_engineering_all", "add_arithmetic_xgb"):           (REJECTED, "四則演算。打ち止め"),
    ("03_feature_engineering_all", "all_numeric_pairs_xgb"):        (REJECTED, "同上のペア列挙"),
    ("03_feature_engineering_all", "make_interaction_keys_xgb"):    (REJECTED, "交互作用キー。全滅"),
    ("03_feature_engineering_all", "cat_pairs_xgb"):                (REJECTED, "同上のペア列挙"),
    ("03_feature_engineering_all", "fit_target_encoding_xgb"):      (REJECTED, "Out-of-Fold でない旧版。Out-of-Fold 版が +0.00108 で置き換え"),
    ("03_feature_engineering_all", "apply_target_encoding_xgb"):    (REJECTED, "同上"),
    ("03_feature_engineering_all", "fit_apply_te_cv_xgb"):          (REJECTED, "同上"),
    ("03_feature_engineering_all", "fit_apply_te_cv_nested_xgb"):   (REJECTED, "単一 smooth 版。Triple TE が置き換え"),
    ("03_feature_engineering_all", "fit_target_encoding_multi_xgb"): (REJECTED, "同上(Out-of-Fold でない複数smooth版)"),
    ("03_feature_engineering_all", "apply_target_encoding_multi_xgb"): (REJECTED, "同上"),
    ("03_feature_engineering_all", "add_row_aggregates_xgb"):       (REJECTED, "行方向の集約。効果なし"),
    ("03_feature_engineering_all", "add_arithmetic_catboost"):      (REJECTED, "四則演算 -0.00099。最も悪化"),
    ("03_feature_engineering_all", "add_interactions_catboost"):    (REJECTED, "交互作用キー。全滅"),
    ("03_feature_engineering_all", "add_count_encoding_catboost"):  (REJECTED, "内部の Ordered TS と重複して無効"),
    ("03_feature_engineering_all", "build_te_key_frame_realmlp"):   (REJECTED, "--exact-te 用。単体 +0.000156 だがアンサンブル寄与ゼロ"),
    ("03_feature_engineering_all", "target_encode_highcard_realmlp"): (REJECTED, "同上。GBDTとの相関が上がり多様性を損なう"),
    ("03_feature_engineering_all", "realmlp_rejected_extras"): (REJECTED, "RealMLP の digit・通勤距離÷年齢・通勤距離/5。いずれも誤差"),
    ("03_feature_engineering_all", "orig_income_rate_gbdt"):   (REJECTED, "元データの年収ごとの購入率を GBDT に。LightGBM -0.000003 / XGBoost -0.000005 / CatBoost -0.000012 で誤差(2026-09-29)")
}


# 関数の採否(記号と根拠)を返す
def status_of(module: str, func: str):
    """(記号, 根拠) を返す。未登録なら ("?", "未分類")。"""
    return FUNC_STATUS.get((module, func), ("?", "未分類"))


# 「この関数が採用されていれば、この種類の列が本番に存在するはず」という対応。
# verify_status() がこれを使って表と実データの矛盾を検出する。
_EXPECT = {
    "digit / 小数の分解": ["add_digit_features", "add_digits", "digit_block"],
    "Count / Frequency": ["count_encode", "add_count_encoding"],
    # RealMLP の target_encode_highcard はここに入れない。RealMLP の本番にも TE 列は
    # 2本あるが、それは 04_train_and_evaluate_realmlp.py が sklearn の TargetEncoder で作るもので、
    # この関数(--exact-te 専用)とは別経路。種類の有無では判別できない。
    "Target Encoding":   ["target_encode_fold", "target_encode",
                          "fit_apply_te_cv_nested_multi"],
    "四則演算":           ["add_arithmetic_meaningful", "add_arithmetic_all_pairs",
                          "add_arithmetic", "arithmetic_meaningful"],
    "交互作用キー":        ["pair_keys", "triple_keys", "make_interaction_keys",
                          "add_interactions", "interaction_keys"],
}
_MODULE_OF_TAG = {"lgbm": "03_feature_engineering_lgbm", "xgb": "03_feature_engineering_xgb",
                  "catboost": "03_feature_engineering_catboost", "realmlp": "03_feature_engineering_realmlp"}


# 採否表と docs/features_<tag>.json の矛盾を洗い出す
def verify_status(tags=("lgbm", "xgb", "catboost", "realmlp")):
    """採否表と docs/features_<tag>.json の矛盾を洗い出して行のリストで返す。

    「〇 なのにその種類の列が 1 つも無い」「✖ なのに列がある」を検出する。
    空リストなら矛盾なし。
    """
    rows = []
    for tag in tags:
        mod = _MODULE_OF_TAG[tag]
        groups = {classify(c) for c in load(tag)["columns"]}
        for group, funcs in _EXPECT.items():
            for fn in funcs:
                if (mod, fn) not in FUNC_STATUS:
                    continue
                mark, why = FUNC_STATUS[(mod, fn)]
                if mark == SUPPORT:
                    continue
                present = group in groups
                if mark == ADOPTED and not present:
                    rows.append(f"{mod}.{fn}: 〇 だが本番に「{group}」の列が無い")
                if mark == REJECTED and present:
                    rows.append(f"{mod}.{fn}: ✖ だが本番に「{group}」の列がある ({why})")
    return rows
