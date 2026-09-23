"""生成された特徴量の列名をダンプする共通ヘルパー。

各 `04_train_and_evaluate_<model>.py` は `--dump-features` を受け取ると、**fold 1 の学習行列を
組み上げた直後**に本モジュールを呼び、列名一覧を `docs/features_<tag>.json` に
書いて終了する(学習はしない)。本番と同じコードパスを通るので、列の取りこぼしがない。

`notebooks/03_feature_engineering.ipynb` はこの JSON を読んで表示する。分類規則(`classify`)も
ここに置き、ノートブックと実行スクリプトで同じ定義を共有する。

再生成:
    uv run src/04_train_and_evaluate_lgbm.py --patterns base,te1,cnt1,digit,sk --smooths auto,10,100 \
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


def load(tag: str) -> dict:
    """ダンプ済み JSON を読む(ノートブック用)。"""
    with open(os.path.join(OUT_DIR, f"features_{tag}.json"), encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# FE 関数の採否
# ---------------------------------------------------------------------------
# `03_feature_engineering_<model>.py` は**採用した関数だけを置く場所ではない**。検証して捨てた施策も
# 「再検証しないための記録」として残してある。どれが本番で生きているのかを
# コードから読み取るのは難しいので、ここに一覧で持つ。
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
    # ---- 03_feature_engineering_lgbm.py (本番: --patterns base,te1,cnt1,digit,sk) ----
    ("03_feature_engineering_lgbm", "load_data"):               (SUPPORT,  "読み込み"),
    ("03_feature_engineering_lgbm", "make_categorical"):        (ADOPTED,  "native category として渡す"),
    ("03_feature_engineering_lgbm", "make_key_frame"):          (ADOPTED,  "TE / Count のキー生成"),
    ("03_feature_engineering_lgbm", "add_arithmetic_meaningful"): (REJECTED, "四則演算 -0.00014。打ち止め"),
    ("03_feature_engineering_lgbm", "add_arithmetic_all_pairs"): (REJECTED, "全ペア四則演算。同上"),
    ("03_feature_engineering_lgbm", "add_group_means"):         (REJECTED, "行方向の平均。効果なし"),
    ("03_feature_engineering_lgbm", "add_digit_features"):      (ADOPTED,  "15列。digit パターン"),
    ("03_feature_engineering_lgbm", "add_smooth_keys"):         (ADOPTED,  "TEキー4本を追加 (sk)"),
    ("03_feature_engineering_lgbm", "count_encode"):            (ADOPTED,  "+0.00083"),
    ("03_feature_engineering_lgbm", "target_encode_fold"):      (ADOPTED,  "最大の改善要因"),
    ("03_feature_engineering_lgbm", "single_keys"):             (ADOPTED,  "te1 / cnt1 のキー集合"),
    ("03_feature_engineering_lgbm", "pair_keys"):               (REJECTED, "2列交互作用TE。全滅"),
    ("03_feature_engineering_lgbm", "triple_keys"):             (REJECTED, "3列交互作用TE。全滅"),
    ("03_feature_engineering_lgbm", "all_columns_key"):         (REJECTED, "行フィンガープリント。全行ユニークで原理的に不可"),
    ("03_feature_engineering_lgbm", "fingerprint_key"):         (REJECTED, "同上(部分集合版)"),

    # ---- 03_feature_engineering_xgb.py (本番: --pattern tte_sk_dig) ----
    ("03_feature_engineering_xgb", "make_base"):                (ADOPTED,  "素の特徴量フレーム"),
    ("03_feature_engineering_xgb", "as_native_category"):       (REJECTED, "ordinal と差なし。非相関性を狙い ordinal を採用"),
    ("03_feature_engineering_xgb", "as_ordinal"):               (ADOPTED,  "XGBoost の最終採用方式"),
    ("03_feature_engineering_xgb", "as_onehot"):                (REJECTED, "列が増えるだけで効果なし"),
    ("03_feature_engineering_xgb", "add_arithmetic"):           (REJECTED, "四則演算。打ち止め"),
    ("03_feature_engineering_xgb", "all_numeric_pairs"):        (REJECTED, "同上のペア列挙"),
    ("03_feature_engineering_xgb", "add_count_encoding"):       (ADOPTED,  "+0.00049"),
    ("03_feature_engineering_xgb", "digit_block"):              (SUPPORT,  "add_digit_features の内部"),
    ("03_feature_engineering_xgb", "add_digit_features"):       (ADOPTED,  "16列 (小数第1位を含む)"),
    ("03_feature_engineering_xgb", "make_smooth_keys"):         (ADOPTED,  "TEキー4本を追加"),
    ("03_feature_engineering_xgb", "make_interaction_keys"):    (REJECTED, "交互作用キー。全滅"),
    ("03_feature_engineering_xgb", "cat_pairs"):                (REJECTED, "同上のペア列挙"),
    ("03_feature_engineering_xgb", "fit_target_encoding"):      (REJECTED, "非入れ子の旧版。入れ子版が +0.00108 で置き換え"),
    ("03_feature_engineering_xgb", "apply_target_encoding"):    (REJECTED, "同上"),
    ("03_feature_engineering_xgb", "fit_apply_te_cv"):          (REJECTED, "同上"),
    ("03_feature_engineering_xgb", "fit_apply_te_cv_nested"):   (REJECTED, "単一 smooth 版。Triple TE が置き換え"),
    ("03_feature_engineering_xgb", "fit_target_encoding_multi"): (REJECTED, "同上(非入れ子の複数smooth版)"),
    ("03_feature_engineering_xgb", "apply_target_encoding_multi"): (REJECTED, "同上"),
    ("03_feature_engineering_xgb", "prepare_te_codes"):         (SUPPORT,  "TEキーの整数コード化(高速化)"),
    ("03_feature_engineering_xgb", "fit_apply_te_cv_nested_multi"): (ADOPTED, "本番の入れ子 Triple TE"),
    ("03_feature_engineering_xgb", "add_row_aggregates"):       (REJECTED, "行方向の集約。効果なし"),

    # ---- 03_feature_engineering_catboost.py (本番: --fe te_all,catify,digits,skeys,te3) ----
    ("03_feature_engineering_catboost", "add_arithmetic"):      (REJECTED, "四則演算 -0.00099。最も悪化"),
    ("03_feature_engineering_catboost", "add_interactions"):    (REJECTED, "交互作用キー。全滅"),
    ("03_feature_engineering_catboost", "add_digits"):          (ADOPTED,  "+0.00061。既定ビン64が粗いため効いた"),
    ("03_feature_engineering_catboost", "drop_constant"):       (SUPPORT,  "定数列の除去"),
    ("03_feature_engineering_catboost", "add_smooth_keys"):     (ADOPTED,  "TEキー4本を追加 (skeys)"),
    ("03_feature_engineering_catboost", "add_count_encoding"):  (REJECTED, "内部の Ordered TS と重複して無効"),
    ("03_feature_engineering_catboost", "target_encode"):       (ADOPTED,  "te_all + te3 (Triple smooth)"),
    ("03_feature_engineering_catboost", "cast_to_str"):         (ADOPTED,  "catify +0.00170。CatBoost 単体最大"),

    # ---- 03_feature_engineering_realmlp.py (本番: 追加フラグなし) ----
    ("03_feature_engineering_realmlp", "build_features"):       (ADOPTED,  "catify / ビン分割 / Smooth Keys を一括生成"),
    ("03_feature_engineering_realmlp", "build_te_key_frame"):   (REJECTED, "--exact-te 用。単体 +0.000156 だがアンサンブル寄与ゼロ"),
    ("03_feature_engineering_realmlp", "target_encode_highcard"): (REJECTED, "同上。GBDTとの相関が上がり多様性を損なう"),
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
}


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
