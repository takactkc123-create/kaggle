"""生成された特徴量の列名をダンプする共通ヘルパー。

各 `04_fe_run_<model>.py` は `--dump-features` を受け取ると、**fold 1 の学習行列を
組み上げた直後**に本モジュールを呼び、列名一覧を `docs/features_<tag>.json` に
書いて終了する(学習はしない)。本番と同じコードパスを通るので、列の取りこぼしがない。

`notebooks/03_fe.ipynb` はこの JSON を読んで表示する。分類規則(`classify`)も
ここに置き、ノートブックと実行スクリプトで同じ定義を共有する。

再生成:
    uv run src/04_fe_run_lgbm.py --patterns base,te1,cnt1,digit,sk --smooths auto,10,100 \
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
