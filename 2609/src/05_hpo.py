"""④ HPO — ハイパーパラメータ探索のフレーム。

設計方針:
- **学習コードは書かない。** 既存の `04_fe_run_<model>.py` を引数違いで呼ぶだけにする。
  収束設定(lr 引き下げ + early stopping)や FE 構成を本番とズラさないため。
- **1本ずつ順番に実行する。** 8コア環境で並列にすると CPU を取り合って完走しない。
- **実行前に必ず所要時間を見積もる。** `--estimate` で試行数 × 実測の1本あたり時間を出す。
- 判定は AUC の目視ではなく **paired DeLong 検定**(`07_compare_oof.py`)で行う。

使い方:
    uv run src/05_hpo.py lgbm --estimate          # 試行一覧と所要時間の見積もりだけ表示
    uv run src/05_hpo.py lgbm                     # 実行(1本ずつ順番に)
    uv run src/05_hpo.py lgbm --only num_leaves   # 特定の軸だけ
    uv run src/05_hpo.py --report                 # これまでの結果を表で表示

結果は `hpo_results.csv` に追記される。
"""

import argparse
import csv
import datetime
import os
import re
import subprocess
import time

RESULTS = "hpo_results.csv"

# 1本あたりの実測時間(秒)。フル5-fold・7スレッド。見積もりに使う。
RUNTIME = {"lgbm": 250, "xgb": 650, "catboost": 4000}

# 各モデルの現行ベスト(比較の基準)。07_compare_oof.py に渡す OOF 名も兼ねる。
BASELINE = {"lgbm": 0.946095, "xgb": 0.946077, "catboost": 0.94589, "realmlp": 0.945888}

# 本番構成の固定部分。ここは探索対象ではない(FE と収束設定)。
FIXED = {
    "lgbm": [
        "--patterns", "base,te1,cnt1,digit,sk", "--smooths", "auto,10,100",
        "--max_bin", "1024", "--feature_fraction", "0.3", "--max_depth", "5",
        "--folds", "5", "--learning_rate", "0.03", "--n_estimators", "8000",
        "--early_stopping", "200", "--n_jobs", "7", "--save",
    ],
    "xgb": [
        "--pattern", "tte_sk_dig", "--max-bin", "1024",
        "--set-param", "colsample_bytree=0.3", "--set-param", "max_depth=5",
        "--folds", "5", "--learning-rate", "0.03", "--n-estimators", "8000",
        "--early-stopping", "200", "--n-jobs", "7", "--save",
    ],
    "catboost": [
        "--fe", "te_all,catify,digits,skeys,te3", "--folds", "5",
        "--iters", "1000", "--lr", "0.06", "--fast",
        "--border", "64", "--hc-border", "1024", "--threads", "7",
    ],
}

SCRIPT = {
    "lgbm": "src/04_fe_run_lgbm.py",
    "xgb": "src/04_fe_run_xgb.py",
    "catboost": "src/04_fe_run_catboost.py",
}

# ── 探索空間 ────────────────────────────────────────────────────────────────
# {軸の名前: [(試行名, 上書きする引数), ...]}
# 現行ベストと同じ値は含めない(基準は BASELINE 側に持っているため)。
SPACE = {
    "lgbm": {
        # 葉の数。現在デフォルトの31のまま。特徴量が92列に増えているので伸びしろがありうる
        "num_leaves": [
            ("nl15", ["--num_leaves", "15"]),
            ("nl63", ["--num_leaves", "63"]),
            ("nl127", ["--num_leaves", "127"]),
        ],
        # 列サンプリングの比率。0.3 が最良かどうかの確認
        "feature_fraction": [
            ("ff02", ["--feature_fraction", "0.2"]),
            ("ff04", ["--feature_fraction", "0.4"]),
            ("ff05", ["--feature_fraction", "0.5"]),
        ],
        # 木の深さ。5 が最良かどうかの確認
        "max_depth": [
            ("d4", ["--max_depth", "4"]),
            ("d6", ["--max_depth", "6"]),
            ("d7", ["--max_depth", "7"]),
        ],
    },
    "xgb": {
        "max_depth": [
            ("d4", ["--set-param", "max_depth=4"]),
            ("d6", ["--set-param", "max_depth=6"]),
            ("d7", ["--set-param", "max_depth=7"]),
        ],
        "colsample": [
            ("cs02", ["--set-param", "colsample_bytree=0.2"]),
            ("cs04", ["--set-param", "colsample_bytree=0.4"]),
            ("cs05", ["--set-param", "colsample_bytree=0.5"]),
        ],
        "min_child_weight": [
            ("mcw5", ["--set-param", "min_child_weight=5"]),
            ("mcw20", ["--set-param", "min_child_weight=20"]),
        ],
        "subsample": [
            ("sub08", ["--set-param", "subsample=0.8"]),
            ("sub06", ["--set-param", "subsample=0.6"]),
        ],
    },
    "catboost": {
        # CatBoost は列サンプリング(rsm)が有害と判明済み。深さと正則化のみ探索する
        "depth": [
            ("d5", ["--depth", "5"]),
            ("d7", ["--depth", "7"]),
            ("d8", ["--depth", "8"]),
        ],
        "one_hot": [
            ("ohms16", ["--one-hot", "16"]),
            ("ohms64", ["--one-hot", "64"]),
        ],
    },
}

OOF_RE = re.compile(r"OOF[_ ]AUC[:=]\s*([0-9.]+)")


def build_trials(model, only=None):
    trials = []
    for axis, items in SPACE[model].items():
        if only and axis != only:
            continue
        for name, extra in items:
            trials.append((axis, f"{model}_hpo_{name}", extra))
    return trials


def tag_args(model, tag):
    """成果物を本番と別名で保存させる引数。CatBoost は保存しない(本番を上書きするため)。"""
    if model == "lgbm":
        return ["--tag", tag]
    if model == "xgb":
        return ["--out-suffix", "_" + tag.split("_hpo_")[1]]
    return ["--tag", tag]


def estimate(model, trials):
    per = RUNTIME[model]
    total = per * len(trials)
    print(f"モデル: {model}   試行数: {len(trials)} 本")
    print(f"1本あたり: 約 {per / 60:.1f} 分(フル5-fold・7スレッドの実測値)")
    print(f"**合計見積もり: 約 {total / 3600:.1f} 時間 ({total / 60:.0f} 分)**\n")
    print(f"{'軸':18s} {'試行名':22s} 引数")
    print("-" * 78)
    for axis, tag, extra in trials:
        print(f"{axis:18s} {tag:22s} {' '.join(extra)}")
    print("\n※ 1本ずつ順番に実行する(並列にすると CPU を取り合って完走しない)")
    print("※ 判定は実行後に: uv run src/07_compare_oof.py <baseline> <tag>")


def append_result(row):
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["日時", "モデル", "軸", "試行名", "OOF_AUC", "基準との差", "秒", "引数"])
        w.writerow(row)


def run(model, trials):
    print(f"{len(trials)} 本を順番に実行する。結果は {RESULTS} に追記。\n")
    for i, (axis, tag, extra) in enumerate(trials, 1):
        cmd = ["uv", "run", SCRIPT[model]] + FIXED[model] + extra + tag_args(model, tag)
        print(f"[{i}/{len(trials)}] {tag} ... ", end="", flush=True)
        t0 = time.time()
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        sec = time.time() - t0
        m = OOF_RE.search(out)
        auc = float(m.group(1)) if m else float("nan")
        diff = auc - BASELINE[model]
        print(f"OOF={auc:.6f}  基準差={diff:+.6f}  ({sec / 60:.1f}分)")
        append_result([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), model, axis, tag,
            f"{auc:.6f}", f"{diff:+.6f}", f"{sec:.0f}", " ".join(extra),
        ])
    print(f"\n完了。次は DeLong 検定で判定する:\n  uv run src/07_compare_oof.py {model} --all")


def report():
    if not os.path.exists(RESULTS):
        print(f"{RESULTS} がまだありません(未実行)。")
        return
    import pandas as pd
    df = pd.read_csv(RESULTS)
    print(df.sort_values("基準との差", ascending=False).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", choices=list(SPACE), help="探索するモデル")
    ap.add_argument("--only", help="特定の軸だけ実行する")
    ap.add_argument("--estimate", action="store_true", help="所要時間の見積もりだけ表示")
    ap.add_argument("--report", action="store_true", help="これまでの結果を表示")
    args = ap.parse_args()

    if args.report:
        report()
        return
    if not args.model:
        ap.error("model を指定するか --report を使ってください")

    trials = build_trials(args.model, args.only)
    if args.estimate:
        estimate(args.model, trials)
    else:
        run(args.model, trials)


if __name__ == "__main__":
    main()
