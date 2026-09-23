"""2つの OOF 予測を paired DeLong 検定で比較する。

全モデルが同じ fold 分割 (StratifiedKFold(5, shuffle=True, random_state=42)) を使っているため、
2つの予測は「まったく同じ行・同じ分割」で作られている。DeLong 検定はこの対応関係を利用して
AUC の差の標準誤差を直接求めるので、AUC を別々に眺めるより桁違いに細かく差を見分けられる
(ノイズ床 0.00015 -> 0.00003 程度)。

使い方:
    uv run src/07_compare_predictions.py <base> <new> [<new2> ...]
    uv run src/07_compare_predictions.py --all <base>     # base と他の全候補を比較

<base>/<new> は oof/oof_<name>.npy の <name>、または .npy への直接パス。

参考: Sun & Xu (2014) の高速 DeLong 実装。
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

SEARCH_DIRS = ["oof", "experiments_rejected", "experiments_rejected/digit_ab", "backup_20260918/oof"]


def _midrank(x):
    """同順位を平均順位で扱う midrank。DeLong の分散計算に必要。"""
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and x_sorted[j + 1] == x_sorted[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_cov(preds, y):
    """複数の予測について AUC と、その共分散行列を返す。

    preds : (k, n) の予測値、y : (n,) の 0/1 ラベル
    """
    pos = preds[:, y == 1]
    neg = preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]

    tx = np.array([_midrank(pos[i]) for i in range(k)])
    ty = np.array([_midrank(neg[i]) for i in range(k)])
    tz = np.array([_midrank(np.concatenate([pos[i], neg[i]])) for i in range(k)])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:, :m] - tx) / n            # 正例側の構造成分
    v10 = 1.0 - (tz[:, m:] - ty) / m      # 負例側の構造成分
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def paired_test(y, p_base, p_new):
    """AUC の差・標準誤差・z 値・両側 p 値を返す。"""
    aucs, cov = delong_cov(np.vstack([p_base, p_new]), y)
    diff = aucs[1] - aucs[0]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(max(var, 0.0)))
    z = diff / se if se > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    return aucs[0], aucs[1], diff, se, z, p


def resolve(name):
    """名前またはファイルパスから .npy のパスを引く。"""
    if name.endswith(".npy") and os.path.exists(name):
        return name
    for d in SEARCH_DIRS:
        for pat in (f"{d}/oof_{name}.npy", f"{d}/**/oof_{name}.npy"):
            hit = glob.glob(pat, recursive=True)
            if hit:
                return hit[0]
    raise SystemExit(f"見つかりません: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="比較の基準となる OOF の名前")
    ap.add_argument("new", nargs="*", help="比較対象(複数可)")
    ap.add_argument("--all", action="store_true", help="oof/ の全候補を base と比較")
    args = ap.parse_args()

    y = (pd.read_csv("data/train.csv")["Will_Buy_EV"] == "Yes").astype(int).values
    base_path = resolve(args.base)
    p_base = np.load(base_path)

    targets = args.new
    if args.all:
        targets = sorted(
            os.path.basename(f)[4:-4] for f in glob.glob("oof/oof_*.npy")
            if os.path.basename(f)[4:-4] != args.base
        )

    print(f"base: {args.base}  ({base_path})  AUC={roc_auc_score(y, p_base):.6f}")
    print(f"{'候補':24s} {'AUC':>9s} {'差分':>10s} {'SE':>9s} {'z':>7s} {'p値':>9s}  判定")
    print("-" * 82)

    rows = []
    for name in targets:
        p_new = np.load(resolve(name))
        a0, a1, diff, se, z, p = paired_test(y, p_base, p_new)
        # happyc0der の採否規則: 差分 >= +0.00008 かつ z >= 3
        if diff >= 8e-5 and z >= 3:
            verdict = "採用"
        elif diff <= -8e-5 and z <= -3:
            verdict = "悪化"
        else:
            verdict = "誤差"
        rows.append((name, a1, diff, se, z, p, verdict))
        print(f"{name:24s} {a1:9.6f} {diff:+10.6f} {se:9.6f} {z:+7.2f} {p:9.2e}  {verdict}")

    if len(rows) > 1:
        print("\n差分の大きい順:")
        for r in sorted(rows, key=lambda r: -r[2])[:10]:
            print(f"  {r[0]:24s} {r[2]:+.6f}  z={r[4]:+.2f}  {r[6]}")


if __name__ == "__main__":
    main()
