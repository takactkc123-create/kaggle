"""全OOFバリアントから貪欲法(hill climbing)で最適なブレンドを探索する。

収束させた結果モデル間相関が0.995まで上がりアンサンブル効果が半減したため、
未収束バリアントを含む全候補から多様性を含めて選び直す。
重複選択を許す forward selection で、選ばれた回数がそのまま重みになる。

採用されたモデル間の**順位相関**も併せて出力する。相関が高いほど同質で、
足してもアンサンブルは伸びない(弱くても非相関なら勝てる = diversity beats strength)。

⚠ この出力をそのまま信じないこと。貪欲法は OOF 上の偶然を拾う。
採用を判断する前に必ず `07_compare_oof.py` の DeLong 検定で有意性を確認する。
"""

import glob
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values

cands = {}
for f in sorted(glob.glob("oof/oof_*.npy")):
    name = os.path.basename(f).replace("oof_", "").replace(".npy", "")
    pred_path = f"oof/pred_{name}.npy"
    if os.path.exists(pred_path):
        cands[name] = (
            rankdata(np.load(f)) / len(y),
            rankdata(np.load(pred_path)) / len(test),
        )

print("=== 候補 ===")
for n, (o, _) in sorted(cands.items(), key=lambda kv: -roc_auc_score(y, kv[1][0])):
    print(f"{n:18s} {roc_auc_score(y, o):.5f}")

names = list(cands)
selected, cur_sum, best_auc = [], np.zeros(len(y)), 0.0

for step in range(15):
    scores = {
        n: roc_auc_score(y, (cur_sum + cands[n][0]) / (len(selected) + 1))
        for n in names
    }
    pick = max(scores, key=scores.get)
    if scores[pick] <= best_auc + 1e-7:
        break
    best_auc = scores[pick]
    selected.append(pick)
    cur_sum = cur_sum + cands[pick][0]
    print(f"step {step + 1}: +{pick:18s} -> {best_auc:.5f}")

weights = {n: selected.count(n) / len(selected) for n in set(selected)}
print(f"\n最終 OOF AUC: {best_auc:.5f}")
print("重み:", {k: round(v, 4) for k, v in sorted(weights.items(), key=lambda kv: -kv[1])})

# 採用されたモデル同士がどれだけ似ているか(低いほどアンサンブルに効く)
picked = sorted(weights, key=lambda n: -weights[n])
if len(picked) > 1:
    print("\n=== 採用モデル間の順位相関(低いほど多様)===")
    for a, b in itertools.combinations(picked, 2):
        rho = spearmanr(cands[a][0], cands[b][0]).statistic
        note = "  ← 同質" if rho >= 0.995 else ""
        print(f"{a:20s} x {b:20s} {rho:.5f}{note}")

    # 採用されなかった候補のうち、最も非相関なものを多様性の補充候補として提示する
    rest = [n for n in names if n not in weights]
    if rest:
        worst = {n: max(spearmanr(cands[n][0], cands[p][0]).statistic for p in picked)
                 for n in rest}
        low = sorted(worst.items(), key=lambda kv: kv[1])[:3]
        print("\n未採用のうち最も非相関な候補(多様性の補充先):")
        for n, rho in low:
            print(f"  {n:20s} 採用モデルとの最大相関 {rho:.5f}  (単体 {roc_auc_score(y, cands[n][0]):.5f})")

test_pred = sum(w * cands[n][1] for n, w in weights.items())
pd.DataFrame({"id": test["id"], "Will_Buy_EV": test_pred}).to_csv(
    "submit/submission_hillclimb.csv", index=False
)
print("saved submit/submission_hillclimb.csv")
