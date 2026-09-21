# RealMLP FE 検証結果

RealMLP リーダーの検証記録。採否基準は ±0.0002(CLAUDE.md 準拠)。

## 2026-09-18: 厳密値 Target Encoding(高カーデ2列+Smooth Keysに絞る版) — **不採用**

### 背景

FE Lead のコード実読(`fe_results_all.md` §3-3/§4)により、GBDT3種には入っている
「13列の厳密値TE」が RealMLP には無い(combo TE 2列のみ)ことが判明。54列一括投入は
CPU競合で完走せず、かつ RealMLP は列追加コストが高い(PBLD embedding が数値列ごとに
小ネットワークを持つ)ため、**効果源が確実な高カーディナリティ2列に絞った15列版**を検証した。

### 実装

- `fe_realmlp.py` に追加: `build_te_key_frame`(キー: `Annual_Income_USD`, `Daily_Commute_km` の
  厳密値 + `sk_inc10`/`sk_inc100`/`sk_inc1000` の Smooth Keys、計5キー)、
  `target_encode_highcard`(入れ子CV: outer fold 内で inner `StratifiedKFold(5)` を回し、
  学習行には inner-OOF、valid/test には学習fold全体の統計を適用してリーク防止)、
  smooth は `auto`/10/100 の Triple 同時投入 → **5キー × 3 smooth = 15列**。
- `realmlp_preprocessing.py` に `--exact-te` フラグを追加(デフォルト `False` = 既存動作と完全に同じ)。
  ON のときのみ fold ループ内で上記TEを計算し `X_tr`/`X_val`/`X_tst` に列追加する
  (`num_col_names` は `cat_col_names` 以外の全列として自動判定されるため、追加列は自動的に
  `NumericalPreprocessor`(median_center → robust_scale → smooth_clip)を通る = スケール整合は
  既存パイプラインに委ねられ、特別な対応は不要だった)。
- **エポック数・正則化スケジュールは一切変更していない**(2エポック厳守)。

### 忠実性チェック(ハーネス改変の副作用確認)

`--exact-te` なしで fold1 を再実行し、保存済み `oof_realmlp.npy` の fold1 と完全一致することを確認。
コード変更が既存パイプラインを壊していないことを保証できた。

| 実行 | fold1 AUC |
|---|---|
| 保存済み `oof_realmlp.npy`(公式) | 0.94496 |
| 再実行(`--exact-te` なし、コード変更後) | **0.94496**(完全一致) |

### A/B結果

**fold1 のみ(約20分)**:

| 構成 | fold1 AUC | 差分 |
|---|---|---|
| ベースライン | 0.94496 | — |
| `--exact-te`(15列) | 0.94516 | **+0.00020** |

閾値ちょうどで一見有望だったため、方針通りフル5-foldへ進めた。

**フル5-fold(約42分、7スレッド単独枠)**:

| fold | ベースライン | `--exact-te` | 差分 |
|---|---|---|---|
| 1 | 0.94496 | 0.94516 | +0.00020 |
| 2 | 0.94573 | 0.94581 | +0.00008 |
| 3 | 0.94675 | 0.94688 | +0.00013 |
| 4 | 0.94608 | 0.94634 | +0.00026 |
| 5 | 0.94594 | 0.94607 | +0.00013 |
| **OOF** | **0.94589** | **0.94604** | **+0.00015** |

全foldで一貫してプラスだが、**OOF全体では +0.00015 で採否基準 ±0.0002 に届かない**。

### GBDT3種との順位相関(アンサンブル価値の確認)

`realmlp_preprocessing.py` 内の自動チェックは `oof/oof_lgbm.npy` を参照するが、これは
**LightGBM の Triple TE 採用前の古い成果物(OOF 0.94535)** であり、公式の最終版は
`oof_lgbm_tte_dig.npy`(OOF 0.94587)である(LGBM Lead のファイル命名上の見落としと思われる。
自分のファイルではないため修正はせず、指揮官へ申告のみ)。正しいファイルで再計算した結果:

| ペア | ベースライン(RealMLP現行) | `--exact-te`版 | 変化 |
|---|---|---|---|
| RealMLP × LightGBM(`lgbm_tte_dig`) | 0.9920 | 0.9931 | **上昇**(非相関性↓) |
| RealMLP × XGBoost | 0.9923 | 0.9933 | **上昇**(非相関性↓) |
| RealMLP × CatBoost | 0.9953 | 0.9963 | **上昇**(非相関性↓) |

**厳密値TEを入れると、GBDTとの相関が全ペアで上昇する。** これは想定通りの結果:
GBDT側の厳密値TEは全く同じ情報源(値そのものと教師変数の関係)から作られているため、
RealMLPに同じ特徴を与えるとGBDTの予測分布に引き寄せられ、**多様性が失われる**。

### `ensemble_hillclimb.py` への影響

`oof/oof_realmlp_te_highcard.npy` を候補に追加して貪欲法を再実行:

```
=== 候補(単体OOF AUC)===
realmlp_te_highcard 0.94604   ← 新規、単体最強タイ
xgb                 0.94591
catboost            0.94589
realmlp              0.94589   ← 現行
lgbm_tte_dig         0.94587
...
step1: +realmlp_te_highcard -> 0.94604
step2: +xgb                 -> 0.94613
step3: +realmlp             -> 0.94617
step4: +catboost            -> 0.94619

最終 OOF AUC: 0.94619   (現行ベスト 0.94618 から +0.00001)
重み: realmlp 0.25 / realmlp_te_highcard 0.25 / xgb 0.25 / catboost 0.25
```

**アンサンブル全体でも +0.00001 とノイズレベルの変化に留まった。** さらに注目すべきは、
貪欲法が `realmlp`(現行)と `realmlp_te_highcard`(新版)を**両方**選び、`lgbm` を落としている点。
これは「新版が現行版を置き換えるほど強くない(相関が高すぎて置き換えの旨味が薄い)」ことを示しており、
新版単独でも既存の4モデルアンサンブルに対する上積みはほぼゼロと判断できる。

### 判定: **不採用**

- 単体OOF: +0.00015 で採否基準未達(境界線のわずかに下)
- 相関: 全GBDTペアで上昇 = 多様性の観点でも**逆方向**(改善ではなく悪化)
- アンサンブルCV: +0.00001 でノイズ床(0.00015〜0.00033)以下

3つの評価軸すべてで「効果なし」〜「わずかに逆効果」の一致した結論となった。

### 考察: なぜRealMLPでは厳密値TEが(GBDTほど)効かないのか

1. **既存実装が厳密値情報をすでに部分的に持っている。** `fe_realmlp.build_features` の
   `{col}_cat_`(floor値の factorize → embedding)が、TEが担うはずの「値ごとの粒度」情報を
   すでに embedding 経由でモデルに与えている。GBDTにはこの機構がなく、TEが唯一の値粒度情報
   だったのに対し、RealMLPでは重複気味の情報源になっていたと考えられる。
2. **相関上昇が示すとおり、TEが持ち込む情報はGBDTの決定関数に極めて近い。** 厳密値TEは
   「その値における経験的な購買確率」そのものであり、GBDT予測とほぼ同じ統計量を直接特徴として
   与える格好になる。RealMLPの非線形embedding経由の学習よりも直接的にGBDTの答えに寄せてしまい、
   非相関性という戦略的価値を損なう。
3. **信号自体はゼロではない**(全foldで一貫して微小プラス)。GBDTほどの破壊力(+0.003)がないのは、
   上記1の重複と、RealMLPが元々別の経路(PBLD周期埋め込み+embedding)で似た情報を近似できていた
   ためと考えられる。

### 成果物の取り扱い

- **`oof/oof_realmlp.npy` / `pred/pred_realmlp.npy` / `submit/submission_realmlp.csv` は変更していない**
  (不採用のため)。
- 実験結果は別名で保存: `oof/oof_realmlp_te_highcard.npy`, `oof/pred_realmlp_te_highcard.npy`,
  `submit/submission_realmlp_te_highcard.csv`(負の結果の記録として残す。`oof_realmlp_e6.npy` と同様の扱い)。
- コード変更(`fe_realmlp.py` の新規関数、`realmlp_preprocessing.py` の `--exact-te` フラグ)は
  **デフォルト無効**のまま残す。フラグを付けない限り既存動作と完全に同一であることを確認済みなので、
  「元に戻す」作業は不要(オプトイン方式のため常に安全にロールバック済み状態)。
- `ensemble_hillclimb.py` を実行した副作用として `submit/submission_hillclimb.csv` が
  新しい候補を含めて再生成された(このスクリプトは実行するたびに `oof/oof_*.npy` を全件走査して
  上書きする仕様のため)。今回の不採用判定を受け、**このファイルを公式のベスト構成として扱わないよう
  指揮官に申告する**(採用可否の最終判断は指揮官の管轄)。

### 副次的な発見(要指揮官確認・自分のファイルではないため修正せず)

`oof/oof_lgbm.npy`(OOF 0.94535)が LightGBM の Triple TE 採用前の古い成果物のままで、
最終版は `oof_lgbm_tte_dig.npy`(OOF 0.94587)という別名で保存されている。
`ensemble_hillclimb.py` は `oof_*.npy` を全件走査するため、この命名の不整合により
古い `lgbm` 候補と新しい `lgbm_tte_dig` 候補が両方候補プールに残っている
(実害は無いが整理の余地あり、LGBM Lead 管轄)。

## まとめ

| 指標 | 現行(公式) | `--exact-te`版 | 差分 | 採否 |
|---|---|---|---|---|
| OOF AUC | 0.94589 | 0.94604 | +0.00015 | 基準未達 |
| corr vs lgbm | 0.9920 | 0.9931 | 相関↑(非相関性↓) | 逆効果 |
| corr vs xgb | 0.9923 | 0.9933 | 相関↑(非相関性↓) | 逆効果 |
| corr vs catboost | 0.9953 | 0.9963 | 相関↑(非相関性↓) | 逆効果 |
| ensemble hillclimb | 0.94618 | 0.94619 | +0.00001 | ノイズ域 |

**総合判定: 不採用。** 現行の `oof_realmlp.npy`(OOF 0.94589)を維持する。
