# CLAUDE.md

Kaggle コンペ [Playground Series - Season 6, Episode 9](https://www.kaggle.com/competitions/playground-series-s6e9)
(Predicting Electric Vehicle Purchases、二値分類・評価指標 **ROC-AUC**) のプロジェクトです。
実験ログは [Log.md](Log.md) に記録します。

## 最優先事項

- **ROC-AUC の改善を最優先の目的とする。** あらゆる作業判断は「CV AUC が上がるか」を基準に評価する。
- **目標: リーダーボード上位15%以内。**(2026-09-20 時点で 406位 / 2282チーム = 上位17.8%、
  15%のラインは 342位。必要な改善は **+0.0002 程度**)
- ~~当面の目標: CV AUC = 0.95~~ → **2026-09-20 に撤回**。較正した確率からラベルを再サンプリングして
  推定された「完璧なモデルのAUC」は **0.9459 ± 0.0003**(`research.md` 参照)。現LB1位が 0.94675 で
  既に理論上限付近であり、0.95 は原理的に到達不能と判断した。
- 最終提出は3モデルのアンサンブルになるため、**単体AUCだけでなく他モデルとの非相関性(特異性)も評価軸**とする。

## 現状スコア(2026-09-08 ベースライン、数値+カテゴリのみ・デフォルトパラメータ)

| モデル | OOF AUC | Public LB AUC |
|---|---|---|
| LightGBM | 0.94123 | 0.94093 |
| XGBoost | 0.94124 | 0.94152 |
| CatBoost | 0.94156 | 0.94170 |

参考: 当時のLB 1位 = 0.94672(2026-09-20 時点では 0.94675)。

## データ特性(調査済み・再調査不要)

- train 668,665行 / test 286,571行、**欠損値ゼロ**。目的変数 `Will_Buy_EV` は Yes 17.5% / No 82.5%。
- 数値7列(id除く)・カテゴリ6列。カーディナリティは以下の通り:

| 列 | nunique | 種別 |
|---|---|---|
| Age | 45 | 数値(低カーデ) |
| Annual_Income_USD | 13,214 | 数値(高カーデ) |
| Daily_Commute_km | 805 | 数値(中カーデ) |
| Number_of_Cars_Owned | 4 | 数値(低カーデ) |
| Charging_Stations_Near_Home | 15 | 数値(低カーデ) |
| Charging_Stations_Near_Work | 20 | 数値(低カーデ) |
| Environmental_Concern_Level | 5 | 数値(低カーデ) |
| Gender / City_Type / Current_Car_Type | 3 / 3 / 4 | カテゴリ |
| Home_Charging_Possible / Subsidy_Available / Range_Anxiety_Level | 2 / 2 / 3 | カテゴリ |

- 小数は全列で1桁に統一されており、合成データ特有の離散構造を持つ。

## 最重要の戦略ヒント(前コンペ S6E8 の教訓)

- **HPO/Optuna より先に特徴量エンジニアリングとアンサンブルを追求すること。**
  S6E8 では四則演算・ビン分割・閾値フラグ系のFEはほぼ無効(±0.0001)だった一方、
  **「数値列も含めた全列を厳密な値のまま Target Encoding」** した施策のみが CV AUC +0.006 以上という
  桁違いの改善を生んだ。Playground の合成データはラベルが特徴量の厳密値に紐づく確率から
  サンプリングされているため、この構造が再現する可能性が高い。
- 本コンペでも低カーディナリティ列が多く、**厳密値TE・カテゴリ交互作用TE・Count Encoding** が
  最有力候補。四則演算FEは「やるべきだが期待値は低い」という前提で臨むこと。
- TEは必ず **fold内でfitしてvalidationに適用**(リーク厳禁)。OOF AUCが不自然に跳ねたらリークを疑う。

## 上位カーネル調査で判明した重要事実(2026-09-12、`reference_URL.md` に詳細)

- **元データ(original dataset)の行concatは効かない。** Playground定番の施策だが本コンペでは実測で
  -0.00002。元データは10,000行しかなく668,665行の1.5%増にすぎず、かつ生成器が分布を変えている。
  生成ルール(buy_score)を特徴量にしても -0.00002(importance 74%で1位を占めるのにスコアは動かない)。
- **ノイズ床は 0.00015。** ただし **max_bin/border_count を引き上げると 0.00033 に上がる**ため、
  その検証時は採否基準を ±0.0004 に引き上げること(調査元著者が1ヶ月間誤判定していた落とし穴)。
- **我々の採否基準 ±0.0002 は妥当**と裏付けられた(過去4エピソードの実測で、0.0002以上の差は99.3%が
  privateでも順位保持。0.0001未満は50.3% = 情報ゼロ)。
- **fold数はLBに効かない**(5/10/15-foldでLB 0.94459/0.94460/0.94461)。CV比較はfold数を固定して行う。
- **交互作用を禁止すると強くなる**という実測報告があり、我々の「交互作用は全滅」という知見と整合する。
- 警告: 一部の上位カーネルは四則演算の交互作用を入れているが**アブレーションしていない**。真似しない。

## モデルリーダー体制

指揮官(ユーザー・Claude本体=CLAUDE.md)の下に、モデル別の担当リーダー(サブエージェント)を配置する。

| リーダー | 定義ファイル | FE関数群 | 実行スクリプト |
|---|---|---|---|
| LightGBM Lead | `.claude/agents/lgbm-lead.md` | `03_fe_lgbm.py` | `04_fe_run_lgbm.py` |
| XGBoost Lead | `.claude/agents/xgb-lead.md` | `03_fe_xgb.py` | `04_fe_run_xgb.py` |
| CatBoost Lead | `.claude/agents/catboost-lead.md` | `03_fe_catboost.py` | `04_fe_run_catboost.py` |

### 競争ルール

- 各リーダーは**自分のモデルのCV AUCで1位を目指して競う**。
- ただし**他モデルの足を引っ張る行為は禁止**。他モデルのファイル(`03_fe_*.py` / `04_fe_run_*.py` で
  自分の管轄外のもの)を編集・削除しない。共有ファイル(`Log.md` / `CLAUDE.md`)は追記のみ。
- 有効だった知見は Log.md を通じて共有してよい(**知見の共有は推奨、実装の横取りは各自の責任で**)。
- **審査は単体AUCだけでなくアンサンブル貢献度(他モデルOOFとの非相関性)も含む。**
  弱くても非相関なら勝てる(diversity beats strength)。

## ファイル構成の規約

```
03_fe_<model>.py          # 特徴量エンジニアリング「関数」の定義のみ。実行コードを書かない
04_fe_run_<model>.py      # 03_fe_<model>.py を import → CV学習 → 評価 → 成果物出力
submit/                   # submission_<model>.csv
oof/                      # oof_<model>.npy, pred_<model>.npy(アンサンブル用・必須)
importance/               # importance_<model>.png(feature importance 棒グラフ)
```

- `read_csv` までのフローは `02_bl_*.py` と同一にする(data/train.csv, data/test.csv)。
- CVは **StratifiedKFold(n_splits=5, shuffle=True, random_state=42)** で全モデル統一。
  fold分割を揃えないとOOF同士のアンサンブル・相関評価ができないため、**変更禁止**。
- 目的変数は `(train["Will_Buy_EV"] == "Yes").astype(int)`。

## FE の採否基準

- **ベースライン OOF AUC を上回れば採用、下回る/横ばいなら不採用。**
- 判定の誤差水準は ±0.0002 程度。これ未満の差は「効果なし」と判定すること。
- 採否の結果は**成功も失敗も必ず Log.md の Feature Engineering 表に記録する**(重複検証の防止)。

## 作業前の確認・警告ルール

- 以下を実行する**前に**所要時間を見積もり、指揮官に報告・確認すること。無断実行しない。
  - 大規模なハイパーパラメータ探索(Optuna等の多数トライアル)
  - 多数のFEパターン × フルデータ5-fold の総当たり
- **まず高速スクリーニング(サブサンプル or fold数削減 or n_estimators削減)で方向性を掴み、
  有望なものだけフルCVで確認する**という段階設計を徹底する。
- CatBoost はフル5-foldで10分以上かかる。スクリーニング時は必ず軽量設定を使うこと。

## Kaggle CLI・API 利用時の注意

- submission は**1日の残数を意識し、CV改善の根拠があるもののみ**提出する。
- **提出は指揮官の承認を得てから行うこと**(各リーダーは勝手に submit しない)。
- 同一データの再ダウンロードを繰り返さない(`data/` を優先利用)。

## ログ運用

- 実験の経過は [Log.md](Log.md) に追記する。**新しいタスクに着手する前に必ず Log.md を読むこと。**
- FEの試行は Log.md の「Feature Engineering 検証結果」表に、**効果あり/なしを分類して**記録する。
- 過去に効果がなかった施策を重複して試さないよう、着手前に必ず表を確認すること。
