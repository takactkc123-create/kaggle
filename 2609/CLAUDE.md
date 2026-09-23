# CLAUDE.md

Kaggle コンペ [Playground Series - Season 6, Episode 9](https://www.kaggle.com/competitions/playground-series-s6e9)
(Predicting Electric Vehicle Purchases、二値分類・評価指標 **ROC-AUC**) のプロジェクトです。
実験ログは [Log.md](Log.md) に記録します。

## 最優先事項

- **ROC-AUC の改善を最優先の目的とする。** あらゆる作業判断は「CV AUC が上がるか」を基準に評価する。
- 目標は**リーダーボード上位15%以内**。2026-09-23 時点で **320位 / 2,732チーム(上位11.7%)** と暫定的に圏内
  (15%のラインは 409位)。**参加チームは増え続けるため、同じスコアでも順位は下がる。**
  実際 9/21 266位 → 9/22 307位 → 9/23 320位 と、スコア据え置きのまま54位下がっている。
- ~~当面の目標: CV AUC = 0.95~~ → **撤回**。較正した確率からラベルを再サンプリングして推定した
  「完璧なモデルのAUC」は **0.9459 ± 0.0003**。2位以下が 0.94672 で既に理論上限付近であり到達不能。
- 最終提出はアンサンブルになるため、**単体AUCだけでなく他モデルとの非相関性(特異性)も評価軸**とする。
  ただし **非相関でありさえすれば効く、わけではない**(後述の Lookup Transformer の例)。

## 現状(2026-09-23 時点)

| 項目 | 値 |
|---|---|
| 最良CV(アンサンブル) | **0.94623** |
| 最良 Public LB | **0.94645** |
| 順位(2026-09-23 時点の暫定) | **320位 / 2,732チーム(上位11.7%)** |
| 提出構成 | **LightGBM 1/3 + XGBoost 1/3 + RealMLP 1/3**(rank平均) |
| LB1位 | 0.94945(2位は 0.94672)。**1位だけ 0.0027 離れている**。Public は test の20%のみで算出されるため、この幅は Private で縮む可能性がある |

| モデル | ベースライン | 現在 |
|---|---|---|
| LightGBM | 0.94123 | **0.94610**(アンサンブル採用) |
| XGBoost | 0.94124 | **0.94608**(アンサンブル採用) |
| RealMLP | — | **0.94589**(アンサンブル採用。多様性の供給源) |
| CatBoost | 0.94156 | 0.94589(**重み0**。他GBDTと同質で選ばれない) |
| Lookup Transformer | — | 0.94425(**不採用**。相関0.982と理想的だが入れると有意に悪化) |

**改善の内訳**: FE +0.004 / 収束確認 +0.0008 / 列サブサンプリング +0.0002 / RealMLP追加 +0.0005。
**大半は特徴量エンジニアリング**で、パラメータ側で効いたのは「見落としの是正」2件のみ。

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

## 上位カーネル調査で判明した重要事実(2026-09-12。詳細な調査記録はローカルのみ)

- **元データ(original dataset)の行concatは効かない。** Playground定番の施策だが本コンペでは実測で
  -0.00002。元データは10,000行しかなく668,665行の1.5%増にすぎず、かつ生成器が分布を変えている。
  生成ルール(buy_score)を特徴量にしても -0.00002(importance 74%で1位を占めるのにスコアは動かない)。
- **ノイズ床は 0.00015**(DeLong 導入後は 0.00003)。ただし **max_bin/border_count を動かす検証では
  0.00033 に上がる**ため、その場合は基準を引き上げること(出典の著者自身が、この点を長く見落としていたと注記している)。
- **fold数はLBに効かない**(5/10/15-foldでLB 0.94459/0.94460/0.94461)。CV比較はfold数を固定して行う。
- **交互作用を禁止すると強くなる**という実測報告があり、我々の「交互作用は全滅」という知見と整合する。
- 注意: 一部の上位カーネルは四則演算の交互作用を含むが、効果を切り分けた記載は見当たらない。
  **我々の環境では実測で無効だったため、根拠なく取り込まない。**

## モデルリーダー体制

指揮官(ユーザー・Claude本体=CLAUDE.md)の下に、モデル別の担当リーダー(サブエージェント)を配置する。

| 担当 | 定義ファイル | FE関数群 | 実行スクリプト |
|---|---|---|---|
| LightGBM Lead | `.claude/agents/lgbm-lead.md` | `03_feature_engineering_lgbm.py` | `04_train_and_evaluate_lgbm.py` |
| XGBoost Lead | `.claude/agents/xgb-lead.md` | `03_feature_engineering_xgb.py` | `04_train_and_evaluate_xgb.py` |
| CatBoost Lead | `.claude/agents/catboost-lead.md` | `03_feature_engineering_catboost.py` | `04_train_and_evaluate_catboost.py` |
| RealMLP Lead | `.claude/agents/realmlp-lead.md` | `03_feature_engineering_realmlp.py` | `04_train_and_evaluate_realmlp.py` |
| **FE Lead**(競争しない) | `.claude/agents/fe-lead.md` | `03_feature_engineering_all.py` | モデル間の取りこぼしを横展開 |
| **Research Lead**(競争しない) | `.claude/agents/research-lead.md` | — | Kaggle の Code / Discussion から新しい手を持ち込む |

支援役2名(FE Lead / Research Lead)は自分のスコアを持たず、**他モデルが伸びたかで評価する**。

### 競争ルール

- 各リーダーは**自分のモデルのCV AUCで1位を目指して競う**。
- ただし**他モデルの足を引っ張る行為は禁止**。他モデルのファイル(`03_feature_engineering_*.py` / `04_train_and_evaluate_*.py` で
  自分の管轄外のもの)を編集・削除しない。共有ファイル(`Log.md` / `CLAUDE.md`)は追記のみ。
- 有効だった知見は Log.md を通じて共有してよい(**知見の共有は推奨、実装の横取りは各自の責任で**)。
- **審査は単体AUCだけでなくアンサンブル貢献度(他モデルOOFとの非相関性)も含む。**
  ただし **2026-09-21 の実測で「非相関でさえあれば勝てる」わけではないと判明した**。
  Lookup Transformer は相関 0.982(候補中もっとも非相関)ながら、単体差 0.0019 が埋まらず
  アンサンブルに入れると**有意に悪化**した(w=0.10 で z=-4.5)。
  `diversity beats strength` が成立するのは**単体スコアが同水準のとき**に限られる。

## ファイル構成の規約

**ファイル名の先頭の番号が工程の順番。**

```
01_eda.py                 # ① EDA
02_baseline_<model>.py          # ② ベースライン
03_feature_engineering_<model>.py          # ③ 特徴量エンジニアリング「関数」の定義のみ。実行コードを書かない
04_train_and_evaluate_<model>.py      # ④ ③を import → CV学習 → 評価 → 成果物出力
05_hyperparameter_tuning.py                 # ⑤ HPO(学習コードを持たず ④ を引数違いで呼ぶだけ)
06_ensemble_hill_climbing.py  # ⑥ 貪欲法でブレンド + 採用モデル間の相関を出力
07_compare_predictions.py         # ⑦ paired DeLong 検定(学習しない。保存済みOOFを読むだけ)
feature_catalog.py           # 補助。工程ではないので番号なし
submit/                   # submission_<model>.csv
oof/                      # oof_<model>.npy, pred_<model>.npy(アンサンブル用・必須)
importance/               # importance_<model>.png(feature importance 棒グラフ)
docs/features_<tag>.json  # 各モデルが使っている列名(--dump-features で生成)
```

- `04_train_and_evaluate_<model>.py` に `--dump-features` を付けると、**fold 1 の学習行列を組み上げた直後に
  列名を `docs/features_<tag>.json` へ書いて終了する**(学習しない)。本番と同じコードパスを通るので
  列の取りこぼしがない。**FE を変えたら必ず再生成すること**(コマンドは README 参照)。
  現行の列数は LightGBM 92 / XGBoost 93 / CatBoost 80 / RealMLP 38。

- **`03_feature_engineering_<model>.py` は採用した関数だけを置く場所ではない。** 検証して捨てた施策も
  再検証しないための記録として残す。どれが本番で生きているかは `src/feature_catalog.py` の
  **`FUNC_STATUS`** が持つ(現在 〇28 / ✖28 / 補助8 の計64関数)。
  **FEを採用・不採用にしたらこの表も更新すること。**
  `fd.verify_status()` が `docs/features_*.json` と突き合わせて矛盾を検出する。

- `read_csv` までのフローは `02_baseline_*.py` と同一にする(data/train.csv, data/test.csv)。
- CVは **StratifiedKFold(n_splits=5, shuffle=True, random_state=42)** で全モデル統一。
  fold分割を揃えないとOOF同士のアンサンブル・相関評価ができないため、**変更禁止**。
- 目的変数は `(train["Will_Buy_EV"] == "Yes").astype(int)`。

## 採否基準(2026-09-21 更新: paired DeLong 検定に移行)

**AUC を目視で比べるのをやめ、`src/07_compare_predictions.py` の paired DeLong 検定で判定する。**
fold 分割が全モデル共通(`random_state=42`)なので、2つの予測の差から共通のノイズが差し引きで消え、
判別できる下限が **0.00015 → 0.00003** と約5倍細かくなる。

| | 旧基準 | 現行 |
|---|---|---|
| 判定 | 差分 ≥ +0.0002 | **差分 ≥ +0.00008 かつ z ≥ 3** |

- **hill climbing の出力をそのまま信じないこと。** 貪欲法は OOF 上の偶然を拾う。
  CV +0.000009(z=+1.57、有意でない)の構成を提出したら **LB は -0.00002 と逆に動いた**。
- 採否の結果は**成功も失敗も必ず Log.md に記録する**(重複検証の防止)。

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

## 打ち止めが確認済みのもの(2026-09-21 時点・再検証不要)

| 分類 | 施策 |
|---|---|
| 特徴量 | 四則演算(全パターン) / 交互作用TE(2〜13列) / 行フィンガープリント(全行ユニークで原理的に不可) / 元データ追加 / エンコード方式の変更 |
| パラメータ | `num_leaves`(**`max_depth=5` が先に制約になっていた**) / CatBoost の列サンプリング(対称木のため有害) / RealMLP のエポック増(スケジュール連動) |
| モデル追加 | Lookup Transformer(相関0.982でも有意に悪化) |
| アンサンブル | 非有意な構成の採用(CV +0.000009 → LB -0.00002 で実証) / seed平均 |

**残る選択肢はいずれも期待値が誤差水準**: Optuna(探索対象がHPOと同じ) / 92列の特徴量削減
(列サブサンプリングと重複) / feature importance の再確認。

## ログ運用

- 実験の経過は [Log.md](Log.md) に追記する。**新しいタスクに着手する前に必ず Log.md を読むこと。**
- FEの試行は Log.md の「Feature Engineering 検証結果」表に、**効果あり/なしを分類して**記録する。
- 過去に効果がなかった施策を重複して試さないよう、着手前に必ず表を確認すること。
