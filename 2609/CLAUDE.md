# CLAUDE.md

Kaggle コンペ [Playground Series - Season 6, Episode 9](https://www.kaggle.com/competitions/playground-series-s6e9)
(Predicting Electric Vehicle Purchases、二値分類・評価指標 **ROC-AUC**)のプロジェクトです。2026-09-30 に締切済み。

## 構成

| 置き場所 | 内容 |
|---|---|
| `CLAUDE.md`(このファイル) | このコンペ固有の事実・規約・承認が必要な事項 |
| `.claude/skills/tabular-workflow/` | 工程 ①〜⑦ の進め方と完了条件、最終提出の選び方 |
| `.claude/skills/experiment-gate/` | 実験1本の手順(バックアップ → スクリーニング → フル CV → DeLong 検定 → 後処理 → 記録) |
| `.claude/agents/` | 担当リーダー(管轄とモデル固有の注意) |
| `Log.md`(ローカルのみ) | 実験の経過と数値の根拠。成功も失敗も記録する |

| 担当 | 定義ファイル | 管轄 |
|---|---|---|
| LightGBM / XGBoost / CatBoost / RealMLP Lead | `lgbm-lead` / `xgb-lead` / `catboost-lead` / `realmlp-lead` | `03_feature_engineering_<model>.py` / `04_train_and_evaluate_<model>.py`。CV AUC を競う |
| FE Lead(競争しない) | `fe-lead` | `03_feature_engineering_AllCatalog.py`。モデル間の取りこぼしを横展開 |
| Research Lead(競争しない) | `research-lead` | Kaggle の Code / Discussion から新しい手を持ち込む |

- 担当リーダーは管轄外の `03_*` / `04_*` を編集しない。他モデルの足を引っ張らない
- 審査は単体 AUC に加えて非相関性(アンサンブルへの貢献度)も見る。ただし非相関が効くのは単体スコアが同水準のときだけ

## 指揮官(ユーザー)の承認が必要なこと

- Kaggle への提出(CV の改善に根拠があるものだけ)
- git の push
- 大規模なハイパーパラメータ探索、多数の Feature Engineering パターン × フルデータ 5-fold の総当たり(所要時間の見積もりを添える)
- 採用基準に届かない構成の採用

それ以外(軽量スクリーニング、1 本ずつのフル CV、記録、文書の整合)は Claude 本体が判断して進める。

## データ特性(調査済み・再調査不要)

- train 668,665行 / test 286,571行、**欠損値ゼロ**。`Will_Buy_EV` は Yes 17.5% / No 82.5%
- 数値7列(id除く)・カテゴリ6列。小数は全列で1桁にそろった、合成データ特有の離散構造を持つ
- 値の種類（ユニーク値）の数: 年収 13,214 / 通勤距離 805 が多く、
  Age 45・Charging_Stations_Near_Work 20・Near_Home 15・Environmental_Concern_Level 5・Number_of_Cars_Owned 4 は少ない。カテゴリ列は 2〜4 種類
- 元データ(`data/EV_Adoption_and_Range_Anxiety_Dataset.csv`、1万行、CC0)は RealMLP の特徴量1列だけに使う。
  **ないと黙って列が抜けスコアが一致しない**(取得コマンドは README のセットアップ)
- 較正した確率から推定した「完璧なモデルの AUC」は **0.9459 ± 0.0003**。CV 0.95 は到達不能

## 現状(最終結果)

| 項目 | 値 |
|---|---|
| 最終結果 | **Private 0.94543、376位 / 3,575チーム(上位10.5%)** |
| 最終提出 | CV 0.946310 / Public 0.94644 / Private 0.94543(もう1本は元の構成: CV 0.946234 / Private 0.94539。タグ `final-original-20260927`) |
| 構成 | **LightGBM 1/3 + XGBoost 1/3 + RealMLP 1/3**(順位平均)。CatBoost は重み 0 |

| モデル | ベースライン | 現在 | 列数 | 本番の設定 |
|---|---|---|---|---|
| LightGBM | 0.94123 | **0.94620** | 46 | 重複・不要なエンコーディングを作らない、交互作用1組、交互作用の制約(`--interaction income`) |
| XGBoost | 0.94124 | **0.94609** | 69 | 重複を作らない |
| CatBoost | 0.94156 | 0.94592 | 43 | 重複・不要なエンコーディングを作らない(重み 0) |
| RealMLP | — | **0.94603** | 39 | `--combo-home`(交互作用)、`--te-income`(年収の Target Encoding を train だけで作る) |

- 実行コマンドは README「現行ベストの再現コマンド」、**パラメータの正本は `src/05_hyperparameter_tuning.py` の `FIXED`**
- `notebooks/04_train_and_evaluate.ipynb` は 4 モデルを本番と同じ設定で学習し、OOF と `FIXED` との一致を確認する
- 改善の内訳: Feature Engineering +0.004 / 収束確認 +0.0008 / 列サブサンプリング +0.0002 / RealMLP 追加 +0.0005 / LightGBM の交互作用の制約 +0.00005。
  Private の順は CV の順と一致した(詳しい経緯は Log.md)

## ファイル構成の規約

**ファイル名の先頭の番号が工程の順番**(一覧は README「【ソースコード】」)。

- CV は **StratifiedKFold(n_splits=5, shuffle=True, random_state=42)** で全モデル統一。**変更禁止**
- 目的変数は `(train["Will_Buy_EV"] == "Yes").astype(int)`。`read_csv` までのフローは `02_baseline_*.py` と同一
- `03_<model>` には本番で使う関数だけを置き、作る列は各モデルの `te_plan()` などにまとめる
- 不採用の関数は `03_feature_engineering_AllCatalog.py` の「不採用(記録)」に、名前の末尾にモデル名を付けて移す。
  採否は同じファイルの **`FUNC_STATUS`**(〇24 / ✖33 / 補助4)が持つ。`verify_status()` が `docs/features_*.json` と突き合わせて矛盾を検出する
- `04_*` の `--dump-features` は、本番と同じコードパスで列名を `docs/features_<tag>.json` に書いて終了する。**Feature Engineering を変えたら必ず再生成する**

## 採否基準(paired DeLong 検定)

**AUC を目視で比べず、`src/07_compare_predictions.py` の paired DeLong 検定で判定する。**
fold 分割が共通なので共通のノイズが差し引きで消え、判別できる下限が 0.00015 → 0.00003 になる。

- **採用: 差分 ≥ +0.00008 かつ z ≥ 3**(単体またはアンサンブル。アンサンブルが下がらないこと)
- hill climbing の重みも DeLong 検定で確かめてから採用する(CV +0.000009、z=+1.57 の構成は Public -0.00002 / Private -0.00001 で下がった)

## 打ち止めが確認済みのもの(再検証不要。詳細は Log.md)

| 分類 | 施策 |
|---|---|
| 特徴量 | 四則演算(全パターン) / 交互作用の Target Encoding(2〜13列。採用の1組を除く) / 行フィンガープリント(全行ユニーク) / 元データの行追加 / エンコード方式の変更 / **列削減(`Age` などを落とすと -0.00044。周辺相関が小さいこととモデルに不要なことは別物)** / 補助金との組み合わせ / 年収の近傍統計 / RealMLP の通勤距離÷年齢・通勤距離 /5 のキー / 元データの年収ごとの購入率を GBDT に / Target Encoding を内側の乱数を変えて平均(単体が有意に悪化) / RealMLP に年収 /100 のキーの Target Encoding(アンサンブルが有意に悪化) |
| パラメータ | RealMLP のエポック増 / CatBoost の列サンプリング(対称木のため有害) / `num_leaves`(`max_depth=5` が先に制約。深さ 4・深さ 6 + 葉 63 も誤差) / Hyperparameter Tuning 全般(18 試行) / 元の列ごとに分ける交互作用の制約 / XGBoost の交互作用の制約(LightGBM に入れた構成への上積み +0.000004) |
| モデル追加 | Lookup Transformer(相関 0.982 と最も非相関だが、単体差 0.0019 が埋まらず有意に悪化) |
| アンサンブル | 非有意な構成の採用 / seed 平均 / CatBoost の重みを増やす(1/4 で混ぜると Public 0.94641 で下がった) |
