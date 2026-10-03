# Kaggle Playground Series S6E9 — Predicting Electric Vehicle Purchases

## コンペ概要

EV(電気自動車)を購入するか(`Will_Buy_EV`: Yes/No)を予測する二値分類コンペ。
評価指標は **ROC-AUC(0~1)**。 ※ 1 に近いほど精度がよく、順位が高い。

- コンペ: https://www.kaggle.com/competitions/playground-series-s6e9
- 開催: Kaggle 主催の Playground Series
- 期間: 2026-09-01〜09-30 の 1 か月
- 参加: 3,575チーム
- データ: 顧客の属性・通勤距離・充電環境・補助金の有無など 13 列。train 668,665 行 / test 286,571 行。

## 結果

**最終結果: Private LB 0.94543、376位 / 3,575チーム(上位10.5%)**

| 構成 | CV | Public LB | Private LB |
|---|---|---|---|
| LightGBM・XGBoost・RealMLP の順位平均(LightGBM に交互作用の制約) | **0.946310** | 0.94644 | **0.94543** |

Private LB は test の 80%、Public LB は残り 20% で採点される。Private の順は CV の順と一致した(「[提出ごとのスコアの推移](#提出ごとのスコアの推移)」)。

### スコアアップに向けて取り組んだこと(ベースライン → アンサンブル)

| # | 工程 | 取り組み | CV |
|---|---|---|---|
| 1 | ② Baseline | カテゴリ列を落とさずにモデルへ渡す(7 列 → 13 列。+0.057) | 0.94156(CatBoost) |
| 2 | ③ Feature Engineering | 厳密値 Target Encoding(+0.003)、Out-of-Fold によるリーク対策、Count Encoding | — |
| 3 | ④ Train and Evaluate | 収束の確認(学習率を下げ、early stopping で木の本数を決める。+0.0008) | 0.94555 |
| 4 | ③ Feature Engineering | 平滑化3種の Target Encoding + Smooth Keys + digit + ビン数 1024、CatBoost の catify | 0.94604 |
| 5 | ⑥ Ensemble | 異種モデル RealMLP の追加(+0.0005)、列サブサンプリング(+0.0002)、3 モデルの順位平均 | 0.946234 |
| 6 | ⑤ Hyperparameter Tuning | 18 試行すべて誤差か悪化で、採用ゼロ | — |
| 7 | ③ Feature Engineering | 重複する列の整理、交互作用 1 組、RealMLP に年収の Target Encoding | 0.946264 |
| 8 | ④ Train and Evaluate | Feature Importance から、LightGBM に交互作用の制約(+0.000047) | **0.946310** |

CV は、その時点で提出した構成の OOF AUC(1 は CatBoost 単体、3 以降はアンサンブル。下の「提出ごとのスコアの推移」と同じ値)。各施策の採否は ⑦ の paired DeLong 検定で判断した。

## プロセス全体像

![パイプライン全体像](docs/pipeline_overview.png)

```
① EDA → ② Baseline → ③ Feature Engineering → ④ Train and Evaluate → ⑥ Ensemble → submit
                        (関数の定義)            (学習・OOF の保存)  ↑        (oof/ を読む)
                                                      │ 引数を変えて呼ぶ
                                            ⑤ Hyperparameter Tuning
                          ⑦ Paired DeLong Test … ③〜⑥ の変更を採用するかを判定
```

### ディレクトリ構成

【ignore】は `.gitignore` の対象(リポジトリに含めない)。データと実行成果物は再現できるため、記録類はローカルで管理するため除外している。
詳細な実験ログ(`Log.md`)や外部調査の記録も、ローカルだけで管理している。

```
kaggle/                                  # リポジトリのルート(コンペごとのフォルダ)
└── 2609/                                # このコンペ
    ├── README.md
    ├── CLAUDE.md                        # Claude Code への方針・規約
    ├── pyproject.toml / uv.lock / .python-version   # 環境(ライブラリの版を固定)
    ├── src/                             # 本番パイプライン
    │   ├── 01_eda.py
    │   ├── 02_baseline_<model>.py
    │   ├── 03_feature_engineering_<model>.py   # 本番で使う特徴量の関数
    │   ├── 03_feature_engineering_AllCatalog.py  # 採否表・列名の JSON 出力・不採用の関数の記録
    │   ├── 04_train_and_evaluate_<model>.py
    │   ├── 05_hyperparameter_tuning.py
    │   ├── 06_ensemble_hill_climbing.py
    │   ├── 07_compare_predictions.py
    │   └── feature_importance.py               # Feature Importance(特徴量の重要度)の確認
    ├── notebooks/                       # 工程を追える説明版(01〜06)
    ├── docs/
    │   ├── pipeline_overview.png        # プロセス全体像の図
    │   ├── eda_results.md               # EDA の結果
    │   ├── fe_results_<model>.md / fe_results_all.md   # 各担当の検証記録
    │   ├── features_<model>.json        # 本番で使っている列名(--dump-features で生成)
    │   ├── hyperparameter_tuning_results.csv   # ⑤ の試行ごとの記録(05 が書き、05 の --report が読む)
    │   └── reference_URL.md 【ignore】   # 参考カーネルの調査結果
    ├── .claude/agents/                  # サブエージェントの定義
    │   └── lookup-transformer-lead.md 【ignore】
    ├── .claude/skills/                  # 進め方と実験の手順(コンペに依存しない)
    ├── data/ 【ignore】                  # train.csv / test.csv / sample_submission.csv / 元データ
    ├── datacheck/                       # EDA の図(01_eda.py が出力)
    ├── oof/ 【ignore】                   # oof_<model>.npy / pred_<model>.npy(アンサンブル用)
    ├── submit/ 【ignore】                # submission_<model>.csv
    ├── importance/ 【ignore】            # feature importance の棒グラフと値(CSV)
    ├── experiments_rejected/ 【ignore】  # 不採用の実験成果物(記録として保持)
    ├── backup_*/ 【ignore】              # バックアップ
    ├── catboost_info/ 【ignore】         # CatBoost が学習時に書き出すログ
    ├── kernels/ 【ignore】               # 取得した公開カーネル
    ├── kaggle_kernel_*/ 【ignore】       # Kaggle Notebook で動かしたモデル
    ├── tools/ 【ignore】                 # 図の作成などの補助スクリプト
    ├── .venv/ 【ignore】                 # uv が作る仮想環境
    └── Log.md / nippo.md / research.md / prompt.md / main.py / fe_results_lookup.md 【ignore】   # 実験ログなどの記録
```

---

## ソースコード

**ファイル名の先頭の番号が工程の順番**。`src/` が本番、`notebooks/` は同じ番号の工程を上から追える説明版
(03・04 は `src/` と結果が一致することをノートブックの中で確認している)。

| # | 工程 | 本番(`src/`) | 説明版(`notebooks/`) | 出力 |
|---|---|---|---|---|
| ① | EDA(探索的データ分析) | `01_eda.py` | `01_eda.ipynb` | `datacheck/*.png` |
| ② | Baseline(ベースライン) | `02_baseline_<model>.py` | `02_baseline.ipynb` | `submit/submission_baseline_<model>.csv` |
| ③ | Feature Engineering(特徴量エンジニアリング) | `03_feature_engineering_<model>.py` | `03_feature_engineering.ipynb` | (関数のみ。実行しない) |
| ④ | Train and Evaluate(学習と評価) | `04_train_and_evaluate_<model>.py` | `04_train_and_evaluate.ipynb` | `oof/` `submit/` `importance/` |
| ⑤ | Hyperparameter Tuning(ハイパーパラメータ調整) | `05_hyperparameter_tuning.py` | `05_hyperparameter_tuning.ipynb` | `docs/hyperparameter_tuning_results.csv` |
| ⑥ | Ensemble(アンサンブル) | `06_ensemble_hill_climbing.py` | `06_ensemble.ipynb` | `submit/submission_hillclimb.csv` |
| ⑦ | Paired DeLong Test(対応のある DeLong 検定) | `07_compare_predictions.py` | (`06_ensemble.ipynb` に含む) | 採否判定 |
| — | 補助 | `03_feature_engineering_AllCatalog.py` | `03_feature_engineering.ipynb` の 03-7 章 | 関数の採否表(`FUNC_STATUS`)、列名の JSON 出力、不採用の関数の記録。学習には関わらない |
| — | 補助 | `feature_importance.py` | `04_train_and_evaluate.ipynb` の 04-7 章 | Feature Importance(特徴量の重要度)の上位の列。本番の成果物には書き込まない |
| — | Claude Code | `.claude/agents/*.md` / `.claude/skills/*/SKILL.md` | — | `docs/fe_results_*.md` |

| モデル | ② Baseline | ④ 最終構成 | 列数 |
|---|---|---|---|
| LightGBM | 0.94123 | **0.94620** | 46(交互作用の制約あり) |
| XGBoost | 0.94124 | **0.94609** | 69 |
| CatBoost | 0.94156 | 0.94592 | 43(アンサンブルの重みは 0) |
| RealMLP | — | 0.94603 | 39 |
| **アンサンブル**(LightGBM・XGBoost・RealMLP の順位平均) | — | **0.946310** | — |

> **GBDT**(Gradient Boosting Decision Tree:勾配ブースティング決定木)は、決定木を少しずつ足して誤りを補正していくモデルの総称。このリポジトリでは **LightGBM・XGBoost・CatBoost の3つのモデル**を指す。

## Notebooks

工程を上から読んで追えるようにしたもの。リポジトリのルートから起動しても、`notebooks/` から
起動しても動く(先頭セルで作業ディレクトリを揃えている)。

`src/` と同じ番号体系。

| Notebook | 対応する工程 | 内容 |
|---|---|---|
| `notebooks/01_eda.ipynb` | ① | データの素性、値の種類（ユニーク値）の数、値ごとの購入率 |
| `notebooks/02_baseline.ipynb` | ② | 3モデルのベースライン(共通の CV ループ) |
| `notebooks/03_feature_engineering.ipynb` | ③ | read_csv から特徴量の作成までを **GBDT 共通 → GBDT モデル別 → RealMLP** の順にノートブック内で実行し、意図と根拠を説明。本番の .py と値まで一致することも確認。不採用にした施策の一覧も載せる |
| `notebooks/04_train_and_evaluate.ipynb` | ④ | 4モデルを本番と同じ特徴量・設定で上から順に学習し、本番の OOF と一致することを確認(約75分)。04-7 章で Feature Importance を横向き棒グラフで確認 |
| `notebooks/05_hyperparameter_tuning.ipynb` | ⑤ | Hyperparameter Tuning の設計と所要時間の見積もり + 実行した 18 試行の結果(すべて誤差か悪化) |
| `notebooks/06_ensemble.ipynb` | ⑥⑦ | ブレンドの再現。相関の確認と DeLong 検定による採否判定まで |

Jupyter で開く際は、カーネルに **`.venv` の Python** を選ぶこと。

**ノートブックは `src/` を読み込んで動く。クローン・持ち出しの際は `src/` も必ず含めること。**
04・06 などは特徴量を作る関数やモデル本体を `src/` から import しており、`notebooks/` だけでは動かない。
関数をノートブックへ移さない理由は `notebooks/04_train_and_evaluate.ipynb` の冒頭に記載している
(要点: 本番の正本を1か所に保つため。Target Encoding は fold ごとに作り直すので、列名だけ渡しても `src/` の関数は必要になる)。

## ① EDA(Exploratory Data Analysis:探索的データ分析)

モデルを作る前にデータの分布・欠損・目的変数との関係を調べ、どんな特徴量が効きそうかの仮説を立てる工程。

```bash
uv run src/01_eda.py
```

`data/` の CSV を読み込み、`datacheck/` に図を保存する。図の構成と順番は `notebooks/01_eda.ipynb` と同じ(ファイル名先頭の連番が表示順)。

| ファイル | ノートブックの節 | 内容 |
|---|---|---|
| [`01_列の一覧.png`](datacheck/01_%E5%88%97%E3%81%AE%E4%B8%80%E8%A6%A7.png) | 01-1-1 | 列名・日本語名・型・欠損・値の種類の数 |
| [`02_数値列の分布と購入率.png`](datacheck/02_%E6%95%B0%E5%80%A4%E5%88%97%E3%81%AE%E5%88%86%E5%B8%83%E3%81%A8%E8%B3%BC%E5%85%A5%E7%8E%87.png) | 01-1-3 | 数値7列の件数(Yes/No 積み上げ)と、値ごとの購入率 |
| [`03_カテゴリ列の件数と購入率.png`](datacheck/03_%E3%82%AB%E3%83%86%E3%82%B4%E3%83%AA%E5%88%97%E3%81%AE%E4%BB%B6%E6%95%B0%E3%81%A8%E8%B3%BC%E5%85%A5%E7%8E%87.png) | 01-1-4 | カテゴリ6列の件数と、値ごとの購入率 |
| [`04_trainとtestの分布比較.png`](datacheck/04_train%E3%81%A8test%E3%81%AE%E5%88%86%E5%B8%83%E6%AF%94%E8%BC%83.png) | 01-1-8 | train と test の分布の比較(全13列でほぼ一致) |
| [`05_相関ヒートマップ.png`](datacheck/05_%E7%9B%B8%E9%96%A2%E3%83%92%E3%83%BC%E3%83%88%E3%83%9E%E3%83%83%E3%83%97.png) | 01-2-1 | Spearman 相関(数値列 + 目的変数)と Cramér's V(全列) |
| [`06_充電環境のクラスタ.png`](datacheck/06_%E5%85%85%E9%9B%BB%E7%92%B0%E5%A2%83%E3%81%AE%E3%82%AF%E3%83%A9%E3%82%B9%E3%82%BF.png) | 01-2-2 | 関係の強い列の組(スタンド数・居住地・自宅充電・航続距離への不安) |
| [`07_自宅充電と自宅スタンド数の交互作用.png`](datacheck/07_%E8%87%AA%E5%AE%85%E5%85%85%E9%9B%BB%E3%81%A8%E8%87%AA%E5%AE%85%E3%82%B9%E3%82%BF%E3%83%B3%E3%83%89%E6%95%B0%E3%81%AE%E4%BA%A4%E4%BA%92%E4%BD%9C%E7%94%A8.png) | 01-2-2 | 自宅充電の可否ごとの、スタンド数と購入の log-odds |
| [`08_補助金ありの割合.png`](datacheck/08_%E8%A3%9C%E5%8A%A9%E9%87%91%E3%81%82%E3%82%8A%E3%81%AE%E5%89%B2%E5%90%88.png) | 01-2-3 | 各説明変数の値ごとの「補助金ありの割合」 |
| [`09_補助金の効き方.png`](datacheck/09_%E8%A3%9C%E5%8A%A9%E9%87%91%E3%81%AE%E5%8A%B9%E3%81%8D%E6%96%B9.png) | 01-2-3 | 補助金あり/なし別の購入率と、補助金の効果(log-odds 差) |
| [`10_収入と年齢・環境意識の組み合わせ.png`](datacheck/10_%E5%8F%8E%E5%85%A5%E3%81%A8%E5%B9%B4%E9%BD%A2%E3%83%BB%E7%92%B0%E5%A2%83%E6%84%8F%E8%AD%98%E3%81%AE%E7%B5%84%E3%81%BF%E5%90%88%E3%82%8F%E3%81%9B.png) | 01-2-4 | 収入の5分位ごとの log-odds(年齢帯・環境意識レベル別) |

**主な情報**: train 668,665行 / test 286,571行、欠損なし、購入率 17.5%。
`Environmental_Concern_Level` の効きが圧倒的(レベル1で約1% → レベル5で約52%)。
**年収は 13,214 種類の値**を持ち、木の既定ビン数(255)では値ごとの違いが潰れる
→ 後の厳密値 Target Encoding とビン数引き上げにつながる最重要の所見。

## ② Baseline(ベースライン:基準となるモデル)

特徴量を作らず元の列だけで学習し、以降の改善を測る基準のスコアを作る工程。CV の分け方もここで全モデル共通に固定する。

```bash
uv run src/02_baseline_lgbm.py
uv run src/02_baseline_xgb.py
uv run src/02_baseline_catboost.py
```

**数値列のみ(7列)** と、**数値列 + カテゴリ列(13列)** の2通りで学習する
(カテゴリ列はエンコードせず、各ライブラリのネイティブなカテゴリ対応に渡す)。
CV は全モデル共通で `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`。
**この分割を全モデルで揃えることが、後のアンサンブルと DeLong 検定の前提になる。**

| モデル | OOF AUC<br>数値列のみ(7列) | OOF AUC<br>数値列 + カテゴリ列(13列) | Public LB<br>(13列) | Private LB<br>(13列) | Private の順位(上位) |
|---|---|---|---|---|---|
| LightGBM | 0.88380 | 0.94123 | 0.94093 | 0.94040 | 2985〜2990位相当(83.5〜83.6%) |
| XGBoost | 0.88362 | 0.94124 | 0.94152 | 0.94092 | 2533〜2550位相当(70.9〜71.3%) |
| CatBoost | 0.88398 | 0.94156 | 0.94170 | 0.94106 | 2292〜2309位相当(64.1〜64.6%) |

**カテゴリ列を加えるだけで、3モデルとも約 +0.057。** `Subsidy_Available` と `Range_Anxiety_Level` は
数値列より購入率の差がはるかに大きく、落とすとこの情報をまるごと失う。以降は13列を出発点にする。

> 順位は最終の Private LB(3,575チーム)で、そのスコアが何位に相当するかを示したもの。同点は提出時刻の順に並ぶため、同点帯の幅で示す。

## ③ Feature Engineering(特徴量エンジニアリング)

元の列から、モデルが学びやすい形の新しい列(特徴量)を作る工程。ここでは作り方を関数として定義するだけで、実行はしない。

> **RealMLP について**: RealMLP は数値を埋め込みに変えて学ぶニューラルネットワーク。GBDT と予測の間違え方が違うため、アンサンブルの多様性を補う目的で③から導入。前処理と設定が複雑で、ベースラインには含めない。

| モデル | ③ Feature Engineering の関数 |
|---|---|
| LightGBM | `src/03_feature_engineering_lgbm.py` |
| XGBoost | `src/03_feature_engineering_xgb.py` |
| CatBoost | `src/03_feature_engineering_catboost.py` |
| RealMLP | `src/03_feature_engineering_realmlp.py` |
| (横断) | `src/03_feature_engineering_AllCatalog.py`(全モデルの関数の採否表と、不採用の関数の記録) |

**効いたもの**
- **厳密値 Target Encoding**(各モデル +0.003 前後、最大の改善要因)。数値列もビン分割せず値のままキーにする
- **Out-of-Fold Target Encoding によるリーク対策**(+0.00108)。学習行には内側CVのOOF値を当てる
- Count Encoding(LightGBM/XGBoost のみ。CatBoost では無効)
- Triple Target Encoding(平滑化3種)+ Smooth Keys + digit features + ビン数1024(+0.0005〜0.001)
- catify(値の種類（ユニーク値）が少ない数値のカテゴリ化)は **CatBoost 固有**(+0.0017)
- 列の整理(±0。スコアは変わらないが列が減り、読みやすくなる): 他の列と同じ情報しか持たない列を作らない(GBDT 3種)、値の種類が少ない数値列のエンコーディングを Target Encoding だけに絞る(LightGBM・CatBoost)。どの列を作るかは各モデルの `te_plan()` にまとめてある
- 交互作用 1 組「自宅充電の可否 × 自宅スタンド数」(LightGBM・RealMLP。+0.00001 で有意差はないが、CV 最高のため最終構成 D に採用)
- RealMLP の年収の Target Encoding(train だけで作る。`--te-income`)。元データの購入率と並べて単体 +0.000128 / アンサンブル +0.000018

**効かなかったもの**: 四則演算、交互作用の Target Encoding(2〜13列。上の1組を除く)、行フィンガープリント、元データの行の追加、補助金との組み合わせ、元データの年収ごとの購入率を GBDT に足す、Target Encoding を内側の乱数を変えて数回作り平均する。
各施策の詳細(なぜ試したか / 期待した効果 / 結果の考察)は `docs/fe_results_*.md` を参照。

> `src/03_feature_engineering_<model>.py` は**関数の定義のみ**、`src/04_train_and_evaluate_<model>.py` が**実行**という分担。
> この分離により、同じ関数を別の検証スクリプトからも再利用できる。

## ④ Train and Evaluate(学習と評価)

③の関数で特徴量を作り、交差検証(CV)でモデルを学習・評価して、予測と提出ファイルを保存する工程。

| モデル | ④ 学習・評価の実行スクリプト |
|---|---|
| LightGBM | `src/04_train_and_evaluate_lgbm.py` |
| XGBoost | `src/04_train_and_evaluate_xgb.py` |
| CatBoost | `src/04_train_and_evaluate_catboost.py` |
| RealMLP | `src/04_train_and_evaluate_realmlp.py` |

### 現行ベストの再現コマンド

```bash
uv run src/04_train_and_evaluate_lgbm.py \
  --max_bin 1024 --feature_fraction 0.3 --max_depth 5 \
  --folds 5 --learning_rate 0.03 --n_estimators 8000 --early_stopping 200 --n_jobs 7 --interaction income --save --tag lgbm

uv run src/04_train_and_evaluate_xgb.py --max-bin 1024 \
  --set-param colsample_bytree=0.3 --set-param max_depth=5 \
  --folds 5 --learning-rate 0.03 --n-estimators 8000 --early-stopping 200 --n-jobs 7 --save --out-suffix ""

uv run src/04_train_and_evaluate_catboost.py \
  --folds 5 --iters 1000 --lr 0.06 --fast --border 64 --hc-border 1024 --threads 7 --save

uv run src/04_train_and_evaluate_realmlp.py --folds 5 --threads 7 --combo-home --te-income --tag realmlp   # epochs は 2 から変えないこと
```

| モデル | OOF AUC | 備考 |
|---|---|---|
| **LightGBM** | **0.94620** | アンサンブル採用。年収系の列とそれ以外の列を、同じ木の枝で組み合わせない(交互作用の制約) |
| **XGBoost** | **0.94609** | アンサンブル採用 |
| **RealMLP** | **0.94603** | アンサンブル採用。GBDTとの相関が低く多様性を供給 |
| CatBoost | 0.94592 | 現在アンサンブルの重みは 0 |

> **重いジョブは1つずつ実行すること**(8コア環境)。CatBoost と RealMLP を並列で走らせると
> CPU を取り合って完走しない(実際に CatBoost が 0.33 コアまで押し出された)。

### 使っている特徴量の列名を確認する

`--dump-features` を付けると、**fold 1 の学習行列を組み上げた直後に列名を `docs/features_<tag>.json` へ書いて終了する**(学習しない)。本番と同じコードパスを通るので列の取りこぼしがない。`notebooks/03_feature_engineering.ipynb` は、ノートブックで組み上げた列がこの JSON と一致するかを確かめる。

```bash
uv run src/04_train_and_evaluate_lgbm.py --dump-features --tag lgbm
uv run src/04_train_and_evaluate_xgb.py --dump-features --out-suffix ""
uv run src/04_train_and_evaluate_catboost.py --fast --dump-features --tag catboost
uv run src/04_train_and_evaluate_realmlp.py --combo-home --te-income --dump-features --tag realmlp
```

**Feature Engineering を変えたら再生成すること。** JSON は生成物だが `data/` を含めていないためクローン先では作り直せない。ノートブックの表示元になるのでリポジトリに含めている。

### Feature Importance(特徴量の重要度)を確認する

モデルが予測を作るときに、どの列をどれだけ使ったかを確かめる。本番と同じ特徴量・パラメータで fold 1 だけを学習し直し(本番の fold 1 と同じモデルになる)、重要度(gain の割合)の上位の列を print する。**本番の予測・提出ファイルには書き込まない。**

```bash
uv run src/feature_importance.py                  # LightGBM・XGBoost・CatBoost の上位 10 列(約 7 分)
uv run src/feature_importance.py lgbm --top 20    # モデルと表示する列数を指定
```

図と、読み取れること(重要度は相関ではない)は `notebooks/04_train_and_evaluate.ipynb` の 7 章。RealMLP は木のモデルのような重要度を持たないので対象外。

## ⑤ Hyperparameter Tuning(ハイパーパラメータ調整)

学習率や木の深さなど、学習前に決めておく設定値(ハイパーパラメータ)を変えて、より良い組み合わせを探す工程。

**列サブサンプリングが効いた**(2026-09-20、外部カーネル調査で見つけた見落とし)。
92列の Target Encoding 特徴量に対し全列を使うと、どの木も最強列(年収の Target Encoding)を根に選ぶため木が似通う。

| モデル | 設定 | 効果 |
|---|---|---|
| LightGBM | `feature_fraction=0.3, max_depth=5` | **+0.000223**(z=+10.4) |
| XGBoost | `colsample_bytree=0.3, max_depth=5` | **+0.000163**(z=+8.5) |
| CatBoost | — | 列サンプリングは**有害**(-0.00015)。対称木のため木全体が一斉に弱くなる |

**木の本数を増やした理由**: 列を間引いて木を弱くすると、収束までに要る本数が増える。本数を据え置くと、効く施策でも効かないと判定してしまうため、early stopping で本数を取り直した。

**交互作用の制約が効いた**(2026-09-29、Feature Importance から立てた仮説)。重要度の 7〜9 割を補助金と環境意識が占めるので、年収系の列(年収・その Smooth Keys・digit・Count・Target Encoding)とそれ以外の列を、同じ木の枝で組み合わせないようにした。年収の細部が別の枝で学ばれる。

| モデル | 単体の差 | アンサンブルの差 | 判定 |
|---|---|---|---|
| LightGBM | **+0.000091**(z=+4.21) | **+0.000047**(z=+6.50)。5 fold すべてで改善 | **採用** |
| XGBoost | +0.000089(z=+4.20) | LightGBM に入れた構成への上積みは +0.000004(z=+0.52)。提出しても Public・Private とも同点(0.94644 / 0.94543) | **不採用**(LightGBM と予測が似通い、多様性が減る) |

### 探索のフレーム

```bash
uv run src/05_hyperparameter_tuning.py lgbm --estimate          # 試行一覧と所要時間の見積もりだけ表示
uv run src/05_hyperparameter_tuning.py lgbm                     # 実行(1本ずつ順番に)
uv run src/05_hyperparameter_tuning.py lgbm --only num_leaves   # 特定の軸だけ
uv run src/05_hyperparameter_tuning.py --report                 # これまでの結果を表示
```

`src/05_hyperparameter_tuning.py` は**学習コードを持たない**。既存の `04_train_and_evaluate_<model>.py` を引数違いで
呼ぶだけにして、収束設定や Feature Engineering の構成が本番とズレないようにしている。解説は
`notebooks/05_hyperparameter_tuning.ipynb`。

**本番のパラメータの正本は `src/05_hyperparameter_tuning.py` の `FIXED`**(本番コマンドのうち探索しない固定部分)。
05 の結果が 04 に自動で反映される仕組みはなく、`notebooks/04_train_and_evaluate.ipynb` は確定した値を書き写したうえで、
04-6 章で `FIXED` と同じかを照合している(ずれていれば止まる)。④の再現コマンドも `FIXED` と同じ値にそろえること。

**探索の結果: 18 試行と LightGBM の深さ 2 本のすべてが誤差か悪化で、採用ゼロ。**

| モデル | 試した軸 | 結果 |
|---|---|---|
| LightGBM | `num_leaves`(15 / 63 / 127)、深さ(4、6 + 葉 63) | すべて誤差。`max_depth=5`(葉は最大 32)が先に効いている |
| XGBoost | `max_depth`、`colsample_bytree`、`min_child_weight`、`subsample` | 深さ 6・7 は有意に悪化、ほかは誤差 |
| CatBoost | `depth`、`one_hot_max_size` | 深さは既定の 6 が最適、`one_hot_max_size` は無反応 |

## ⑥ Ensemble(アンサンブル)

複数のモデルの予測を組み合わせ、1つのモデルより安定して高いスコアを出す工程。どのモデルをどの比率で混ぜるかを決める。

```bash
uv run src/06_ensemble_hill_climbing.py
```

`oof/oof_<name>.npy` と `oof/pred_<name>.npy` の組をすべて候補として読み込み、rank 正規化した予測を
hill climbing で足し合わせる。選ばれた回数がそのまま重みになる。

**採用モデル間の順位相関**も併せて出力する。相関が高いほど同質で、足しても伸びない
(弱くても非相関なら勝てる)。未採用の候補のうち最も非相関なものも提示するので、
多様性が枯渇したときにどれを足せばよいかが分かる。

**最終提出**: LightGBM 1/3 / XGBoost 1/3 / RealMLP 1/3 の順位平均(CatBoost は重み 0)。CV 0.946310 / Private 0.94543。各提出のスコアは「[提出ごとのスコアの推移](#提出ごとのスコアの推移)」。

### hill climbing の重みの検証(DeLong 検定)

貪欲法は OOF 上の偶然の当たりも拾うため、重みは ⑦ の DeLong 検定で有意性を確かめてから採用した。
有意でない構成(CV +0.000009、z=+1.57)を提出すると、Public -0.00002 / Private -0.00001 で下がった。

### 提出

```bash
kaggle competitions submit -c playground-series-s6e9 -f submit/submission_hillclimb.csv -m "<説明>"
```

`06_ensemble_hill_climbing.py` は実行のたびに `submit/submission_hillclimb.csv` を上書きし、`oof/` の予測をすべて候補にする。そのため、不採用の予測は `experiments_rejected/` に移して候補から外している。

## ⑦ Paired DeLong Test(対応のある DeLong 検定)

2つの予測の AUC の差が偶然か本物かを統計的に判定し、変更を採用するかを決める工程。学習はせず、保存済みの予測を読むだけ。

> **DeLong 検定とは**: DeLong 検定は、同じデータに対する2つの予測の AUC の差が、偶然の揺れで説明できる程度かを調べる統計的な検定。差が偶然でないと言えるときだけ変更を採用するために行う。

```bash
uv run src/07_compare_predictions.py lgbm lgbm_shallow      # 2つを比較
uv run src/07_compare_predictions.py --all realmlp          # 全候補と比較
```

全モデルが fold 分割を `random_state=42` で固定して
いるため、2つの予測は**同じ行・同じ分割**で作られている。DeLong 検定はこの対応を使って差の標準誤差を
直接求めるので、**両者に共通のノイズが差し引きで消え**、AUC を別々に眺めるより桁違いに細かく差を
見分けられる。

**算出式**: 正例の予測を $X_1,\dots,X_m$、負例の予測を $Y_1,\dots,Y_n$ とする。

```math
\hat\theta=\frac{1}{mn}\sum_{i=1}^{m}\sum_{j=1}^{n}\psi(X_i,Y_j),\qquad \psi(x,y)=\begin{cases}1 & x>y \\ 1/2 & x=y \\ 0 & x<y\end{cases}
```

各行が AUC にどれだけ寄与したか(構造成分)を、モデルごとに求める。

```math
V_{10}(X_i)=\frac{1}{n}\sum_{j=1}^{n}\psi(X_i,Y_j),\qquad V_{01}(Y_j)=\frac{1}{m}\sum_{i=1}^{m}\psi(X_i,Y_j)
```

2 つのモデル A・B の構造成分の共分散 $S_{10}$・$S_{01}$(2×2 の行列)から、AUC の共分散と差の z 値を出す。

```math
S=\frac{S_{10}}{m}+\frac{S_{01}}{n},\qquad z=\frac{\hat\theta_B-\hat\theta_A}{\sqrt{S_{AA}+S_{BB}-2S_{AB}}}
```

同じ行で作った予測は $S_{AB}$(共分散)が大きいので、差の標準誤差が小さくなる。実装は `src/07_compare_predictions.py` の `delong_cov` / `paired_test`。

| | 従来 | DeLong |
|---|---|---|
| ノイズ床(SE) | 0.00015 | **0.00003 前後** |
| 採否基準 | 差分 ≥ +0.0002 | 差分 ≥ +0.00008 **かつ** z ≥ 3 |

導入の効果は早速出た。XGBoost の列サブサンプリング(+0.000163)は**従来基準なら捨てていた**数字だが、
z=+8.48 で誤差でないことが確定した。

---

## 提出ごとのスコアの推移

| 提出 | CV | Public LB | Private LB | Private の順位(上位) |
|---|---|---|---|---|
| CatBoost 単体(ベースライン) | 0.94156 | 0.94170 | 0.94106 | 2292〜2309位相当(64.1〜64.6%) |
| Feature Engineering + 収束確認 | 0.94555 | 0.94576 | 0.94479 | 1378〜1380位相当(38.5〜38.6%) |
| Triple Target Encoding + digit | 0.94604 | 0.94622 | 0.94529 | 939〜969位相当(26.3〜27.1%) |
| 列サブサンプリング + 3モデル | 0.946234 | **0.94645** | 0.94539 | 611〜669位相当(17.1〜18.7%) |
| 列の整理 + 交互作用 + RealMLP の年収 Target Encoding | 0.946264 | 0.94642 | 0.94540 | 572〜611位相当(16.0〜17.1%) |
| **上 + LightGBM の交互作用の制約(最終提出)** | **0.946310** | **0.94644** | **0.94543** | **376位(10.5%。最終順位)** |
| (参考)最終提出 + CatBoost(4 モデル各 1/4) | 0.946294 | 0.94641 | 0.94543 | (最終順位と同点) |
| (参考)最終提出から元データを抜く(締切後の提出) | 0.946288 | 0.94642 | 0.94541 | 529〜572位相当(14.8〜16.0%) |

> Private は CV の改善にそって上がり続けた。Public は 0.94645 で頭打ちになり、09-20 以降の改善を拾えていなかった。
> 「相当」は Private LB(3,575チーム)に照らした位置。同点は提出時刻の順に並ぶため、同点帯の幅で示す。

## この取り組みで効いたこと

1. **厳密値 Target Encoding**(+0.003)— 前コンペの教訓がそのまま再現した最大の要因
2. **収束の確認**(+0.0008)— Feature Engineering で特徴量を13→39列に増やしたのに木の本数がデフォルト100のままだった
3. **列サブサンプリング**(+0.0002)— 外部調査で見つけた見落とし。引数を足すだけ
4. **異種モデル(RealMLP)の追加** — GBDT同士は相関0.99で同質化しており、多様性の供給源になった
5. **paired DeLong 検定** — 従来なら誤差として捨てていた改善を拾えるようになった
6. **交互作用の制約**(+0.000047)— Feature Importance で「補助金と環境意識が大半を占める」と分かり、年収の細部を別の枝で学ばせた

**効かなかったこと**: 四則演算・交互作用など「人間が意味を考えて作った特徴量」は全滅だった。
合成データの生成過程にそうした関係がなかったため。
## Claude Code の構成

| 置き場所 | 内容 |
|---|---|
| [CLAUDE.md](CLAUDE.md) | このコンペ固有の事実・規約・採否基準、指揮官(ユーザー)の承認が必要な事項 |
| `.claude/skills/tabular-workflow/` | 表形式データのコンペの進め方(工程 ①〜⑦ の完了条件、最終提出の選び方)。コンペに依存しない |
| `.claude/skills/experiment-gate/` | 実験1本の手順(バックアップ → スクリーニング → フル CV → DeLong 検定 → 後処理 → 記録)。コンペに依存しない |
| `.claude/agents/` | 担当リーダー(サブエージェント)の管轄と、モデル固有の注意 |

| エージェント | 役割 | 記録 |
|---|---|---|
| `lgbm-lead` / `xgb-lead` / `catboost-lead` / `realmlp-lead` | 各モデルの CV AUC 向上を**競う** | `docs/fe_results_<model>.md` |
| `fe-lead` | Feature Engineering の統括。**競争せず**、モデル間の取りこぼしを横展開する | `docs/fe_results_all.md` |
| `research-lead` | 情報収集。**競争せず**、Kaggle の Code / Discussion から新しい手を持ち込む | (ローカル管理) |

- モデル担当の審査は単体 AUC だけでなく、**他モデルとの非相関性**(アンサンブルへの貢献度)も見る
- 他モデルのファイルは編集しない。Kaggle への提出と push は指揮官の承認後のみ

## 環境

- Python 3.14 / uv で管理(導入は下の「セットアップ」)
- 主要ライブラリ: lightgbm, xgboost, catboost, torch(CPU版), scikit-learn, pandas, matplotlib, seaborn
- GPU は無し。RealMLP は CPU 学習で 1エポック約290秒(7スレッド)

## セットアップ(クローンから再現まで)

動作確認環境: Windows 11 / CPU 8 コア / GPU なし。Python 3.14 と各ライブラリの版は `uv.lock` で固定している。

```bash
git clone https://github.com/takactkc123-create/kaggle.git
cd kaggle/2609
uv sync

# データの取得(Kaggle API の認証ファイル kaggle.json と、コンペの参加規約への同意が必要)
uv run kaggle competitions download -c playground-series-s6e9 -p data --unzip
uv run kaggle datasets download -d itzzomkar/ev-adoption-behavior-and-range-anxiety -p data --unzip
```

- 2 つ目は**元データ**(`data/EV_Adoption_and_Range_Anxiety_Dataset.csv`、1 万行)。コンペのデータはこれを元に Kaggle が生成したもので、
  コンペの Data ページで使用が認められている(CC0)。
- 元データは学習には使わない。RealMLP の特徴量 1 列(年収ごとの元データでの購入率)を作るためだけに使う。
  **元データなしでも上位15%のライン(536位)付近。** RealMLP の元データの列を、train だけで作る年収の購入率に置き換えると Private 0.94541 で 529〜572位相当(上位14.8〜16.0%。締切後の提出で確認)。本番(元データあり)は 0.94543 / 376位
  **ないとエラーにならずにこの列が抜け(39 列 → 38 列)、スコアが本番と一致しない**ので、必ず取得すること。
- 取得後は「[現行ベストの再現コマンド](#現行ベストの再現コマンド)」で 4 モデルを学習(約 75 分)→ `uv run src/06_ensemble_hill_climbing.py` でアンサンブル。
- ノートブックは、カーネルに `.venv` の Python を選べば動く。
