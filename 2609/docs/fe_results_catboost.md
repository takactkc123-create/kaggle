# CatBoost FE 検証結果 (S6E9)

担当: CatBoost Lead / ベースライン `baseline_catboost.py` OOF AUC = **0.94156**(デフォルトパラメータ・5-fold)

実装:
- `03_feature_engineering_catboost.py` — FE関数の定義のみ
- `04_train_and_evaluate_catboost.py` — `--fe` に FEコンポーネントをカンマ区切りで渡して実行
  (例: `uv run src/04_train_and_evaluate_catboost.py --fe te_all,catify --folds 5 --save`)

## 検証環境の制約(重要)

本セッションでは他2エージェント(LightGBM / XGBoost)が同一マシン(8コア)で同時実行しており、
CatBoost の学習速度が大幅に低下した(実測: 10万行 / 200 iterations の単一 fit に約55秒、
200 iterations を超える設定ではフル5-foldで1時間以上)。そのため以下の段階設計を採用した。

- **スクリーニング条件**: train 12万行サブサンプル / 3-fold / `iterations=200` / `learning_rate=0.3` /
  `boosting_type=Plain` + `max_ctr_complexity=1` + `border_count=64`(`--fast`)/ `thread_count=3`
- この条件での **base(FEなし) = 0.93790 を基準に相対比較**する。
  絶対値はフル設定のベースライン 0.94156 とは直接比較できない(軽量設定による低下分 ≈ -0.0037)。
- 判定は「base 比 +0.0002 以上で有望」とし、有望な構成のみフルデータ5-foldで確認した。

## FEコンポーネント一覧(`--fe` の値)

| 名前 | 内容 |
|---|---|
| `base` | 数値7列 + カテゴリ6列(ベースライン相当) |
| `arith` | 四則演算14種(充電スタンド sum/diff/avg/ratio、収入÷通勤距離/年齢/車数、通勤×年齢 ほか) |
| `inter` | カテゴリ交互作用キー10組を文字列結合し cat_features として投入 |
| `cnt` | 13列の Count Encoding(train+test で fit、教師なしのためリークなし) |
| `cnt_ix` | 交互作用キーの Count Encoding |
| `te_cat` | カテゴリ6列の Target Encoding |
| `te_low` | 低カーデ数値5列の Target Encoding |
| `te_all` | **13列全部(数値含む)の厳密値 Target Encoding**(S6E8ブレークスルー施策) |
| `te_ix` | 交互作用キー10組の Target Encoding |
| `catify` | **低カーデ数値5列(Age / Number_of_Cars_Owned / Charging_Stations_*×2 / Environmental_Concern_Level)を文字列化して cat_features に渡す**(CatBoost固有) |
| `digits` | **digit features**。数値7列 × k=-4..3 の `(x // 10**k) % 10` を int8 列で追加。定数列(全行同値)を自動削除して **16列**(56列中40列はこのデータでは常に0)。浮動小数の丸め誤差で下位桁にゴミが入らないよう、値を 1e4 倍した int64 上で桁を取り出している |
| `skeys` | Multi-Scale Smooth Keys。`floor(income)` / `floor(income/100)` / `floor(income/1000)` / `floor(commute)` の4本。**モデルには入れず TEのキーとしてのみ使う**(helper列) |
| `te3` | **Triple Target Encoding**。同一キーに smooth = 10 / 20 / 100 の3系統を別列として同時投入。sum/count の集計は1回だけ行い3つの平滑度で共有するため、追加コストはほぼゼロ |

TE はすべて **fold内 fit** を厳守:
- valid / test → その foldの学習部分のみで fit した写像を適用
- 学習部分自体 → **inner 5-fold の out-of-fold 値**を使用(自分のラベルを絶対に見ない)
- スムージング `(sum + prior*m) / (count + m)`、m=20
- リーク検証: ランダムラベルに対して train/valid ともに TE単体AUC ≈ 0.50 を確認済み(リークなし)

## 効果あり(採用)

| 施策 | ベースライン | 適用後 OOF AUC | 改善幅 | 備考 |
|---|---|---|---|---|
| `te_all,catify`(**採用した最終構成**) | 0.93790 (軽量base) | 0.93956 | **+0.00166** | 厳密値TE + 低カーデ数値のカテゴリ化。スクリーニング最良 |
| `catify`(低カーデ数値→cat_features) | 0.93790 | 0.93883 | +0.00093 | CatBoost固有レバー。単独でも有効 |
| `te_all`(13列 厳密値TE) | 0.93790 | 0.93898 | +0.00108 | S6E8の知見が本コンペでも再現。**内部Ordered TSとの二重適用による悪化は起きなかった** |
| `te_all,cnt,te_ix` | 0.93790 | 0.93905 | +0.00115 | te_all単独とほぼ同等。cnt/te_ixの上積みはなく、`te_all,catify` に劣る |
| `te_cat`(カテゴリ6列のみTE) | 0.93790 | 0.93810 | +0.00020 | 判定閾値ぎりぎり。te_all に完全に内包されるため単独採用せず |

## 最終構成(フルデータ・5-fold)

```
uv run src/04_train_and_evaluate_catboost.py --fe te_all,catify --folds 5 --iters 400 --lr 0.15 --fast --save
```

| 項目 | 値 |
|---|---|
| ベースライン OOF AUC | 0.94156 |
| **最終 OOF AUC** | **0.94472** |
| **改善幅** | **+0.00316** |
| fold別 AUC | 0.94386 / 0.94442 / 0.94565 / 0.94503 / 0.94469 |
| 特徴量数 | 26(生13列 + 厳密値TE13列) |
| 学習時間 | 1647秒(27分、他エージェントとCPU競合下) |

- CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` 厳守(全モデル共通)
- モデルパラメータ: `boosting_type=Plain`, `max_ctr_complexity=1`, `border_count=64`,
  `iterations=400`, `learning_rate=0.15`(**実行時間制約による短縮設定**。デフォルトの
  1000 iterations / 低learning_rate で回せばさらに上積みが期待できる ← 最優先のTODO)
- `cat_features` = カテゴリ6列 + 低カーデ数値5列(catify)= 11列

### 成果物

- `submit/submission_catboost.csv`
- `oof/oof_catboost.npy`, `oof/pred_catboost.npy`
- `importance/importance_catboost.png`

### Feature Importance の所見(`importance/importance_catboost.png`)

上位5特徴は `Subsidy_Available` > `te_Environmental_Concern_Level` > `te_Subsidy_Available` >
`te_Annual_Income_USD` > `Environmental_Concern_Level`。**上位5つのうち3つがTE特徴**であり、
特に `te_Environmental_Concern_Level`(5値)と `te_Annual_Income_USD`(13,214値)は
生の列を大きく上回った。LightGBM担当の「高カーデ数値列の厳密値TEが主因」という所見と整合するが、
CatBoostでは**低カーデ列のTEも同等以上に効いている**点が異なる。

### アンサンブル貢献度

| 組み合わせ | AUC |
|---|---|
| CatBoost 単体 | 0.94472 |
| LightGBM 単体 | 0.94509 |
| CatBoost + LightGBM(rank平均 50/50) | **0.94524** |

OOFのランク相関は 0.9903。単体では LightGBM にわずかに劣るが、**ブレンドで両者を上回る**ため
アンサンブル要員としての価値は確認できた。


## 効果なし(不採用)

| 施策 | ベースライン | 適用後 OOF AUC | 差分 | 備考 |
|---|---|---|---|---|
| `arith`(四則演算14種) | 0.93790 | 0.93691 | **-0.00099** | 明確に悪化。S6E8同様、四則演算FEは無効と確認。特徴量増によるノイズ |
| `cnt`(13列 Count Encoding) | 0.93790 | 0.93773 | -0.00017 | 横ばい。合成データのため count に情報が乗っていない |
| `te_ix`(交互作用キーのTE) | 0.93790 | 0.93782 | -0.00008 | 横ばい。CatBoost が内部で ctr 組み合わせを作るため重複と推測 |
| `cnt` / `te_ix` の te_all への上積み | 0.93898 (te_all) | 0.93905 | +0.00007 | 誤差範囲。特徴量49列に増えるコストに見合わず不採用 |
| `drop_num`(TE後に生の低カーデ数値5列を落とす) | 0.93956 (te_all,catify) | 0.93895 | **-0.00061** | 生の列とTE列は非冗長。両方残すべき。指揮官指示により追加検証 |
| `cnt` の `te_all,catify` への上積み | 0.93956 (te_all,catify) | 0.93958 | +0.00002 | **LightGBMではTE+Countが加算的に効いた(+0.0008)が、CatBoostでは完全に無効。** 内部CTRが件数情報を既に取り込んでいるためと推測。指揮官からの知見共有を受けて追加検証 |

## CatBoost固有の論点

- **外部TEあり/なしの比較(必須確認事項)**: `base` 0.93790 → `te_all` 0.93898。
  CatBoost内部の Ordered Target Statistics と外部TEの**二重適用は悪化させず、むしろ改善**した。
  内部CTRはカテゴリ列にしか効かないため、**数値列の厳密値TEが新しい情報を足している**と解釈できる。
- **catify が効く理由**: Age(45種)や Charging_Stations_*(15/20種)は数値として分割するより
  カテゴリとして CTR を効かせた方が良い。`te_all` と併用すると相補的に効く(+0.00166)。
- `one_hot_max_size` の調整は、実行時間の制約により未検証(TODO)。

---

# セッション2: digit features + border_count 引き上げ / Triple TE

出典: `reference_URL.md` S-1 / S-2 / S-3。開始時点の自分の最良は **OOF 0.94493**(`te_all,catify`、本格学習スケジュール)。

## このセッションの最大の制約: CPU の 9重競合

同一8コアマシン上で LightGBM×3 / XGBoost×2 / RealMLP×1 の他エージェントジョブが同時稼働しており、
`Win32_Process` で実測したところ **CatBoost プロセスが確保できた CPU は約0.3コア**だった
(`--threads 3` を指定しても、実効 0.33 コア相当)。

その結果:

- フル5-fold(668,665行 / 42特徴 / border 254 / iterations 500)の所要は **約7,800 CPU秒 ≒ 実時間2時間以上**と
  実測から推定され、**本セッションの時間枠では完走不可能**と判断して打ち切った。
- したがって**成果物(`oof/`, `submit/`, `importance/`)は更新していない**。
  0.94493 の既存成果物をそのまま維持してある(退行なし)。
- 判定は **100,000行サブサンプル / 3-fold / iterations=250 / lr=0.3 / subsample=0.6 / 1スレッド**の
  軽量スクリーニングで行った。**fold分割は全構成で同一**(`--rows` は seed=42 固定、
  `StratifiedKFold(3, shuffle=True, random_state=42)`)なので、fold単位の比較も有効。

## 検証結果(軽量スクリーニング)

| 構成 | `--fe` | border | fold0 | fold1 | fold2 | OOF |
|---|---|---|---|---|---|---|
| S_ref(参照) | `te_all,catify` | 64 | 0.93884 | 0.93851 | 0.93765 | **0.93826** |
| S_dig | `te_all,catify,digits` | 254 | 0.93938 | 0.93812 | 0.93774 | **0.93835**(+0.00009) |
| S_te3 | `te_all,catify,digits,skeys,te3` | 254 | 0.93990 | 0.93948 | 0.93852 | **0.93918(+0.00092)** |
| S_te3only | `te_all,catify,skeys,te3` | 64 | 0.94101 | 0.93945 | 0.93670 | **0.93892(+0.00066)** |

fold単位の差分(同一fold同士の対比なので有効。`--rows` seed / fold seed とも42固定で全構成同一分割):

| fold | S_ref | S_dig − S_ref | S_te3only − S_ref | S_te3 − S_ref |
|---|---|---|---|---|
| 0 | 0.93884 | +0.00054 | +0.00217 | +0.00106 |
| 1 | 0.93851 | −0.00039 | +0.00094 | +0.00097 |
| 2 | 0.93765 | +0.00009 | −0.00095 | +0.00087 |
| **OOF** | 0.93826 | **+0.00009(判定不能)** | **+0.00066(採用)** | **+0.00092(採用・最良)** |

特徴量数: S_ref 26 / S_dig 42 / S_te3only 64 / S_te3 80。

## 判定と解釈

**採否基準は ±0.0004**(`reference_URL.md` §5。ビン数を上げるとノイズ床が 0.00015 → 0.00033 に上がるため)。

### 1. digit features + border_count 254 → **この検証スケールでは +0.00009(判定不能)**

fold単位では +0.00054 / -0.00039 / +0.00009 とばらつき、合計 +0.00009。基準 ±0.0004 に対して**判定不能**。
ただし**「効かない」と結論するのは誤り**で、以下の理由からこのスクリーニングは本施策を**構造的に過小評価する**:

- digit features と border 引き上げが狙うのは「`Annual_Income_USD` の 13,214種の値レベルの構造」。
  gm_noise が実証した通りこの構造は**実信号**だが、その解像度を学習するには**値あたりの行数**が要る。
- フルデータでは 668,665 / 13,214 = **1値あたり約50行**。本スクリーニングの 100,000行では
  **1値あたり約7.5行**しかなく、値レベルの構造そのものが統計的に存在しない。
- つまり **サブサンプル検証では原理的に digit/border の効果が出ない**。
  gm_noise(+0.00201)も evgendvorkin(+0.00143)も**フルデータでの測定値**である。
- **結論: フルデータ5-foldで測り直すまで判定保留。実装は完了済みなので、CPUが空き次第まず回すべき。**

### 2. Triple TE + Smooth Keys → **同スケールでも fold0 で +0.00106 と明確に反応**

TE はキーあたりの行数が減っても平滑化が効くため、サブサンプルでも効果が観測できる。
参照比で **+0.00106 / +0.00097 / +0.00087(3fold全て正、OOF +0.00092)**。
採否基準 ±0.0004 の 2.3倍で、かつ fold間のばらつきが小さい。**明確に採用**。
`digits+border254` 比でも +0.00052 / +0.00136 / +0.00078(OOF +0.00083)と上積みしている。
**3施策の中で本セッション唯一、軽量検証でも明確に反応した施策**。

### 3. 切り分け: `skeys,te3` 単独(digits/border なし)で +0.00066

`digits` と `border 254` を外した S_te3only でも **+0.00066**(基準の1.7倍)。
すなわち **本セッションの改善の大半(72%)は Triple TE + Smooth Keys 由来**であり、
`digits+border254` はその上に **+0.00026** を足しているにすぎない(単独では +0.00009)。

ただし fold単位のばらつきが大きく(S_te3only は fold0 +0.00217 / fold2 −0.00095)、
**OOF値で見れば digits を足した S_te3 が最良**なので、フル学習では両方入れた構成で測る。
上記1の通り digits/border はフルデータでこそ本領を発揮するため、なおさら外すべきでない。

## 実行コスト削減のために追加したオプション

フル5-fold が回せなかった原因は **border_count を上げると全連続列のビンが増える**ことにある。
本当に高解像度が要るのは `Annual_Income_USD`(13,214種)と `Daily_Commute_km`(805種)、
およびその TE 列だけなので、**その列にだけ border を割り当てる**オプションを追加した。

```
--hc-border N   # per_float_feature_quantization で高カーデ数値列とその TE 列のみ border_count=N
```

- `--border 64 --hc-border 1024` は `--border 254` を全列に掛けるより**総ビン数がはるかに少なく**、
  gm_noise が問題視した「income の 13,214値が潰れる」点だけをピンポイントで解消できる。
- 動作確認済み(`--fe te_all,catify,digits --border 64 --hc-border 1024` で正常終了、42特徴)。

その他、混雑時用に `--rsm`(列サブサンプル)と `--subsample`(Bernoulli 行サブサンプル)も追加した。

## CPUが空いたときに回すべきコマンド(最優先TODO)

```
uv run src/04_train_and_evaluate_catboost.py --fe te_all,catify,digits,skeys,te3 \
  --folds 5 --iters 1000 --lr 0.06 --fast --border 64 --hc-border 1024 \
  --threads 6 --save
```

- 比較対象は現行成果物の **OOF 0.94493**。
- **採否基準は ±0.0004**(reference_URL.md §5: ビン数を上げるとノイズ床が 0.00015 → 0.00033 に上がるため)。
- 所要見積り: 空いた8コアマシンで 60〜90分。他エージェントと同時実行では回らない。

---

# セッション3: Smooth Key 重複除去 + Triple TE への `auto` 追加 (2026-09-18)

背景: FE Lead の横断コードレビューにより、`SMOOTH_KEY_SPECS` の `sk_inc_1 = floor(Annual_Income_USD)`
は **`Annual_Income_USD` が全行整数のため厳密値TEキーと完全に同一の重複列**であると判明
(実測 `(inc == floor(inc)).all() == True`)。加えて `TRIPLE_SMOOTHS = [10, 20, 100]` に
LightGBM/XGBoost が使う `"auto"`(sklearn TargetEncoder の経験ベイズ則)が欠けていることも判明。
指揮官の指示で以下を実装し検証した。

## 実施した変更(検証後にすべて revert 済み)

- `03_feature_engineering_catboost.py`: `SMOOTH_KEY_SPECS` の `sk_inc_1` → `sk_inc_10` に置換、`sk_inc_10000` を追加
- `03_feature_engineering_catboost.py`: `_smooth_tag` を文字列 `"auto"` に対応(`isinstance(smooth, str)` 分岐を追加)
- `03_feature_engineering_catboost.py`: `_te_map_from_agg` を新設し、`smooth="auto"` のとき
  `m_i = p_i(1-p_i) / (p(1-p))`(sklearn `TargetEncoder(smooth="auto")` と同一の経験ベイズ則、
  fe_lgbm.py の実装と同型)を計算するようにした。数値 `smooth` の場合は従来と同じ
  `(sum + prior*m)/(count + m)` で挙動不変(純粋なリファクタ)
- `04_train_and_evaluate_catboost.py`: `TRIPLE_SMOOTHS = ["auto", 10.0, 100.0]`
  (旧: `[10.0, 20.0, 100.0]`)
- `04_train_and_evaluate_catboost.py`: `--hc-border` のターゲット判定タプルに `"auto"` を追加

コード自体は `uv run src/04_train_and_evaluate_catboost.py --fe te_all,catify,digits,skeys,te3 --folds 2 --iters 100
--lr 0.2 --fast --border 32 --hc-border 64 --rows 50000` のスモークテストで例外なく動作することを確認済み
(te_cols=18、正常終了)。

## 検証結果(FE Lead のハーネスによる fold1 実測。本エージェントのフル5-foldは
Windows環境の `uv` 呼び出し起因で `exit code 127` により未完走)

| 構成 | n_feat | fold1 AUC | 差分 |
|---|---|---|---|
| A = 現行相当(`sk_inc_1` 重複あり / smooth 10,20,100) | 93 | 0.94511 | — |
| B = 提案(重複除去 + `/10`,`/10000` 追加 / smooth `auto`,10,100) | 96 | 0.94516 | **+0.00005** |

## 判定: **効果なし(不採用)**

**+0.00005 は採否基準 ±0.0004 を大きく下回る**。単一foldでの測定のためノイズ幅はさらに広く、
判定材料として十分。指揮官の指示により作業を中止し、**コードは `backup_20260918/` の状態に
完全 revert 済み**(`diff -q` で `03_feature_engineering_catboost.py` / `04_train_and_evaluate_catboost.py` ともにバックアップと
バイト同一であることを確認)。成果物(`oof/`, `submit/`, `importance/`)も更新していない
(mtime 変化なし、`oof_catboost.npy` は revert 前と `np.array_equal` で一致確認済み)。

### 解釈

`sk_inc_1` が厳密値TEキーとビット同一であることは数学的に確実だったため、B は
「無駄列3本を削って解像度2つ(`/10`, `/10000`)と `auto` 系統を足した」構成のはずだったが、
それでも AUC は動かなかった。これは以下のいずれか/両方を意味する:

- 重複列(`sk_inc_1`)は CatBoost にとって実害が無かった(木がその列を選ばなければ実質コストゼロ、
  なので削除しても得もしない)
- Smooth Key の追加解像度(`/10`, `/10000`)と `auto` 系統は、既存の Triple TE(smooth 10/20/100 ×
  income/commute の厳密値・複数スケール)の情報をすでに焼き直しているに過ぎない

**結論: 本コンペの CatBoost TE 周りは既に絞り尽くされている**と見るのが妥当。
今後 Smooth Key の解像度追加や smooth 方式の追加バリエーションを重複検証しないこと。

## 未検証(時間切れ・TODO)

- **最優先: 上記「CPUが空いたときに回すべきコマンド」のフル5-fold**(成果物更新はここで行う)
- `skeys` 単独 と `te3` 単独 のさらなる切り分け(今回は2つセットでしか測っていない)
- Triple TE の平滑度の選び方。現在 10 / 20 / 100。Markus 方式(smooth=100 は Smooth Keys 4本のみ)は未試行
- `digits` と `border` の分離測定(本セッションはセットでしか測っていない)
- `--border 254` 全列 と `--border 64 --hc-border 1024` の比較
- `one_hot_max_size` の調整(5 / 10 / 20)
- `te_low` 単独、`drop_num`(TE後に生の低カーデ数値を落とす)
- TEスムージング係数 m の調整(現在 20 固定)
- 3列以上の高次交互作用
