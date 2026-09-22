# FE 統括レポート(FE Lead)

S6E9 の4モデル(LightGBM / XGBoost / CatBoost / RealMLP)に散らばっていた FE を
`03_fe_all.py` に集約し、**「あるモデルで有効だが別のモデルに未適用」** の取りこぼしを
洗い出して検証した記録。

- 集約カタログ: [03_fe_all.py](../src/03_fe_all.py)
- 検証ハーネス: `tools/crosstest_gbdt.py`(GBDT 3種)/ `tools/crosstest_realmlp.py`
  (検証用スクリプトのためリポジトリには含めていない)
- 既存の `03_fe_<model>.py` / `04_fe_run_<model>.py` は**一切変更していない**。

---

## 1. モデル × FE 適用状況マトリクス(2026-09-18 時点・コード実読による)

凡例: ✅適用済 / ❌無効と実測済 / ⬜**未適用かつ未検証(=横展開候補)** / △部分適用

| FE | LightGBM | XGBoost | CatBoost | RealMLP |
|---|---|---|---|---|
| 厳密値 TE(13列) | ✅ | ✅ | ✅ | ⬜ **未適用** |
| Triple TE(smooth 3系統同時) | ✅ auto/10/100 | ✅ auto/10/100 | △ **10/20/100(`auto` 欠落)** | ⬜ 未適用 |
| 入れ子 TE(学習行 inner-OOF) | ✅ | ✅ (+0.00108) | ✅ | △ sklearn TargetEncoder(cv=5) |
| Smooth Keys(income 粗解像度) | △ /10,/100,/1000 | △ /100,/1000,/10000 | △ **/1(重複),/100,/1000** | △ cat特徴のみ(TEキーでない) |
| Smooth Key を Count のキーにも使う | ⬜(`skcnt` 未使用) | ⬜(`smooth_keys_ce` 未使用) | ⬜ | ⬜ |
| Count / Frequency Encoding | ✅ (+0.00083) | ✅ (+0.00049) | ❌ (-0.00017) | △ **1列のみ・train のみで fit** |
| digit features | ✅ float実装 | ✅ float実装 | ✅ 整数実装 | △ `is_multiple_10` / `decimal` のみ |
| catify(低カーデ数値→カテゴリ) | ⬜ **未検証** | ⬜ **未検証** | ✅ (+0.00170) | ✅ 相当(`{col}_cat_` factorize) |
| ビン数引き上げ | ✅ max_bin=1024 | ✅ max_bin=1024 | ✅ hc-border=1024 | n/a |
| 交互作用禁止(interaction_constraints) | ⬜ 未検証 | ⬜ 未検証 | ✅ 相当(`max_ctr_complexity=1`) | n/a |
| 四則演算 | ❌ | ❌ | ❌ (-0.00099) | △ `commute/age` 1本のみ |
| 交互作用 TE | ❌ | ❌ | ❌ | △ 2組のみ(income×anxiety, age×anxiety) |
| org_mean(元データ target mean) | ⬜ | ❌報告あり | ⬜ | ✅ |
| KBins(quantile 多解像度) | ⬜ | ⬜ | ⬜ | ✅ (400/600/800/900/1100) |

### 最大の発見

**RealMLP は「厳密値 Target Encoding」を持っていない。**
Log.md Run 12 には「全モデルに Triple TE を適用」とあるが、コードを読むと
`04_fe_run_realmlp.py` の TE は `combo_names`(= income×RangeAnxiety, age×RangeAnxiety の
**2列だけ**)に `sklearn.TargetEncoder(smooth="auto")` を当てているにすぎない。
13列の厳密値 TE も Smooth Keys の TE も入っていない。
GBDT で最大の改善要因(各 +0.003 前後)だった施策が、4モデル中1モデルだけ抜けている。

---

## 2. 同一施策の実装差分(重要)

### 2-1. Smooth Keys の除数が3モデルで全部違う

| | income の除数 | commute |
|---|---|---|
| fe_lgbm | 10 / 100 / 1000 | floor(km) |
| fe_xgb | 100 / 1000 / **10000** | floor(km) |
| fe_catboost | **1** / 100 / 1000 | floor(km) |
| fe_realmlp | 100 / 1000 / 10000(cat特徴) | floor(km/5)(cat特徴) |

`Annual_Income_USD` は整数値なので **`floor(income) == income`**。
つまり **CatBoost の `sk_inc_1` は厳密値 TE キーとビット同一の重複列**で、
TE 列が1本まるまる無駄になっている。さらに CatBoost には `/10` も `/10000` も無い。
`03_fe_xgb.py` のコメントはこの重複に気づいて意図的に除外している(実装間で知見が共有されていない)。

### 2-2. Triple TE の smooth 集合が CatBoost だけ違う

- LGBM / XGB: `("auto", 10, 100)`
- CatBoost: `[10.0, 20.0, 100.0]`(`04_fe_run_catboost.py:44 TRIPLE_SMOOTHS`)

`auto` は sklearn TargetEncoder 相当の経験ベイズ則(`m_i = p_i(1-p_i) / p(1-p)`)で、
純粋なキーをほぼ無平滑、50/50 のキーを強く縮める **固定 m とは質的に別物**。
CatBoost の 10/20/100 は固定 m を3つ並べているだけで、実質「粗さ違いの同じもの」に近い。

### 2-3. digit features の実装が float 版と整数版に分かれている

- `fe_lgbm` / `fe_xgb`: `floor(x * 10**-k) % 10`(float 演算)
- `fe_catboost`: `rint(x * 10000)` で整数化してから整数除算

本データは小数1桁なので実害は出ていないと思われるが、**整数版の方が安全**。
`fe_all.add_digit_features` は整数版を採用。

### 2-4. 入れ子 TE と単純 fold 内 TE

XGB リーダーの実測で **入れ子(学習行に inner-OOF を当てる)方が +0.00108 高い**。
リーク防止ではなく「学習行の楽観バイアス除去」が理由。
現在は GBDT 3種とも入れ子。RealMLP は sklearn TargetEncoder(cv=5) で実質同等。

### 2-5. Count Encoding の fit 範囲

- GBDT 3種: train+test 結合で fit(教師なしなのでリークなし)
- RealMLP: **train だけ**で fit → test の頻度情報を捨てている(しかも 1 列のみ)

---

## 3. 横展開テストの結果

### 検証条件

すべて `tools/crosstest_gbdt.py`(GBDT)/ `tools/crosstest_realmlp.py`(NN)による**同一ハーネス内 A/B**。
軽量スクリーニングは **fold 1-2 のみ**(fold 分割は共通の StratifiedKFold(5,42) なので、
既存 OOF から計算した各モデルの fold 別 AUC と直接比較できる)。

**ハーネスの忠実度検証**: LGBM ベースライン(lr 0.06 / 1200本)の fold1-2 平均 = **0.94520** に対し、
公式 `oof_lgbm_tte_dig.npy` の fold1-2 平均 = 0.94528。差 -0.00008 は lr 引き上げ分で説明でき、
**ハーネスは公式パイプラインを忠実に再現している**と判断した。

参考: 各モデルの公式 OOF から計算した fold 別 AUC(比較基準)

| モデル | fold1 | fold2 | fold3 | fold4 | fold5 | OOF | fold1-2平均 |
|---|---|---|---|---|---|---|---|
| lgbm_tte_dig | 0.94501 | 0.94554 | 0.94674 | 0.94622 | 0.94590 | 0.94587 | 0.94528 |
| xgb | 0.94496 | 0.94565 | 0.94673 | 0.94625 | 0.94600 | 0.94591 | 0.94530 |
| catboost | 0.94504 | 0.94557 | 0.94675 | 0.94613 | 0.94598 | 0.94589 | 0.94531 |
| realmlp | 0.94496 | 0.94573 | 0.94675 | 0.94608 | 0.94594 | 0.94589 | 0.94535 |

### 3-1. catify → LightGBM : **❌ 不採用(-0.00016)**

| 構成 | fold1 | fold2 | 平均 | 差分 |
|---|---|---|---|---|
| ベースライン | 0.94492 | 0.94548 | 0.94520 | — |
| **+ catify** | 0.94479 | 0.94530 | **0.94504** | **-0.00016** |

**両 fold とも一貫して負**(-0.00013 / -0.00018)。採否基準 ±0.0002 には届かないが符号は安定しており、
少なくとも「プラスではない」と結論できる。

**解釈**: catify が CatBoost で +0.00170 効いたのは、CatBoost が cat_features に対して
**Ordered Target Statistics(順序付き CTR)** を自動で回すからであり、
「カテゴリとして扱うこと」自体に価値があるわけではない。LightGBM のネイティブ
categorical split は Fisher の最適分割(カテゴリを target mean 順に並べて分割点を探す)で、
**低カーデ数値列に対しては数値のまま使う通常の分割とほぼ同じ情報**しか得られない。
むしろ順序情報(Age の 25 < 26 < 27)を捨てる分だけ損をしている。

→ **catify は CatBoost 固有の武器であり、横展開の対象ではない。**

### 3-2. catify → XGBoost : **未完(ハーネス側の実装バグで中断)**

ベースラインは取得済み(下表)だが、catify 側が XGBoost のエラーで落ちた。

| 構成 | fold1 | fold2 | 平均 |
|---|---|---|---|
| ベースライン | 0.94508 | 0.94568 | 0.94538(公式 xgb の fold1-2 平均 0.94530 とよく一致) |
| + catify | — | — | 実行時エラー |

```
XGBoostError: Category index from DataFrame has floating point dtype,
consider using strings or integers instead.
```

`Age` などの低カーデ数値列は float dtype なので、`pd.Categorical` にそのまま包むと
**カテゴリ値が float** になり XGBoost が拒否する。`fe_all.catify` を「整数コードに変換してから
category dtype 化」するよう修正済み(修正後は未実行)。

**ただし再実行の優先度は低い**: LightGBM で catify が一貫して負(-0.00016)だった理由
(= CatBoost の Ordered TS 由来であって「カテゴリ化」自体に価値が無い)は XGBoost にも
そのまま当てはまる。XGBoost のカテゴリ分割は LightGBM と同じ partition 方式なので、
**同様に効かない可能性が高い**と見るべき。

### 3-3. 厳密値 Triple TE → RealMLP : **⚠ コスト面の重大な発見**

`fe_crosstest_realmlp.py --te` で 13列 + Smooth Keys 5本 = 18キー × 3 smooth = **54列**を
追加したところ、特徴量は 38 → **92列**になった。

**結果: 3時間枠内に fold 1 すら完走しなかった。スコアは未取得。**

**完走しなかった理由は CPU 競合であり、FE 自体の失敗ではない。**
実測すると、この RealMLP プロセスは2時間の壁時計時間に対し累計 **2,523 core-秒**しか
CPU を取れていなかった(**平均 0.35 コア**)。同時に GBDT のスクリーニングを回していたため、
8コア環境で NN が完全に押し出された形。ベースラインの1 fold は約 4,000 core-秒(290秒 × 7スレッド
× 2エポック)なので、**割り当てさえ足りていれば 1 fold は10分強で終わっていたはず**。

→ 教訓: **CLAUDE.md の「RealMLP は単独枠が必要」は文字どおり守る必要がある。**
GBDT と併走させると NN 側が 0.35 コアまで落ちる。

なお、追加コスト自体は無視できない見込みである(未実測・推定):
RealMLP の PBLD(periodic bias-linear-dense)埋め込みは**数値列1本ごとに**
`pbld_hidden_dim=20 → pbld_out_dim=5` の小ネットワークを持つため、数値列 11 → 65 は
前段の計算量と第1層入力幅にそのまま比例する。**GBDT と違い RealMLP では列追加のコストが高い**。

→ **横展開するなら 54列を丸ごと入れるのは得策ではない。** 下記「推奨」を参照。

---

## 4. モデル別の推奨(指揮官への提案)

実装は指揮官の判断。FE Lead は `fe_<model>.py` / `04_fe_run_<model>.py` を変更していない。

### CatBoost — ★最優先・コスト最小

**(a) `sk_inc_1` を捨てる(証明済みの重複列)**

`fe_catboost.py:SMOOTH_KEY_SPECS` の1行目 `("sk_inc_1", "Annual_Income_USD", 1.0)` は
`floor(income)` を作るが、**`Annual_Income_USD` は全行が整数**(実測で確認済み:
`(inc == floor(inc)).all() == True`)。したがってこのキーは厳密値 TE キーと**完全に同一**で、
Triple TE により **TE列を3本まるごと無駄に消費**している(さらに CatBoost はこれを
`per_float_feature_quantization` の対象にもしている)。

`03_fe_xgb.py` は同じ問題に気づいて意図的に除外しており(コードコメントあり)、
**モデル間で知見が共有されていなかった典型例**。

変更箇所: `03_fe_catboost.py` の `SMOOTH_KEY_SPECS`

```python
SMOOTH_KEY_SPECS = [
    ("sk_inc_10", "Annual_Income_USD", 10.0),      # sk_inc_1 (= 厳密値の重複) を置換
    ("sk_inc_100", "Annual_Income_USD", 100.0),
    ("sk_inc_1000", "Annual_Income_USD", 1000.0),
    ("sk_inc_10000", "Annual_Income_USD", 10000.0),  # XGB 側にあって CatBoost に無い解像度
    ("sk_com_1", "Daily_Commute_km", 1.0),
]
```

各解像度のキー数(実測): /10 → 5,933 / /100 → 1,142 / /1000 → 150 / /10000 → 16。
重複列が消え、代わりに中解像度(5,933キー)と粗解像度(16キー)が入る。

**(b) Triple TE に `auto` を入れる**

`04_fe_run_catboost.py:44` の `TRIPLE_SMOOTHS = [10.0, 20.0, 100.0]` は
**固定 m を3つ並べているだけ**で、LGBM / XGB が使っている `auto`(sklearn TargetEncoder の
経験ベイズ則 `m_i = p_i(1-p_i) / p(1-p)`)が入っていない。`auto` はキーごとに平滑強度を変える
**質的に別系統**の推定量で、10/20/100 とは冗長になりにくい。

```python
TRIPLE_SMOOTHS = ["auto", 10.0, 100.0]   # LGBM / XGB と揃える
```

⚠ 注意: `fe_catboost._smooth_tag` は `f"{smooth:g}"` で、文字列 `"auto"` を渡すと **例外になる**。
`_smooth_tag` を `fe_all._smooth_tag` 相当(str をそのまま返す)に直す必要がある。
さらに `04_fe_run_catboost.py` の `--hc-border` ターゲット判定が
`for tag in ("", "10", "20", "100")` とハードコードされているので `"auto"` を足すこと。

⚠ **【2026-09-18 追記・実測結果】この提案は 1-fold 検証で +0.00005 しか動かなかった(§3-4)。**
採否基準 ±0.0002 を大きく下回り、**AUC 改善策としては推奨しない**。
`sk_inc_1` の削除は「無駄列3本を消す」コード衛生上の意味しか無い(学習が少し速くなる)。
フル5-fold で再確認する価値はあるが、**優先度は低い**。

### LightGBM / XGBoost — catify は**入れないこと**(§3-1, §3-2)

横展開候補として最有力視されていた catify は、LightGBM で **-0.00016**(両fold一貫して負)。
CatBoost の +0.00170 は「カテゴリ化」ではなく **CatBoost の Ordered Target Statistics** に
由来するもので、他モデルには移植できない。

代わりに LGBM / XGB に推奨するのは以下。

**(a) Smooth Key の解像度を揃える(未検証・低コスト)**
- LightGBM: `/10000` が無い(`fe_lgbm.add_smooth_keys`)。1行追加で試せる。
- XGBoost: `/10` が無い(`fe_xgb.SMOOTH_KEY_SPECS`)。5,933キーの中解像度は
  13,214値の厳密値と 1,142値の /100 の**間が空いている**ので、埋める価値がある。

**(b) Smooth Key を Count Encoding のキーにも使う(未検証)**
Count Encoding は LGBM +0.00083 / XGB +0.00049 と両モデルで有効なのに、
**Smooth Key に対しては Count を取っていない**(`lgbm_preprocessing` の `skcnt` パターンも、
`xgb_preprocessing` の `smooth_keys_ce` も、現行ベスト構成では OFF)。
TE キーとして有効だったキーに Count も付けるのは自然な拡張で、実装済みのフラグを立てるだけ。

**(c) `interaction_constraints` による交互作用の全面禁止(未検証・reference A-2)**
CatBoost は既に `max_ctr_complexity=1` / `boosting_type="Plain"` で交互作用を抑えているが、
**LGBM / XGB には同等の制約が無い**。我々の「交互作用は全滅」という知見と方向が一致する。
reference_URL.md は +0.00149 と報告するが、TE 適用後は期待値を下方修正すべきと明記されている。

### RealMLP — 厳密値 TE の**部分適用**(§3-3)

RealMLP だけ 13列の厳密値 TE を持っていない(現状は combo 2列のみ)。
ただし 54列の一括投入は列追加コストが高く、単独枠が無いと完走しない。

**推奨: 高カーディナリティ2列に絞って TE を入れる。**
GBDT の feature importance でも TE の効果源は `Annual_Income_USD`(13,214値)と
`Daily_Commute_km`(805値)に集中していたことが Log.md に記録されている。
低カーデ5列 + カテゴリ6列の TE は XGB の実測で **±0.00000 と完全に無効**だったので、
最初から外してよい。

- 対象キー: `Annual_Income_USD`, `Daily_Commute_km`, `sk_inc10`, `sk_inc100`, `sk_inc1000`
- smooth: `auto` / 10 / 100 → **5キー × 3 = 15列**(54列の 1/4 弱)
- 実行: `uv run fe_crosstest_realmlp.py --folds 1 --te`(要・単独枠。GBDT を同時に回さないこと)
- 比較対象: 公式 `oof_realmlp.npy` の fold1 = **0.94496**

期待改善幅: 不明(GBDT では +0.003 だったが、RealMLP は `{col}_cat_` の embedding 経由で
厳密値情報を既に持っているため、**上積みは GBDT より小さいと見るべき**)。

---

### 3-4. CatBoost Smooth Key 修正 + `auto` 追加 : **差が出ず(+0.00005 / 1-fold のみ)**

| 構成 | n_feat | fold1 AUC | 差分 |
|---|---|---|---|
| A = 現行相当(`sk_inc_1` 重複あり / smooth 10,20,100) | 93 | 0.94511 | — |
| B = 提案(重複除去 + /10,/10000 追加 / smooth auto,10,100) | 96 | 0.94516 | **+0.00005** |

参考: 公式 `oof_catboost.npy` の fold1 = 0.94504(ハーネス A = 0.94511 とよく一致)。

**判定: 効果なし(判定不能)。** +0.00005 は採否基準 ±0.0002 を大きく下回り、
しかも **1 fold のみの測定**なのでノイズ幅はさらに広い(5-fold OOF のノイズ床 0.00015〜0.00033 に対し、
単一 fold は少なくともその倍は見ておくべき)。**「改善した」とは言えない。**

**解釈**: `sk_inc_1` が厳密値キーとビット同一であることは数学的に確実なので、
B は「無駄列3本を削って、代わりに解像度2つと `auto` 系統を足した」構成のはず。
にもかかわらず動かなかったということは、

- **重複列は CatBoost にとって実害が無かった**(木が単に選ばなければ済む)
- **Smooth Key の解像度も `auto` 系統も、Triple TE が既に捕らえている情報の焼き直しだった**

のいずれか(あるいは両方)。Log.md の「交互作用・多列組み合わせは全滅」「TEの効果源は
高カーデ数値列に集中」という既存知見と整合的で、**TE 周りの情報はすでに絞り尽くされている**
可能性が高い。

**残る価値**: スコア上の利得は無いが、`sk_inc_1` の削除は **TE列3本と
`per_float_feature_quantization` の枠を無償で節約**する(学習が少し速くなる)。
コード衛生としては直す価値があるが、**AUC 目的で優先すべき施策ではない**。

再現コマンド(ログ: `<scratchpad>/cat_ab1.log`):

```bash
# A = 現行 CatBoost 相当 (sk_inc_1 重複あり / smooth 10,20,100)
uv run fe_crosstest.py --model cat --folds 1 --n-jobs 7 --lr 0.06 --n-est 1000 --es 60 \
  --catify --te-smooths 10,20,100 --sk-scales 1,100,1000 --tag cbA_current
# B = 提案 (重複除去 + /10,/10000 追加 / smooth auto,10,100)
uv run fe_crosstest.py --model cat --folds 1 --n-jobs 7 --lr 0.06 --n-est 1000 --es 60 \
  --catify --te-smooths auto,10,100 --sk-scales 10,100,1000,10000 --tag cbB_auto_sk
```

比較基準: 公式 `oof_catboost.npy` の fold1 = **0.94504**。

---

## 4b. 次回セッションへの引き継ぎ(未完タスク)

| # | タスク | 状態 | 優先度 |
|---|---|---|---|
| 1 | **RealMLP 厳密値TE(高カーデ2列に絞る版)** | 54列版はCPU競合で未完。絞り込み版は未実行 | **★最優先**・20分(**必ず単独枠**) |
| 2 | Smooth Key への Count Encoding(LGBM/XGB) | 未実行(既存フラグを立てるだけ) | 中・各10分 |
| 3 | CatBoost A/B のフル5-fold 再確認 | 1-fold で +0.00005(効果なし判定) | 低 |
| 4 | LGBM/XGB の Smooth Key 解像度統一 | 未実行。ただし CatBoost で同種の変更が無効だった(§3-4) | **低**(期待値を下方修正) |
| 5 | XGBoost catify | バグ修正済み・未再実行。LGBM で負なので見込み薄 | 低 |

**§3-4 の結果を受けた優先度の再評価**: Smooth Key の解像度いじりと smooth 値の追加は
CatBoost で動かなかった。同じ性質の #4 は期待値を下げるべきで、
**残る本命は「RealMLP に厳密値TEを入れる」(=モデル間で本当に欠けている施策)だけ**。

**重要**: CLAUDE.md の「CatBoost / RealMLP は単独枠」は厳守すること。
本セッションでは RealMLP と GBDT を併走させた結果、RealMLP が平均 0.35 コアまで
押し出され、2時間かけて 1 fold すら終わらなかった(§3-3)。

---

## 5. 検証しなかった/する必要がない施策

Log.md の「打ち止めが確認済みのもの」に加え、本調査で以下を除外した。

| 施策 | 除外理由 |
|---|---|
| 四則演算の横展開 | 3モデルで実測済み。全滅 |
| 交互作用 TE の横展開 | 2/3/6/10/13列すべて全滅 |
| 行フィンガープリント | train 全行ユニークで原理的に機能しない |
| catify → RealMLP | `{col}_cat_`(floor値の factorize → embedding)が既に相当機能 |
| Count Encoding → CatBoost | 単独 -0.00017、上積み +0.00002 で実測済み |
