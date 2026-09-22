# XGBoost FE 検証結果 (S6E9)

- ベースライン: `baseline_xgb.py` OOF AUC **0.94124**(数値7列 + カテゴリ6列、native category dtype、デフォルトパラメータ)
- CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 固定。全て**フル5-fold・フルデータ**で測定
- 採否基準: ベースライン +0.0002 以上で採用
- 実装: FE関数 = `03_fe_xgb.py` / 実行 = `04_fe_run_xgb.py`(`--pattern` で切替)
- 最終構成の再現: `uv run src/04_fe_run_xgb.py --pattern final --save`

## 効果あり(採用)

| 施策 | ベースライン | 適用後 OOF AUC | 改善幅 | 備考 |
|---|---|---|---|---|
| 厳密値TE(全13列、単純fold内fit) `te_exact` | 0.94124 | 0.94307 | +0.00183 | S6E8の知見が本コンペでも再現。数値列もビン分割せず値そのままをキーに |
| 厳密値TE(高カーデ数値2列のみ)`te_high` | 0.94124 | 0.94323 | +0.00199 | Annual_Income_USD + Daily_Commute_km のみ。**単純TEでは全13列より良い** |
| 厳密値TE(高カーデ2列+Age)`te_mid` | 0.94124 | 0.94321 | +0.00197 | te_high と同等。Age追加の効果なし |
| **入れ子TE(全13列、inner-OOF)** `nte_exact` | 0.94124 | 0.94415 | +0.00291 | 学習行に inner StratifiedKFold(5) の out-of-fold 値を割当。単純TEより **+0.00108** |
| 入れ子TE(高カーデ2列)`nte_high` | 0.94124 | 0.94418 | +0.00294 | nte_exact とほぼ同等 |
| **入れ子TE(全13列)+ Count Encoding(全13列)** `nte_exact_ce` | 0.94124 | **0.94464** | **+0.00340** | TEとCountは非冗長で加算的(CE単独寄与 +0.00049) |
| **入れ子TE + CE + Ordinal エンコーディング** `nte_exact_ce_ord` ← **最終採用** | 0.94124 | **0.94463** | **+0.00339** | native category と同点(差 0.00001)。LightGBMがnative categoryのため**アンサンブル非相関性を狙って Ordinal を選択** |
| TEスムージング m=20 | - | - | - | m=5: 0.94311 / **m=20: 0.94323** / m=50: 0.94291 / m=100: 0.94271(te_high基準)。m=20 が最適 |

## 効果なし(不採用)

| 施策 | ベースライン | 適用後 OOF AUC | 差分 | 備考 |
|---|---|---|---|---|
| Ordinal エンコーディング(単体) `ord` | 0.94124 | 0.94133 | +0.00009 | 誤差水準。native category と実質同等 |
| One-Hot エンコーディング(単体) `ohe` | 0.94124 | 0.94129 | +0.00005 | 誤差水準。XGBoostではnative categoryと差が出ない |
| 四則演算(意味ベース7ペア × diff/ratio/sum/avg = 28特徴) `arith` | 0.94124 | 0.94128 | +0.00004 | 完全に無効。LightGBM担当の結果とも一致 |
| TE(カテゴリ6列のみ)`te_cat` | 0.94124 | 0.94123 | -0.00001 | 低カーデのカテゴリ列はモデルが既に扱えており無意味 |
| TE(低カーデ数値5列+カテゴリ6列)`te_lowcard` | 0.94124 | 0.94123 | -0.00001 | **低カーデ列のTEは完全に無効**。TEの効果源は高カーデ数値列であることが確定 |
| TEスムージング m=2(全13列)`te_exact_s2` | 0.94124 | 0.94272 | -0.00035 vs te_exact | 平滑化不足で過学習 |
| TEスムージング m=50 / m=100(高カーデ2列) | 0.94124 | 0.94291 / 0.94271 | -0.00032 / -0.00052 vs te_high | 平滑化しすぎで情報が消える |
| Count Encoding(高カーデ2列のみ)+ TE `te_high_ce` | 0.94124 | 0.94139 | +0.00015 | **CEは2列だけでは無効**。全13列に付けて初めて効く(`nte_exact_ce` 参照) |
| 入れ子TE + CE + One-Hot `nte_exact_ce_ohe` | 0.94124 | 0.94460 | -0.00004 vs final | 特徴量50列に増えるが Ordinal と同点。採用理由なし |

## 未検証で打ち切った施策(指揮官の知見共有により省略)

LightGBM担当が検証済みで無効と判明したため、重複回避のため打ち切り。

| 施策 | 理由 |
|---|---|
| 四則演算(全ペア総当たり) | LightGBM側で +0.00030(誤差水準)、実行時間6倍で見合わないと判明 |
| カテゴリ交互作用TE(6列15ペア) | LightGBM側で -0.00007、完全に無効と判明 |
| avg(2列平均) | 和の単調変換であり木モデルでは和と同一。実装不要 |

## 得られた知見

1. **厳密値TEの効果源は高カーディナリティ数値列**(Annual_Income_USD 13,214 unique / Daily_Commute_km 805 unique)。
   低カーディナリティ列(Age以下、カテゴリ全列)のTEは単独では完全に無効(`te_lowcard` +0.00000)。
2. **リーク対策の実装方式が精度を左右する。** 学習行にも fold全体の統計をそのまま与える単純TEより、
   inner StratifiedKFold(5) の out-of-fold 値を割り当てる**入れ子TE**の方が **+0.00108** 高い。
   これは「リーク防止」ではなく「学習行の楽観バイアス除去」による純粋な精度向上。
3. **入れ子TEにすると全13列TEが高カーデ2列TEに追いつく。**
   単純TEでは te_high(0.94323) > te_exact(0.94307) だったが、入れ子にすると逆転はほぼ解消
   (nte_high 0.94418 ≒ nte_exact 0.94415)。楽観バイアスが低カーデ列で特に強く出ていたため。
4. **Count Encoding は全列に付けて初めて効く**(2列のみでは無効)。TEと非冗長で加算的(+0.00049)。
5. **XGBoost固有の軸:** One-Hot / Ordinal / native category dtype の3方式は単体スコアで有意差なし
   (0.94129 / 0.94133 / 0.94124、いずれも誤差水準)。スコアが同点なので、
   **LightGBMがnative categoryを使う以上、XGBoostはOrdinalを選ぶのがアンサンブル上有利**という判断で
   最終構成に Ordinal を採用した。

## 最終構成 `--pattern final`

- エンコーディング: カテゴリ6列を **Ordinal**(整数コード、XGBoost には数値として渡す)
- **入れ子 Target Encoding**: 数値7列 + カテゴリ6列の全13列、**厳密値**をキー、スムージング m=20
  - 外側foldの学習データ内でのみ fit
  - 学習行 → inner StratifiedKFold(5, shuffle, seed=42) の out-of-fold 値
  - validation / test → 外側学習fold全体の統計
  - 未知値は fold の prior にフォールバック
- **Count(Frequency)Encoding**: 全13列、train+test結合で fit(教師なしのためリークなし)
- 特徴量数: 39(元13 + TE13 + CE13)
- XGBoost: デフォルトパラメータ(`tree_method="hist"`, `random_state=42`)。HPOは未実施
- **最終 OOF AUC = 0.94463(ベースライン 0.94124 から +0.00339)**
- fold別: 0.94353 / 0.94444 / 0.94557 / 0.94492 / 0.94472

### 成果物

| ファイル | 内容 |
|---|---|
| `submit/submission_xgb.csv` | test予測(5-fold平均) |
| `oof/oof_xgb.npy` | OOF予測 668,665行(NaNなし) |
| `oof/pred_xgb.npy` | test予測 286,571行 |
| `importance/importance_xgb.png` | feature importance 棒グラフ(上位30) |

### feature importance 上位(gain, fold平均)

`Subsidy_Available` が突出(0.65)、次いで `Environmental_Concern_Level`(0.26)。
FE由来では `Range_Anxiety_Level_ce`(3位)、**`Annual_Income_USD_te`(4位)** が生の `Annual_Income_USD`(6位)を
上回っており、高カーディナリティ数値列の厳密値TEが効いていることを裏付けている。

### 他モデルOOFとの相関(アンサンブル非相関性)

| 相手 | Pearson | Spearman |
|---|---|---|
| LightGBM (`oof_lgbm.npy`) | 0.99403 | 0.98609 |
| LightGBM s5 (`oof_lgbm_s5.npy`) | 0.99396 | 0.98643 |
| CatBoost (`oof_catboost.npy`) | 0.99104 | 0.98743 |

CatBoost との相関が最も低く(Pearson 0.99104)、XGBoost は CatBoost とのブレンドで最も寄与が期待できる。
Ordinal エンコーディング採用により LightGBM(native category)との実装差も確保している。
