"""LightGBM の学習・評価(S6E9)。

`03_feature_engineering_lgbm.py` の関数で**本番で使う列だけ**を作り、
StratifiedKFold(n_splits=5, shuffle=True, random_state=42) で学習する。--save で提出ファイル・OOF・重要度を保存する。

作る列(46 列)
    生の 13 列(カテゴリは native category)
    + digit(年収・通勤距離)
    + Count(年収・通勤距離)
    + Target Encoding(fe.te_plan(): 数値7列・Smooth Keys 4本・自宅充電 × 自宅スタンド数。fold 内で Out-of-Fold)

交互作用の制約(--interaction income。2026-09-29 採用): 年収系の列とそれ以外の列を、同じ木の枝で組み合わせない

例
    uv run src/04_train_and_evaluate_lgbm.py --max_bin 1024 --feature_fraction 0.3 --max_depth 5 \\
        --learning_rate 0.03 --n_estimators 8000 --early_stopping 200 --n_jobs 7 --interaction income --save
"""

from __future__ import annotations

import argparse
import importlib
import os
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

fe = importlib.import_module("03_feature_engineering_lgbm")

N_SPLITS = 5  # fold の分け方は全モデル共通。変更禁止
SEED = 42


# コマンドライン引数の定義を作る
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=N_SPLITS, help="実際に学習する fold 数(分け方は常に 5)")
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--num_leaves", type=int, default=31)
    p.add_argument("--feature_fraction", type=float, default=1.0,
                   help="木ごとに使う列の割合 (colsample_bytree)。1.0 = 全列")
    p.add_argument("--max_depth", type=int, default=0, help="木の深さ上限 (0 = 無制限)")
    p.add_argument("--max_bin", type=int, default=255, help="LightGBM の max_bin(本番は 1024)")
    p.add_argument("--sample", type=float, default=1.0, help="スクリーニング用に train を割合で間引く")
    p.add_argument("--inner", type=int, default=5, help="Target Encoding の内側の分割数")
    p.add_argument("--n_jobs", type=int, default=-1, help="LightGBM のスレッド数")
    p.add_argument("--early_stopping", type=int, default=0,
                   help="検証 fold の AUC が指定本数改善しなければ止める(0 = 使わない)")
    p.add_argument("--save", action="store_true", help="提出ファイル・OOF・重要度を保存する")
    p.add_argument("--tag", default="lgbm", help="成果物のファイル名(oof/oof_<tag>.npy など)")
    p.add_argument("--interaction", choices=["none", "income"], default="none",
                   help="交互作用の制約。income=年収系の列と他の列を同じ木の枝で組み合わせない(本番は income。2026-09-29 採用)")
    p.add_argument("--dump-features", action="store_true",
                   help="学習せず、fold 1 の列名を docs/features_<tag>.json に書いて終了する")
    return p


# fold によらない部分(生の列・digit・Count)と、Target Encoding のキーを作る
def prepare(train: pd.DataFrame, test: pd.DataFrame):
    train_c, test_c = fe.make_categorical(train, test)
    keys_tr, keys_te = fe.make_key_frame(train, test)
    keys_tr, keys_te = fe.add_smooth_keys(keys_tr, train), fe.add_smooth_keys(keys_te, test)

    d_tr = fe.add_digit_features(train, cols=fe.HIGH_CARD_NUMERIC)
    d_te = fe.add_digit_features(test, cols=fe.HIGH_CARD_NUMERIC, keep=list(d_tr.columns))   # test も train と同じ列に
    c_tr, c_te = fe.count_encode(keys_tr, keys_te, fe.HIGH_CARD_NUMERIC)

    raw = fe.NUMERIC_COLS + fe.CATEGORICAL_COLS
    static_tr = pd.concat([train_c[raw], d_tr, c_tr], axis=1)
    static_te = pd.concat([test_c[raw], d_te, c_te], axis=1)
    return static_tr, static_te, keys_tr, keys_te


# 1 つの fold の学習行・検証行・test の行列を作る(Target Encoding は学習行だけで作る)
def fold_matrices(prep, y: np.ndarray, tr_idx, va_idx, n_inner: int = 5):
    static_tr, static_te, keys_tr, keys_te = prep
    te_tr, (te_va, te_te) = fe.target_encode_plan(
        keys_tr.iloc[tr_idx], y[tr_idx], [keys_tr.iloc[va_idx], keys_te], fe.te_plan(), n_inner=n_inner, seed=SEED)
    X_tr = pd.concat([static_tr.iloc[tr_idx].reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1)
    X_va = pd.concat([static_tr.iloc[va_idx].reset_index(drop=True), te_va.reset_index(drop=True)], axis=1)
    X_te = pd.concat([static_te.reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)
    return X_tr, X_va, X_te


# 交互作用の制約のグループ(列の番号のリスト)。年収系の列(年収・その Smooth Keys・digit・Count・TE)と、それ以外に分ける
def interaction_groups(cols, mode="income"):
    """同じグループの列どうしだけが、同じ木の枝で組み合わさる。補助金・環境意識が分割の大半を占めるなか、
    年収の細部を別の枝で学ばせる(2026-09-29 採用。単体 +0.000091, z=+4.21 / アンサンブル +0.000047, z=+6.50)。"""
    assert mode == "income", mode
    inc = [i for i, c in enumerate(cols) if "Annual_Income_USD" in c or "sk_inc" in c]
    return [inc, [i for i in range(len(cols)) if i not in inc]]


# LightGBM を 5-fold で学習・評価し、成果物を保存する
def main():
    args = build_parser().parse_args()
    t0 = time.time()

    train, test = fe.load_data()
    if args.sample < 1.0:
        idx = train.sample(frac=args.sample, random_state=SEED).index.sort_values()
        train = train.loc[idx].reset_index(drop=True)
    y = (train[fe.TARGET] == "Yes").astype(int).to_numpy()

    prep = prepare(train, test)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(train), np.nan)
    test_pred = np.zeros(len(test))
    importance, feat_names, best_iters, n_used = None, None, [], 0

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        if fold >= args.folds:
            break
        X_tr, X_va, X_te = fold_matrices(prep, y, tr_idx, va_idx, n_inner=args.inner)

        if args.dump_features:
            import feature_catalog
            feature_catalog.dump(args.tag, "LightGBM", X_tr.columns,
                                 cat_features=[c for c in X_tr.columns if str(X_tr[c].dtype) == "category"],
                                 note="本番の構成(fe.te_plan())")
            return

        model = LGBMClassifier(
            random_state=SEED, verbosity=-1, n_estimators=args.n_estimators, learning_rate=args.learning_rate,
            num_leaves=args.num_leaves, max_bin=args.max_bin, n_jobs=args.n_jobs,
            **({"colsample_bytree": args.feature_fraction} if args.feature_fraction < 1.0 else {}),
            **({"max_depth": args.max_depth} if args.max_depth > 0 else {}),
            **({"interaction_constraints": interaction_groups(list(X_tr.columns), args.interaction)}
               if args.interaction != "none" else {}),
        )

        if args.early_stopping > 0:
            from lightgbm import early_stopping, log_evaluation
            model.fit(X_tr, y[tr_idx], eval_set=[(X_va, y[va_idx])], eval_metric="auc",
                      callbacks=[early_stopping(args.early_stopping, verbose=False), log_evaluation(0)])
            best_iters.append(int(model.best_iteration_ or args.n_estimators))
        else:
            model.fit(X_tr, y[tr_idx])

        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1]
        n_used += 1
        if importance is None:
            importance, feat_names = np.zeros(X_tr.shape[1]), list(X_tr.columns)
        importance += model.feature_importances_

        print(f"fold {fold} AUC: {roc_auc_score(y[va_idx], oof[va_idx]):.5f} ({time.time() - t0:.0f}s)", flush=True)

    mask = ~np.isnan(oof)
    auc = roc_auc_score(y[mask], oof[mask])
    test_pred /= max(n_used, 1)
    bi = f" best_iters={best_iters} median_best={int(np.median(best_iters))}" if best_iters else ""
    print(f"folds={n_used} n_feat={len(feat_names)} max_bin={args.max_bin} lr={args.learning_rate} "
          f"n_est={args.n_estimators} es={args.early_stopping} OOF_AUC={auc:.5f} time={time.time() - t0:.0f}s{bi}",
          flush=True)

    if args.save:
        tag = args.tag
        for d in ("submit", "oof", "importance"):
            os.makedirs(d, exist_ok=True)
        np.save(f"oof/oof_{tag}.npy", oof)
        np.save(f"oof/pred_{tag}.npy", test_pred)
        pd.DataFrame({"id": test["id"], fe.TARGET: test_pred}).to_csv(f"submit/submission_{tag}.csv", index=False)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        imp = pd.Series(importance / n_used, index=feat_names).sort_values(ascending=True)
        top = imp.tail(40)
        fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(top))))
        ax.barh(top.index, top.to_numpy(), color="#2f7ed8")
        ax.set_xlabel("mean split importance")
        ax.set_title(f"LightGBM feature importance (OOF AUC {auc:.5f})")
        fig.tight_layout()
        fig.savefig(f"importance/importance_{tag}.png", dpi=130)
        plt.close(fig)
        imp.sort_values(ascending=False).to_csv(f"importance/importance_{tag}.csv")
        print(f"saved submit/submission_{tag}.csv, oof/oof_{tag}.npy, oof/pred_{tag}.npy, importance/importance_{tag}.png")


if __name__ == "__main__":
    main()
