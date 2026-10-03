"""EDA の図を datacheck/ に保存する。

図の構成と順番は notebooks/01_eda.ipynb に合わせている(ファイル名先頭の連番が表示順)。
各関数の描き方はノートブックの同じ節のセルと同じ。

使い方:
    uv run src/01_eda.py
"""

import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

SAVE_DIR = "datacheck"
TARGET = "Will_Buy_EV"

# Windows 標準の日本語フォントを使う(無い場合は英字フォントのまま)
for _f in ["Meiryo", "Yu Gothic", "MS Gothic"]:
    if _f in {f.name for f in mpl.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = _f
        break

JP_NAMES = {
    "id": "ID",
    "Age": "年齢",
    "Annual_Income_USD": "年収(USD)",
    "Daily_Commute_km": "1日の通勤距離(km)",
    "Number_of_Cars_Owned": "保有台数",
    "Charging_Stations_Near_Home": "自宅近くの充電スタンド数",
    "Charging_Stations_Near_Work": "職場近くの充電スタンド数",
    "Environmental_Concern_Level": "環境意識レベル(1-5)",
    "Gender": "性別",
    "City_Type": "居住地タイプ",
    "Current_Car_Type": "現在の車種",
    "Home_Charging_Possible": "自宅充電の可否",
    "Subsidy_Available": "補助金の有無",
    "Range_Anxiety_Level": "航続距離への不安度",
    "Will_Buy_EV": "EV購入(目的変数)",
}

# 順序のある列は意味の順、それ以外はアルファベット順に並べる
CAT_ORDER = {
    "Gender": ["Female", "Male", "Other"],
    "City_Type": ["Rural", "Suburban", "Urban"],
    "Current_Car_Type": ["Hatchback", "SUV", "Sedan", "Truck"],
    "Home_Charging_Possible": ["No", "Yes"],
    "Subsidy_Available": ["No", "Yes"],
    "Range_Anxiety_Level": ["Low", "Medium", "High"],
}
ORDER_MAP = {"City_Type": ["Rural", "Suburban", "Urban"],
             "Range_Anxiety_Level": ["Low", "Medium", "High"],
             "Home_Charging_Possible": ["No", "Yes"]}


# 現在の図を datacheck/ に保存して閉じる
def save(name):
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, name)
    plt.savefig(path, dpi=150)
    print(f"  → {path}")
    plt.close()


# 購入率を log-odds に変換する。0/1 に張り付くと発散するのでクリップする
def to_log_odds(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# 01-1-1: 列の一覧(列名・日本語名・型・欠損・ユニーク数)を表の画像にする
def plot_column_table(train):
    rows = []
    for col in train.columns:
        is_str = train[col].dtype == object or pd.api.types.is_string_dtype(train[col])
        uniq = sorted(train[col].unique()) if is_str else []
        rows.append([col, JP_NAMES.get(col, ""), str(train[col].dtype), f"{train[col].isnull().sum():,}",
                     f"{train[col].nunique():,}", ", ".join(map(str, uniq)) if is_str else ""])

    headers = ["列名", "日本語名", "型", "欠損", "ユニーク数", "ユニーク値(文字列列のみ)"]
    fig, ax = plt.subplots(figsize=(15, 0.42 * len(rows) + 1))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left",
                   colWidths=[0.21, 0.17, 0.07, 0.05, 0.08, 0.30])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:                                   # ヘッダー
            cell.set_facecolor("#4c566a")
            cell.set_text_props(color="white", weight="bold")
        elif rows[r - 1][0] == TARGET:               # 目的変数を強調
            cell.set_facecolor("#ffe9b0")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f5f5")
    ax.set_title(f"train の列一覧  ({train.shape[0]:,} 行 x {train.shape[1]} 列)", fontsize=13, weight="bold", pad=14)
    save("01_列の一覧.png")


# 01-1-3: 数値列ごとに、件数の積み上げ(上段)と値ごとの購入率(下段)を描く
def plot_numeric_distribution_and_rate(train, y, num_cols):
    overall_rate = y.mean()
    fig, axes = plt.subplots(2, len(num_cols), figsize=(22, 7.5))
    rates = {}
    for i, col in enumerate(num_cols):
        is_discrete = train[col].nunique() <= 50
        sns.histplot(data=train, x=col, hue=TARGET, multiple="stack", palette="viridis",
                     discrete=is_discrete, bins="auto" if is_discrete else 50, ax=axes[0, i], legend=(i == 0))
        axes[0, i].set_title(col, fontsize=9)
        axes[0, i].set_xlabel("")
        plt.setp(axes[0, i].get_xticklabels(), rotation=45, ha="right")

        # 値が細かい列は50分位に区切ってから集計する
        group = (train[col] if is_discrete
                 else pd.qcut(train[col], 50, duplicates="drop").apply(lambda iv: iv.mid))
        rates[col] = y.groupby(group, observed=True).mean()
        axes[1, i].plot(rates[col].index.astype(float), rates[col].values, marker="o", ms=3)
        axes[1, i].axhline(overall_rate, color="red", ls="--", lw=1)
        axes[1, i].set_title(f"購入率 by {col}", fontsize=9)
        plt.setp(axes[1, i].get_xticklabels(), rotation=45, ha="right")

    # 下段は全列で縦軸を揃え、列ごとの効き方の差を比べられるようにする
    y_max = max(r.max() for r in rates.values()) * 1.05
    for ax in axes[1]:
        ax.set_ylim(0, y_max)
    axes[1, 0].set_ylabel("購入率")
    save("02_数値列の分布と購入率.png")


# 01-1-4: カテゴリ列ごとに、件数の積み上げ(上段)と値ごとの購入率(下段)を描く
def plot_categorical_count_and_rate(train, y, cat_cols):
    overall_rate = y.mean()
    color_no, color_yes = sns.color_palette("viridis", 2)
    fig, axes = plt.subplots(2, len(cat_cols), figsize=(20, 7))
    for i, col in enumerate(cat_cols):
        order = CAT_ORDER[col]
        x = range(len(order))
        counts = pd.crosstab(train[col], train[TARGET]).reindex(order)
        axes[0, i].bar(x, counts["No"], color=color_no, label="No")
        axes[0, i].bar(x, counts["Yes"], bottom=counts["No"], color=color_yes, label="Yes")
        axes[0, i].set_title(col, fontsize=10)

        rate = y.groupby(train[col]).mean().reindex(order)
        axes[1, i].bar(x, rate.values, color=color_yes)
        axes[1, i].axhline(overall_rate, color="red", ls="--", lw=1)
        axes[1, i].set_title(f"P(Yes) by {col}", fontsize=10)
        for ax in axes[:, i]:
            ax.set_xticks(list(x))
            ax.set_xticklabels(order, rotation=45, ha="right")

    # 下段は全列で縦軸を揃える
    rate_max = max(y.groupby(train[c]).mean().max() for c in cat_cols) * 1.05
    for ax in axes[1]:
        ax.set_ylim(0, rate_max)
    axes[0, 0].legend(title=TARGET)
    axes[0, 0].set_ylabel("件数")
    axes[1, 0].set_ylabel("購入率")
    save("03_カテゴリ列の件数と購入率.png")


# 01-1-8: train と test の分布を重ねて比べる
def plot_train_test_distribution(tr, te, num_cols, cat_cols):
    low_card = [c for c in num_cols if tr[c].nunique() <= 50]
    cols = num_cols + cat_cols
    n_cols = 4
    n_rows = (len(cols) + n_cols - 1) // n_cols
    plt.figure(figsize=(16, 4 * n_rows))
    for i, col in enumerate(cols):
        plt.subplot(n_rows, n_cols, i + 1)
        if col in cat_cols or col in low_card:
            dist = pd.DataFrame({"train": tr[col].value_counts(normalize=True),
                                 "test": te[col].value_counts(normalize=True)}).sort_index()
            dist.plot(kind="bar", ax=plt.gca(), width=0.8, legend=(i == 0))
        else:
            sns.kdeplot(tr[col], label="train", fill=True, alpha=0.3)
            sns.kdeplot(te[col], label="test", fill=True, alpha=0.3)
            if i == 0:
                plt.legend()
        plt.title(f"train vs test: {col}", fontsize=10)
        plt.xticks(rotation=45, ha="right")
    save("04_trainとtestの分布比較.png")


# 2列の関連の強さを 0〜1 で返す(Cramér's V。カテゴリ列どうしでも使える)
def cramers_v(a, b):
    table = pd.crosstab(a, b)
    if min(table.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(table)[0]
    return np.sqrt((chi2 / table.to_numpy().sum()) / (min(table.shape) - 1))


# 01-2-1: Spearman 相関(数値列 + 目的変数)と Cramér's V(全列)を並べる
def plot_correlation_heatmaps(train, y, num_cols, cat_cols):
    # Cramér's V 用に、値の細かい列は10分位に区切る
    binned = {c: (train[c] if train[c].nunique() <= 20 else pd.qcut(train[c], 10, duplicates="drop").astype(str))
              for c in num_cols}
    binned.update({c: train[c] for c in cat_cols})
    binned[TARGET] = train[TARGET]

    all_cols = num_cols + cat_cols + [TARGET]
    cramer = pd.DataFrame(np.nan, index=all_cols, columns=all_cols, dtype=float)
    for i, a in enumerate(all_cols):
        for b in all_cols[i:]:
            value = 1.0 if a == b else cramers_v(binned[a], binned[b])
            cramer.loc[a, b] = cramer.loc[b, a] = value

    numeric_frame = train[num_cols].copy()
    numeric_frame[TARGET] = y
    spearman = numeric_frame.corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(21, 8))
    sns.heatmap(spearman, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
                linewidths=.5, cbar_kws={"shrink": .7}, annot_kws={"size": 8}, ax=axes[0])
    axes[0].set_title("Spearman 相関(数値列 + 目的変数)— 向きが分かる", fontsize=11)
    sns.heatmap(cramer, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, square=True,
                linewidths=.5, cbar_kws={"shrink": .7}, annot_kws={"size": 7}, ax=axes[1])
    axes[1].set_title("Cramér's V(全列)— 型を問わず関連の強さが分かる", fontsize=11)
    save("05_相関ヒートマップ.png")


# 01-2-2: 関係の強い列の組(充電環境のクラスタ)を、X の値ごとの Y の割合で描く
def plot_charging_cluster(train):
    pairs = [
        ("Charging_Stations_Near_Work", "City_Type"),
        ("Charging_Stations_Near_Home", "City_Type"),
        ("City_Type", "Home_Charging_Possible"),
        ("Home_Charging_Possible", "Range_Anxiety_Level"),
        ("Charging_Stations_Near_Home", "Home_Charging_Possible"),
    ]
    order = {"City_Type": ["Rural", "Suburban", "Urban"],
             "Range_Anxiety_Level": ["Low", "Medium", "High"],
             "Home_Charging_Possible": ["Yes", "No"]}
    fig, axes = plt.subplots(2, 3, figsize=(20, 8))
    for ax, (x_col, y_col) in zip(axes.ravel(), pairs):
        share = pd.crosstab(train[y_col], train[x_col], normalize="columns")
        if y_col in order:
            share = share.reindex(order[y_col])
        if x_col in order:
            share = share[order[x_col]]
        sns.heatmap(share * 100, annot=share.shape[1] <= 6, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100,
                    linewidths=.5, cbar_kws={"label": "X の列内での割合 (%)"}, ax=ax)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col}(X) × {y_col}(Y)", fontsize=10)
    axes.ravel()[-1].axis("off")
    save("06_充電環境のクラスタ.png")


# 01-2-2: 自宅充電の可否ごとに、自宅近くのスタンド数と購入の log-odds を描く
def plot_home_charging_interaction(train, y):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for can_charge in ["No", "Yes"]:
        subset = train["Home_Charging_Possible"] == can_charge
        rate_by_stations = y[subset].groupby(train.loc[subset, "Charging_Stations_Near_Home"], observed=True).mean()
        ax.plot(rate_by_stations.index, to_log_odds(rate_by_stations).values, marker="o", label=f"自宅充電 {can_charge}")
    ax.set_xlabel("自宅近くの充電スタンド数")
    ax.set_ylabel("log-odds(購入)")
    ax.set_title("充電スタンド件数(x) と log-odds(y) _ 自宅充電yes/no 別", fontsize=11)
    ax.legend()
    save("07_自宅充電と自宅スタンド数の交互作用.png")


# 01-2-3: 各説明変数の値ごとに「補助金ありの割合」を描く
def plot_subsidy_share(train, num_cols, cat_cols):
    subsidy = (train["Subsidy_Available"] == "Yes").astype(int)
    other_cols = [c for c in num_cols + cat_cols if c != "Subsidy_Available"]
    fig, axes = plt.subplots(2, 6, figsize=(22, 7), sharey=True)
    for ax, col in zip(axes.ravel(), other_cols):
        group = train[col] if train[col].nunique() <= 20 else pd.qcut(train[col], 5, duplicates="drop")
        share = subsidy.groupby(group, observed=True).mean()
        if col in ORDER_MAP:
            share = share.reindex(ORDER_MAP[col])
        labels = [str(v) if not hasattr(v, "mid") else f"{v.mid:,.0f}" for v in share.index]
        ax.bar(range(len(share)), share.values, color="#5b8def")
        ax.axhline(subsidy.mean(), color="red", ls="--", lw=1)
        ax.set_xticks(range(len(share)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(col, fontsize=9)
    axes[0, 0].set_ylabel("補助金ありの割合")
    axes[1, 0].set_ylabel("補助金ありの割合")
    axes[0, 0].set_ylim(0, 1)
    plt.suptitle(f"① 各説明変数の値ごとに見た「補助金ありの割合」(赤線は全体平均 {subsidy.mean():.1%})", fontsize=11)
    save("08_補助金ありの割合.png")


# 01-2-3: 補助金あり/なしで分けた購入率(上段)と、補助金の効果の大きさ(下段)を描く
def plot_subsidy_effect(train, y):
    min_buyers = 20   # どちらかの群の購入者がこれ未満の水準は、log-odds が不安定なので除外する
    check_cols = ["Environmental_Concern_Level", "Annual_Income_USD", "Range_Anxiety_Level",
                  "Home_Charging_Possible", "City_Type", "Age"]
    fig, axes = plt.subplots(2, len(check_cols), figsize=(22, 7.5))
    for i, col in enumerate(check_cols):
        group = train[col] if train[col].nunique() <= 20 else pd.qcut(train[col], 5, duplicates="drop")
        grouped = y.groupby([group, train["Subsidy_Available"]], observed=True)
        rate = grouped.mean().unstack()
        buyers = grouped.sum().unstack()
        if col in ORDER_MAP:
            rate, buyers = rate.reindex(ORDER_MAP[col]), buyers.reindex(ORDER_MAP[col])
        labels = [str(v) if not hasattr(v, "mid") else f"{v.mid:,.0f}" for v in rate.index]
        x = list(range(len(rate)))

        for s, color in [("No", "#999999"), ("Yes", "#e4572e")]:
            axes[0, i].plot(x, rate[s].values * 100, marker="o", color=color, label=f"補助金 {s}")
        axes[0, i].set_title(col, fontsize=9)
        axes[0, i].set_xticks(x)
        axes[0, i].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

        # 補助金あり - なし の log-odds 差。横一線なら補助金の効き方は誰でも同じ
        lift = to_log_odds(rate["Yes"]) - to_log_odds(rate["No"])
        too_few = buyers.min(axis=1) < min_buyers
        lift[too_few] = np.nan
        axes[1, i].plot(x, lift.values, marker="o", color="#2e86ab")
        for xi in np.flatnonzero(too_few.to_numpy()):
            axes[1, i].text(xi, 0.4, "除外", ha="center", fontsize=8, color="gray")
        axes[1, i].set_ylim(0, 6)
        axes[1, i].set_xticks(x)
        axes[1, i].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0, 0].set_ylabel("購入率 (%)")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set_ylabel("補助金の効果\n(log-odds 差)")
    plt.suptitle("② 補助金あり/なしで分けた購入率(上)と、補助金の効果の大きさ(下)", fontsize=11)
    save("09_補助金の効き方.png")


# 01-2-4: 収入の5分位ごとの log-odds を、年齢帯・環境意識レベル別に描く
def plot_income_combinations(train, y):
    age_band = pd.cut(train["Age"], [0, 30, 40, 50, 60, 100], labels=["~30歳", "31-40", "41-50", "51-60", "61歳~"])
    income_band = pd.qcut(train["Annual_Income_USD"], 5, labels=["最低20%", "下位40%", "中位", "上位40%", "最高20%"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, (row_key, row_name) in zip(axes, [(age_band, "年齢帯"),
                                              (train["Environmental_Concern_Level"], "環境意識レベル")]):
        rate_table = y.groupby([row_key, income_band], observed=True).mean().unstack()
        for label, series in to_log_odds(rate_table).iterrows():
            ax.plot(range(len(series)), series.values, marker="o", label=f"{row_name} {label}")
        ax.set_xticks(range(rate_table.shape[1]))
        ax.set_xticklabels(rate_table.columns, rotation=30, ha="right")
        ax.set_ylabel("log-odds(購入)")
        ax.set_title(f"収入 × {row_name}", fontsize=11)
        ax.legend(fontsize=8)
    save("10_収入と年齢・環境意識の組み合わせ.png")


# データを読み、概要を表示してから EDA の図をすべて保存する
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = (train[TARGET] == "Yes").astype(int)
    num_cols = train.select_dtypes(include=[np.number]).columns.drop("id").tolist()
    cat_cols = [c for c in train.columns if c not in num_cols + ["id", TARGET]]

    print(f"train: {train.shape} / test: {test.shape}")
    print("欠損値(train / test):", int(train.isnull().sum().sum()), "/", int(test.isnull().sum().sum()))
    print("件数カウント:", train[TARGET].value_counts().to_dict())
    print("購入率:", round(y.mean(), 4))

    plot_column_table(train)
    plot_numeric_distribution_and_rate(train, y, num_cols)
    plot_categorical_count_and_rate(train, y, cat_cols)
    plot_train_test_distribution(train, test, num_cols, cat_cols)
    plot_correlation_heatmaps(train, y, num_cols, cat_cols)
    plot_charging_cluster(train)
    plot_home_charging_interaction(train, y)
    plot_subsidy_share(train, num_cols, cat_cols)
    plot_subsidy_effect(train, y)
    plot_income_combinations(train, y)


if __name__ == "__main__":
    main()
    print("finish !!")
