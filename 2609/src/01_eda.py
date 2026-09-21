import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

SAVE_DIR = "datacheck"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET = "Will_Buy_EV"
PALETTE = "viridis"

# 列名が長く、4列並びのサブプロットでタイトルが隣と重なるため全図で縮小する
plt.rcParams["axes.titlesize"] = 10

# データ読み込み
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")
print(f"trainの形状: {train.shape}")
print(f"testの形状: {test.shape}")
print(f"sample_submissionの形状: {sample_submission.shape}")
print("=" * 50)

print("train.info():")
train.info()
print("=" * 30)
print("test.info():")
test.info()
print("=" * 30)

print(f"trainの目的変数の分布:\n{train[TARGET].value_counts()}")
print(f"購入率(Yes): {(train[TARGET] == 'Yes').mean():.4f}")
print("-" * 30)
print(f"欠損値確認(train):\n{train.isnull().sum()}")
print("-" * 30)
print(f"欠損値確認(test):\n{test.isnull().sum()}")
print("-" * 80)

print("カテゴリ列の unique チェック")
for col in train.select_dtypes(exclude=[np.number]).columns:
    print(f"  {col:28s} {sorted(train[col].unique())}")
print("-" * 80)

print("数値列の nunique チェック")
for col in train.select_dtypes(include=[np.number]).columns.drop("id"):
    print(f"  {col:28s} nunique={train[col].nunique():,}")
print("-" * 80)

NUM_COLS = train.select_dtypes(include=[np.number]).columns.drop("id").tolist()
CAT_COLS = [c for c in train.select_dtypes(exclude=[np.number]).columns if c != TARGET]
LOW_CARD_MAX = 50
LOW_CARD_NUM = [c for c in NUM_COLS if train[c].nunique() <= LOW_CARD_MAX]
HIGH_CARD_NUM = [c for c in NUM_COLS if train[c].nunique() > LOW_CARD_MAX]


def save(name, show=False):
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, name)
    plt.savefig(path, dpi=150)
    print(f"  → {path}")
    if show:
        plt.show()
    plt.close()


def grid(n, n_cols=4, w=16, h=4):
    n_rows = (n + n_cols - 1) // n_cols
    plt.figure(figsize=(w, h * n_rows))
    return n_rows, n_cols


def plot_target_distribution(df, target, show=False):
    counts = df[target].value_counts()
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=counts.index, y=counts.values, palette=PALETTE)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / counts.sum():.1%})", ha="center", va="bottom")
    plt.title(f"Distribution of {target}")
    plt.ylabel("count")
    save("01_target_distribution.png", show)


def plot_categorical_histograms(df, target, show=False):
    n_rows, n_cols = grid(len(CAT_COLS), n_cols=3)
    for i, col in enumerate(CAT_COLS):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(data=df, x=col, hue=target, multiple="stack",
                     palette=PALETTE, shrink=0.8)
        plt.title(f"Countplot of {col}")
        plt.xticks(rotation=45, ha="right")
        if i > 0:
            plt.legend().remove()
    save("02_categorical_hist.png", show)


def plot_numeric_histograms(df, target, show=False):
    n_rows, n_cols = grid(len(NUM_COLS))
    for i, col in enumerate(NUM_COLS):
        plt.subplot(n_rows, n_cols, i + 1)
        # 低カーデ列は値ごとに1本のバーにしないと、ビンが値をまたいで分布が歪む
        discrete = df[col].nunique() <= LOW_CARD_MAX
        sns.histplot(data=df, x=col, hue=target, multiple="stack", palette=PALETTE,
                     discrete=discrete, bins="auto" if discrete else 50)
        plt.title(f"Histogram of {col}")
        plt.xticks(rotation=45, ha="right")
        if i > 0:
            plt.legend().remove()
    save("03_numeric_hist.png", show)


def plot_numeric_log_histograms(df, target, show=False):
    n_rows, n_cols = grid(len(HIGH_CARD_NUM), n_cols=2, w=12)
    for i, col in enumerate(HIGH_CARD_NUM):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(x=np.log1p(df[col].clip(lower=0)), hue=df[target],
                     bins=50, multiple="stack", palette=PALETTE)
        plt.title(f"Log Histogram of {col}")
        plt.xlabel(f"log1p({col})")
    save("04_numeric_log_hist.png", show)


def plot_correlation_heatmap(df, target, show=False):
    corr_df = df[NUM_COLS].copy()
    corr_df[target] = (df[target] == "Yes").astype(int)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (target: Yes=1)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    save("05_correlation_heatmap.png", show)


def plot_boxplots_by_target(df, target, show=False):
    n_rows, n_cols = grid(len(NUM_COLS))
    for i, col in enumerate(NUM_COLS):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.boxplot(data=df, x=target, y=col, palette=PALETTE)
        plt.title(f"{col} by Target")
    save("06_boxplots_by_target.png", show)


def plot_target_rate_by_category(df, target, show=False):
    y = (df[target] == "Yes").astype(int)
    base = y.mean()
    n_rows, n_cols = grid(len(CAT_COLS), n_cols=3)
    for i, col in enumerate(CAT_COLS):
        plt.subplot(n_rows, n_cols, i + 1)
        rate = y.groupby(df[col]).mean().sort_values()
        sns.barplot(x=rate.index, y=rate.values, palette=PALETTE)
        plt.axhline(base, color="red", ls="--", lw=1, label=f"overall {base:.3f}")
        plt.title(f"Purchase rate by {col}")
        plt.ylabel("P(Yes)")
        plt.xticks(rotation=45, ha="right")
        plt.legend(fontsize=8)
    save("07_target_rate_by_category.png", show)


def plot_target_rate_by_numeric(df, target, show=False):
    # 厳密値TEが効いた構造を確認するため、低カーデ列は値そのまま、高カーデ列は分位ビンで購入率を見る
    y = (df[target] == "Yes").astype(int)
    base = y.mean()
    n_rows, n_cols = grid(len(NUM_COLS))
    for i, col in enumerate(NUM_COLS):
        plt.subplot(n_rows, n_cols, i + 1)
        if col in LOW_CARD_NUM:
            key = df[col]
        else:
            key = pd.qcut(df[col], q=50, duplicates="drop").apply(lambda iv: iv.mid)
        grp = y.groupby(key)
        stats = pd.DataFrame({"rate": grp.mean(), "n": grp.size()}).reset_index()
        plt.plot(stats[col].astype(float), stats["rate"], marker="o", ms=3)
        plt.axhline(base, color="red", ls="--", lw=1)
        plt.title(f"Purchase rate by\n{col}" + ("" if col in LOW_CARD_NUM else " (50 bins)"))
        plt.xlabel(col)
        plt.ylabel("P(Yes)")
    save("08_target_rate_by_numeric.png", show)


def plot_train_test_distribution(tr, te, show=False):
    cols = NUM_COLS + CAT_COLS
    n_rows, n_cols = grid(len(cols))
    for i, col in enumerate(cols):
        plt.subplot(n_rows, n_cols, i + 1)
        if col in CAT_COLS or col in LOW_CARD_NUM:
            dist = pd.DataFrame({
                "train": tr[col].value_counts(normalize=True),
                "test": te[col].value_counts(normalize=True),
            }).sort_index()
            dist.plot(kind="bar", ax=plt.gca(), width=0.8, legend=(i == 0))
        else:
            sns.kdeplot(tr[col], label="train", fill=True, alpha=0.3)
            sns.kdeplot(te[col], label="test", fill=True, alpha=0.3)
            if i == 0:
                plt.legend()
        plt.title(f"train vs test: {col}")
        plt.xticks(rotation=45, ha="right")
    save("09_train_test_distribution.png", show)


def main():
    plot_target_distribution(train, TARGET)
    plot_categorical_histograms(train, TARGET)
    plot_numeric_histograms(train, TARGET)
    plot_numeric_log_histograms(train, TARGET)
    plot_correlation_heatmap(train, TARGET)
    plot_boxplots_by_target(train, TARGET)
    plot_target_rate_by_category(train, TARGET)
    plot_target_rate_by_numeric(train, TARGET)
    plot_train_test_distribution(train, test)


if __name__ == "__main__":
    main()
    print("finish !!")
