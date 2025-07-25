# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from functools import reduce
import processing

# %%
sns.set_theme(font_scale=0.7)

dir_list = os.listdir("datasets")
datasets = [f for f in dir_list if f.endswith(".xlsx")]
print("Available datasets:", datasets)

# %%
df_list = []
for path in datasets:
    df = pd.read_excel(f"datasets/{path}")
    print(f"Dataset: {path}, Shape: {df.shape}")
    df_list.append(df)
df = reduce(
    lambda left, right: pd.merge(left, right, on="Subscriber_ID", how="inner"), df_list
)
df.head()
df.to_csv("datasets/df_input.csv", index=False)
# %%
df = processing.missing_values(df)
print(df.isna().sum())

# %% feature engineering

df = processing.feature_engineering(df)
df.to_csv("datasets/df_engineered.csv", index=False)
# %%
target = "Customer_Churn_Flag"
X = df.drop(columns=[target, "Subscriber_ID"]).copy()
# Identify categorical columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
# %%
# visualize numerical features (box plots)
df_num_cols = X[num_cols]
cols = 3
rows = -(-len(num_cols) // cols)  # round up
fig_size = (20, 10)
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=fig_size)
for ax, col in zip(axes.flatten(), df_num_cols.columns):
    sns.boxplot(x=df_num_cols[col], color="#d136c7", ax=ax)
fig.suptitle("Box plots of numerical features", fontsize=16)
fig.tight_layout()
# fig.delaxes(axes[1, 2])
fig.savefig("plots/box_plots_numerical_features.png", dpi=300, bbox_inches="tight")
# %% # Visualize distribution of numerical features
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=fig_size)
for ax, col in zip(axes.flatten(), df_num_cols.columns):
    sns.histplot(x=df_num_cols[col], ax=ax, color="#d136c7")
fig.suptitle("Distribution of numerical attributes", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Prevent title overlap
plt.show()
fig.savefig("plots/histograms_numerical_features.png", dpi=300, bbox_inches="tight")


# %%
def plot_churn_hist_grid(
    df, numeric_features, target=target, bins=10, nrows=5, ncols=3
):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3 * nrows))
    axes = axes.flatten()

    for idx, feature in enumerate(numeric_features):
        ax1 = axes[idx]

        temp = df.copy()
        temp["bin"] = pd.cut(temp[feature], bins=bins)

        churn_rate = temp.groupby("bin")[target].mean()
        counts = temp.groupby("bin")[feature].count()

        # Bar plot (counts)
        mids = [interval.mid for interval in churn_rate.index.categories]
        ax1.bar(mids, counts, width=(mids[1] - mids[0]) * 0.9, color="#d136c7")
        ax1.set_title(feature)
        ax1.tick_params(axis="x", rotation=45)
        ax1.set_ylabel("Count", color="#d136c7")
        ax1.tick_params(axis="y", labelcolor="#d136c7")

        # Churn rate (line plot)
        ax2 = ax1.twinx()
        ax2.plot(mids, churn_rate, color="#f47c36", marker="o")
        ax2.set_ylabel("Churn Rate", color="#f47c36")
        ax2.tick_params(axis="y", labelcolor="#f47c36")

    plt.tight_layout()
    fig.savefig("plots/churn_hist_grid.png", dpi=300, bbox_inches="tight")


plot_churn_hist_grid(df, num_cols, nrows=cols, ncols=rows)


# %% Visualize churn by categorical features
cols = 3
rows = -(-len(cat_cols) // cols)  # round up
fig, axarr = plt.subplots(rows, cols, figsize=fig_size)
for ind, item in enumerate(cat_cols):
    sns.countplot(
        x=item,
        hue=target,
        data=df,
        ax=axarr[ind % rows][ind // rows],
        palette=["#d136c7", "#f47c36"],
    )
fig.subplots_adjust(hspace=0.7)
fig.suptitle("Churn by Categorical Features", fontsize=16)
fig.savefig("plots/churn_by_categorical_features.png", dpi=300, bbox_inches="tight")
