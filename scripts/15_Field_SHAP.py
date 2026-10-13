from project_paths import DATA_DIR, RESULTS_DIR
from project_paths import require_review, read_field_dataset
require_review("field_shap_parameters")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

import shap

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# ============================================================
# 1. File paths
# ============================================================

root_dir = str(RESULTS_DIR)
input_file = str(DATA_DIR / 'Machine-learning dataset.xlsx')

output_dir = os.path.join(
    root_dir,
    "Field_Nested_ML_results"
)

os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 2. Read data
# ============================================================

df = read_field_dataset(input_file)

df = df[
    df["Data quality tier"] == "A_strict_matched"
].copy().reset_index(drop=True)

print("Strict dataset shape:", df.shape)

# ============================================================
# 3. Variables
# ============================================================

cat_cols = [
    "Plant tissue group",
    "Morphotype"
]

num_cols = [
    "Soil pH",
    "SOM",
    "Measured soil concentration of the corresponding PFAS",
    "Chi3v",
    "MinPartialCharge",
    "TPSA",
    "ALogP",
    "GATS3c",
    "SpMin8_Bhi"
]

target = "Log PFASs concentration"

X = df[cat_cols + num_cols].copy()
y = df[target].copy()

# ============================================================
# 4. Preprocess
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols
        ),
        (
            "num",
            "passthrough",
            num_cols
        )
    ]
)

X_processed = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()

print("Processed X shape:", X_processed.shape)

# ============================================================
# 5. Final XGBoost model
# 用于 SHAP 解释；如果你有 nested CV 得到的 best params，
# 可以在这里替换
# ============================================================

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_processed, y)

# ============================================================
# 6. SHAP values
# ============================================================

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_processed)

# mean absolute SHAP for each transformed column
mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "transformed_feature": feature_names,
    "mean_abs_shap": mean_abs_shap
})

# ============================================================
# 7. Map transformed columns back to original feature names
# ============================================================

def map_to_original_feature(name):
    # categorical one-hot columns
    if name.startswith("cat__Plant tissue group_"):
        return "Plant tissue group"
    if name.startswith("cat__Morphotype_"):
        return "Morphotype"

    # numeric columns
    if name == "num__Soil pH":
        return "Soil pH"
    if name == "num__SOM":
        return "SOM"
    if name == "num__Measured soil concentration of the corresponding PFAS":
        return "Measured soil concentration of the corresponding PFAS"
    if name == "num__Chi3v":
        return "Chi3v"
    if name == "num__MinPartialCharge":
        return "MinPartialCharge"
    if name == "num__TPSA":
        return "TPSA"
    if name == "num__ALogP":
        return "ALogP"
    if name == "num__GATS3c":
        return "GATS3c"
    if name == "num__SpMin8_Bhi":
        return "SpMin8_Bhi"

    return name

shap_df["original_feature"] = shap_df["transformed_feature"].apply(map_to_original_feature)

# 对 one-hot 后的列按原始特征聚合
feature_importance = (
    shap_df.groupby("original_feature", as_index=False)["mean_abs_shap"]
    .sum()
)

# ============================================================
# 8. Display names + category classes
# ============================================================

display_name_map = {
    "Plant tissue group": "Tissue group",
    "Morphotype": "Morphotype",
    "Soil pH": "Soil pH",
    "SOM": "SOM",
    "Measured soil concentration of the corresponding PFAS": "Soil PFAS concentration",
    "Chi3v": "Chi3v",
    "MinPartialCharge": "Min partial charge",
    "TPSA": "TPSA",
    "ALogP": "ALogP",
    "GATS3c": "GATS3c",
    "SpMin8_Bhi": "SpMin8_Bhi"
}

class_map = {
    "Plant tissue group": "Plant",
    "Morphotype": "Plant",

    "Soil pH": "Environment",
    "SOM": "Environment",

    "Measured soil concentration of the corresponding PFAS": "Exposure",

    "Chi3v": "Molecular",
    "MinPartialCharge": "Molecular",
    "TPSA": "Molecular",
    "ALogP": "Molecular",
    "GATS3c": "Molecular",
    "SpMin8_Bhi": "Molecular"
}

feature_importance["display_name"] = feature_importance["original_feature"].map(display_name_map)
feature_importance["class"] = feature_importance["original_feature"].map(class_map)

# 排序：从高到低
feature_importance = feature_importance.sort_values(
    "mean_abs_shap",
    ascending=False
).reset_index(drop=True)

print(feature_importance)

# Preserve precise model-derived values for the lab/field comparison plot.
feature_importance.to_csv(
    os.path.join(output_dir, "Field_aggregated_SHAP_importance.csv"),
    index=False, encoding="utf-8-sig"
)

# ============================================================
# 9. Colors
# ============================================================

color_map = {
    "Plant": "#2E8B57",         # green
    "Environment": "#2C7FB8",   # blue
    "Exposure": "#7A4FA3",      # purple
    "Molecular": "#D95F02"      # orange
}

feature_importance["color"] = feature_importance["class"].map(color_map)

# ============================================================
# 10. Plot style
# ============================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "font.weight": "bold",

    "axes.labelsize": 18,
    "axes.labelweight": "bold",

    "xtick.labelsize": 12,
    "ytick.labelsize": 14,

    "axes.linewidth": 1.3,

    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,

    "xtick.major.size": 6,
    "ytick.major.size": 6,

    "xtick.direction": "out",
    "ytick.direction": "out"
})

# ============================================================
# 11. Create figure
# ============================================================

fig, ax = plt.subplots(figsize=(7.5, 5.2))

# 让最大值排在最上面
plot_df = feature_importance.iloc[::-1].reset_index(drop=True)
y_pos = np.arange(len(plot_df))

# 竖向浅灰色虚线网格
ax.set_axisbelow(True)
ax.xaxis.grid(
    True,
    linestyle="--",
    linewidth=0.8,
    color="#d9d9d9"
)
ax.yaxis.grid(False)

# lollipop lines
for i, row in plot_df.iterrows():
    ax.hlines(
        y=i,
        xmin=0,
        xmax=row["mean_abs_shap"],
        color=row["color"],
        linewidth=2.0,
        alpha=0.25,
        zorder=1
    )

# points
ax.scatter(
    plot_df["mean_abs_shap"],
    y_pos,
    s=135,
    c=plot_df["color"],
    edgecolors="none",
    zorder=3
)

# y labels
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df["display_name"], fontweight="bold")

# x label
ax.set_xlabel("Mean absolute SHAP value", fontsize=18, fontweight="bold")

# x range
xmax = plot_df["mean_abs_shap"].max() * 1.10
ax.set_xlim(0, xmax)

# clean frame
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for tick in ax.get_xticklabels():
    tick.set_fontweight("bold")
for tick in ax.get_yticklabels():
    tick.set_fontweight("bold")

ax.tick_params(axis="x", pad=6)
ax.tick_params(axis="y", pad=6)

# ============================================================
# 12. Legend
# ============================================================

legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markersize=9,
        markerfacecolor=color_map["Plant"],
        markeredgecolor="none",
        label="Plant"
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markersize=9,
        markerfacecolor=color_map["Environment"],
        markeredgecolor="none",
        label="Environment"
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markersize=9,
        markerfacecolor=color_map["Exposure"],
        markeredgecolor="none",
        label="Exposure"
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markersize=9,
        markerfacecolor=color_map["Molecular"],
        markeredgecolor="none",
        label="Molecular"
    )
]

legend = ax.legend(
    handles=legend_handles,
    loc="lower right",
    frameon=False,
    fontsize=12,
    handletextpad=0.6,
    labelspacing=0.7
)

for text in legend.get_texts():
    text.set_fontweight("bold")

# ============================================================
# 13. Layout
# ============================================================

plt.subplots_adjust(
    left=0.28,
    right=0.97,
    bottom=0.15,
    top=0.97
)

# ============================================================
# 14. Save
# ============================================================

tiff_file = os.path.join(output_dir, "Fig_Field_SHAP_importance.tiff")
pdf_file = os.path.join(output_dir, "Fig_Field_SHAP_importance.pdf")
png_file = os.path.join(output_dir, "Fig_Field_SHAP_importance.png")

plt.savefig(tiff_file, dpi=600, bbox_inches="tight")
plt.savefig(pdf_file, bbox_inches="tight")
plt.savefig(png_file, dpi=300, bbox_inches="tight")

plt.show()

print("\nFigure saved to:")
print(tiff_file)
print(pdf_file)
print(png_file)
