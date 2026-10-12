from project_paths import DATA_DIR, RESULTS_DIR
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

input_dir = str(RESULTS_DIR)

output_dir = os.environ.get(
    "PFAS_FIG_OUTPUT_DIR",
    input_dir
)

os.makedirs(output_dir, exist_ok=True)

shap_file = os.path.join(
    input_dir,
    "Final_XGBoost_raw_SHAP_values.csv"
)

X_file = os.path.join(
    input_dir,
    "Final_XGBoost_transformed_X.csv"
)

importance_file = os.path.join(
    input_dir,
    "Overall_aggregated_SHAP_importance.csv"
)


# ============================================================
# 2. Read data
# ============================================================

for file_path in [
    shap_file,
    X_file,
    importance_file
]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Required input file not found:\n{file_path}"
        )

raw_shap = pd.read_csv(shap_file)
X_transformed = pd.read_csv(X_file)
importance_df = pd.read_csv(importance_file)

print("SHAP shape:", raw_shap.shape)
print("Transformed X shape:", X_transformed.shape)


# ============================================================
# 3. Check importance table
# ============================================================

required_columns = {
    "Feature",
    "Mean_abs_SHAP"
}

if not required_columns.issubset(
    importance_df.columns
):
    raise ValueError(
        "Overall_aggregated_SHAP_importance.csv must contain "
        "'Feature' and 'Mean_abs_SHAP'."
    )


# ============================================================
# 4. Define all variables shown in combined SHAP plot
# ============================================================

feature_info = [

    {
        "display_name": "SOM",
        "model_name": "SOM",
        "importance_name": "SOM"
    },

    {
        "display_name": "PFAS concentration",
        "model_name": "Exposure Concentration",
        "importance_name": "Exposure concentration"
    },

    {
        "display_name": "Soil pH",
        "model_name": "Soil pH",
        "importance_name": "Soil pH"
    },

    {
        "display_name": "Growth temperature",
        "model_name": "Growth Temperature",
        "importance_name": "Growth temperature"
    },

    {
        "display_name": "Growth duration",
        "model_name": "Growth Duration",
        "importance_name": "Growth duration"
    },

    {
        "display_name": "Chi3v",
        "model_name": "Chi3v",
        "importance_name": "Chi3v"
    },

    {
        "display_name": "Min partial charge",
        "model_name": "MinPartialCharge",
        "importance_name": "MinPartialCharge"
    },

    {
        "display_name": "TPSA",
        "model_name": "TPSA",
        "importance_name": "TPSA"
    },

    {
        "display_name": "ALogP",
        "model_name": "ALogP",
        "importance_name": "ALogP"
    },

    {
        "display_name": "GATS3c",
        "model_name": "GATS3c",
        "importance_name": "GATS3c"
    },

    {
        "display_name": "SpMin8_Bhi",
        "model_name": "SpMin8_Bhi",
        "importance_name": "SpMin8_Bhi"
    },
]


# ============================================================
# 5. Check that all model variables exist
# ============================================================

missing_features = []

for info in feature_info:

    model_name = info["model_name"]

    if model_name not in raw_shap.columns:
        missing_features.append(
            f"{model_name} missing from SHAP"
        )

    if model_name not in X_transformed.columns:
        missing_features.append(
            f"{model_name} missing from X"
        )

if missing_features:
    raise ValueError(
        "\n".join(missing_features)
    )


# ============================================================
# 6. Merge with SHAP importance and rank ALL features together
# ============================================================

feature_meta = pd.DataFrame(
    feature_info
)

combined_importance = feature_meta.merge(

    importance_df[
        [
            "Feature",
            "Mean_abs_SHAP"
        ]
    ],

    left_on="importance_name",

    right_on="Feature",

    how="left"
)


if combined_importance[
    "Mean_abs_SHAP"
].isna().any():

    missing = combined_importance.loc[
        combined_importance[
            "Mean_abs_SHAP"
        ].isna(),
        "importance_name"
    ].tolist()

    raise ValueError(
        "Features missing from importance table:\n"
        f"{missing}"
    )


# ------------------------------------------------------------
# Global ranking across ALL features
# ------------------------------------------------------------

combined_importance = (

    combined_importance

    .sort_values(
        "Mean_abs_SHAP",
        ascending=False
    )

    .reset_index(
        drop=True
    )
)


print(
    "\nCombined SHAP importance ranking:"
)

for i, row in combined_importance.iterrows():

    print(
        f"{i + 1:2d}. "
        f"{row['display_name']:<24s} "
        f"{row['Mean_abs_SHAP']:.5f}"
    )


# ============================================================
# 7. Export final ranking
# ============================================================

ranking_df = combined_importance[
    [
        "display_name",
        "model_name",
        "Mean_abs_SHAP"
    ]
].copy()

ranking_df.columns = [
    "Feature",
    "Model_feature",
    "Mean_abs_SHAP"
]

ranking_df.insert(
    0,
    "Rank",
    np.arange(
        1,
        len(ranking_df) + 1
    )
)

ranking_file = os.path.join(
    output_dir,
    "Fig3c_Combined_SHAP_feature_ranking.csv"
)

ranking_df.to_csv(
    ranking_file,
    index=False
)


# ============================================================
# 8. Prepare long-format source data
# ============================================================

long_tables = []

for _, row in combined_importance.iterrows():

    display_name = row[
        "display_name"
    ]

    model_name = row[
        "model_name"
    ]

    mean_abs_shap = row[
        "Mean_abs_SHAP"
    ]

    temp_df = pd.DataFrame({

        "Feature":
            display_name,

        "Model_feature":
            model_name,

        "Mean_abs_SHAP":
            mean_abs_shap,

        "Feature_value":
            X_transformed[
                model_name
            ].to_numpy(),

        "SHAP_value":
            raw_shap[
                model_name
            ].to_numpy()

    })

    long_tables.append(
        temp_df
    )


long_df = pd.concat(
    long_tables,
    ignore_index=True
)


plot_data_file = os.path.join(
    output_dir,
    "Fig3c_Combined_SHAP_plot_data.csv"
)

long_df.to_csv(
    plot_data_file,
    index=False
)


# ============================================================
# 9. Publication style
# ============================================================

mpl.rcParams.update({

    "font.family": "Arial",

    "font.size": 13,

    "font.weight": "bold",

    "axes.labelsize": 18,

    "axes.labelweight": "bold",

    "xtick.labelsize": 14,

    "ytick.labelsize": 15,

    "axes.linewidth": 1.4,

    "xtick.major.width": 1.4,

    "ytick.major.width": 1.4,

    "xtick.major.size": 7,

    "ytick.major.size": 7,

    "xtick.direction": "out",

    "ytick.direction": "out",

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

    "svg.fonttype": "none"
})


# ============================================================
# 10. Colormap
# ============================================================

cmap = mpl.colormaps[
    "coolwarm"
]

rng = np.random.default_rng(
    2026
)


# ============================================================
# 11. Create figure
# ============================================================

n_features = len(
    combined_importance
)

fig, ax = plt.subplots(
    figsize=(8.4, 7.2)
)


# ============================================================
# 12. SHAP = 0 line
# ============================================================

ax.axvline(

    x=0,

    color="#6B6B6B",

    linestyle="--",

    linewidth=1.5,

    zorder=1
)


# ============================================================
# 13. Draw unified SHAP beeswarm
# ============================================================

for row_index, row in combined_importance.iterrows():

    # Most important feature appears at top
    y_base = (
        n_features
        - 1
        - row_index
    )

    model_name = row[
        "model_name"
    ]


    shap_values = raw_shap[
        model_name
    ].to_numpy(
        dtype=float
    )

    feature_values = X_transformed[
        model_name
    ].to_numpy(
        dtype=float
    )


    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    valid_mask = (
        np.isfinite(
            shap_values
        )
        &
        np.isfinite(
            feature_values
        )
    )

    shap_values = shap_values[
        valid_mask
    ]

    feature_values = feature_values[
        valid_mask
    ]


    # --------------------------------------------------------
    # Normalize feature values independently
    # --------------------------------------------------------

    low, high = np.nanpercentile(
        feature_values,
        [2, 98]
    )


    if np.isclose(
        high,
        low
    ):

        normalized_values = np.full(
            len(feature_values),
            0.5,
            dtype=float
        )

    else:

        normalized_values = (
            feature_values
            - low
        ) / (
            high
            - low
        )

        normalized_values = np.clip(
            normalized_values,
            0,
            1
        )


    # --------------------------------------------------------
    # Vertical jitter
    # --------------------------------------------------------

    jitter = rng.normal(

        loc=0,

        scale=0.10,

        size=len(
            shap_values
        )
    )


    # --------------------------------------------------------
    # Scatter
    # --------------------------------------------------------

    ax.scatter(

        shap_values,

        y_base
        + jitter,

        s=28,

        c=normalized_values,

        cmap=cmap,

        vmin=0,

        vmax=1,

        alpha=0.55,

        edgecolors="none",

        rasterized=True,

        zorder=3
    )


# ============================================================
# 14. X-axis range
# ============================================================

all_shap_values = long_df[
    "SHAP_value"
].to_numpy(
    dtype=float
)

all_shap_values = all_shap_values[
    np.isfinite(
        all_shap_values
    )
]

xmin = np.nanmin(
    all_shap_values
)

xmax = np.nanmax(
    all_shap_values
)

x_range = (
    xmax - xmin
)

if np.isclose(
    x_range,
    0
):
    x_range = 1.0


x_padding = max(
    x_range * 0.08,
    0.03
)


ax.set_xlim(

    xmin
    - x_padding,

    xmax
    + x_padding
)


# ============================================================
# 15. Y-axis labels
# ============================================================

axis_rows = (

    combined_importance

    .iloc[::-1]

    .reset_index(
        drop=True
    )
)


ax.set_yticks(
    np.arange(
        n_features
    )
)


ax.set_yticklabels(

    axis_rows[
        "display_name"
    ].tolist(),

    fontsize=16,

    fontweight="bold"
)


# ============================================================
# 16. Axis labels
# ============================================================

ax.set_xlabel(

    "SHAP value",

    fontsize=20,

    fontweight="bold"
)

ax.set_ylabel("")


ax.set_ylim(

    -0.55,

    n_features
    - 0.45
)


# ============================================================
# 17. Frame
# ============================================================

ax.grid(
    False
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)


# ============================================================
# 18. Tick style
# ============================================================

ax.tick_params(

    axis="x",

    labelsize=15,

    width=1.5,

    length=7,

    pad=7
)


ax.tick_params(

    axis="y",

    labelsize=16,

    width=1.5,

    length=7,

    pad=8
)


for tick in ax.get_xticklabels():

    tick.set_fontweight(
        "bold"
    )


for tick in ax.get_yticklabels():

    tick.set_fontweight(
        "bold"
    )


# ============================================================
# 19. Panel label
# ============================================================

ax.text(

    -0.25,

    1.02,

    "(c)",

    transform=ax.transAxes,

    ha="left",

    va="bottom",

    fontsize=24,

    fontweight="bold"
)


# ============================================================
# 20. Feature-value colorbar ONLY
# ============================================================

sm = ScalarMappable(

    norm=Normalize(
        vmin=0,
        vmax=1
    ),

    cmap=cmap
)

sm.set_array([])


# [left, bottom, width, height]
cax = fig.add_axes(
    [
        0.82,
        0.38,
        0.020,
        0.27
    ]
)


cbar = fig.colorbar(

    sm,

    cax=cax
)


cbar.set_ticks(
    [0, 1]
)


cbar.set_ticklabels(
    [
        "Low",
        "High"
    ]
)


cbar.set_label(

    "Feature value",

    fontsize=16,

    fontweight="bold",

    labelpad=10
)


cbar.ax.tick_params(

    labelsize=14,

    width=1.3,

    length=5
)


for tick in cbar.ax.get_yticklabels():

    tick.set_fontweight(
        "bold"
    )


# ============================================================
# 21. Layout
# ============================================================

fig.subplots_adjust(

    left=0.34,

    right=0.78,

    top=0.95,

    bottom=0.13
)


# ============================================================
# 22. Export
# ============================================================

output_stem = os.path.join(

    output_dir,

    "Fig3c_Combined_SHAP_ranked"
)


fig.savefig(

    output_stem + ".svg",

    bbox_inches="tight",

    facecolor="white"
)


fig.savefig(

    output_stem + ".pdf",

    bbox_inches="tight",

    facecolor="white"
)


fig.savefig(

    output_stem + ".tiff",

    dpi=600,

    bbox_inches="tight",

    facecolor="white",

    pil_kwargs={
        "compression":
            "tiff_lzw"
    }
)


fig.savefig(

    output_stem + ".png",

    dpi=300,

    bbox_inches="tight",

    facecolor="white"
)


print(
    "\nSaved figure files:"
)

for suffix in [
    ".svg",
    ".pdf",
    ".tiff",
    ".png"
]:

    print(
        output_stem
        + suffix
    )


print(
    "\nSaved ranking table:"
)

print(
    ranking_file
)


print(
    "\nSaved plotting data:"
)

print(
    plot_data_file
)


plt.show()

plt.close(
    fig
)