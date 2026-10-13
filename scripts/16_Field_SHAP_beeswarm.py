from project_paths import DATA_DIR, RESULTS_DIR
from project_paths import require_review, read_field_dataset
require_review("field_shap_parameters")
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import numpy as np
import pandas as pd

import shap

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# ============================================================
# 1. Paths
# ============================================================

input_dir = str(RESULTS_DIR)

output_dir = os.path.join(
    input_dir,
    "Field_Nested_ML_results"
)

os.makedirs(output_dir, exist_ok=True)


input_file = str(DATA_DIR / 'Machine-learning dataset.xlsx')


# ============================================================
# 2. Read field dataset
# ============================================================

df = read_field_dataset(input_file)


df = df[
    df["Data quality tier"] == "A_strict_matched"
].copy().reset_index(drop=True)


print("Field strict dataset:")
print(df.shape)



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

    "SpMin8_Bhi",
]


target = "Log PFASs concentration"


X = df[cat_cols + num_cols].copy()

y = df[target].copy()



# ============================================================
# 4. Preprocessing
# ============================================================


preprocessor = ColumnTransformer(

    transformers=[

        (
            "cat",

            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),

            cat_cols
        ),


        (
            "num",

            "passthrough",

            num_cols
        )

    ]

)



X_transformed = preprocessor.fit_transform(X)


feature_names = (
    preprocessor
    .get_feature_names_out()
)


print(
    "Transformed matrix:",
    X_transformed.shape
)



# ============================================================
# 5. Final XGBoost model
#
# 这里保持与你 nested modeling 最优模型一致
# 如果你后面有最佳参数，可以替换
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



model.fit(
    X_transformed,
    y
)



# ============================================================
# 6. SHAP values
# ============================================================


explainer = shap.TreeExplainer(
    model
)


shap_values = explainer.shap_values(
    X_transformed
)



# ============================================================
# 7. Select continuous variables only
# ============================================================


selected_features = [

    "Soil pH",

    "SOM",

    "Measured soil concentration of the corresponding PFAS",

    "Chi3v",

    "MinPartialCharge",

    "TPSA",

    "ALogP",

    "GATS3c",

    "SpMin8_Bhi",

]



# locate numerical columns

selected_indices = []


for feature in selected_features:

    target_name = "num__" + feature

    idx = list(feature_names).index(
        target_name
    )

    selected_indices.append(idx)



X_selected = X_transformed[:, selected_indices]


SHAP_selected = shap_values[:, selected_indices]



# ============================================================
# 8. Rank by mean absolute SHAP
# ============================================================


mean_abs_SHAP = (
    np.abs(SHAP_selected)
    .mean(axis=0)
)


importance_df = pd.DataFrame({

    "Feature":
        selected_features,

    "Mean_abs_SHAP":
        mean_abs_SHAP

})


importance_df = (
    importance_df
    .sort_values(
        "Mean_abs_SHAP",
        ascending=False
    )
)



ordered_features = (
    importance_df["Feature"]
    .tolist()
)



print("\nFeature order:")

for f in ordered_features:
    print(f)



# reorder arrays

ordered_indices = [

    selected_features.index(f)

    for f in ordered_features

]


SHAP_plot = (
    SHAP_selected[:, ordered_indices]
)


X_plot = (
    X_selected[:, ordered_indices]
)



# ============================================================
# 9. Display labels
# ============================================================


display_labels = {


    "Soil pH":
        "Soil pH",


    "SOM":
        "SOM",


    "Measured soil concentration of the corresponding PFAS":
        "Soil PFAS concentration",


    "Chi3v":
        "Chi3v",


    "MinPartialCharge":
        "Min partial charge",


    "TPSA":
        "TPSA",


    "ALogP":
        "ALogP",


    "GATS3c":
        "GATS3c",


    "SpMin8_Bhi":
        "SpMin8_Bhi",

}




# ============================================================
# 10. Export plotting data
# ============================================================


long_tables = []


for i, feature in enumerate(ordered_features):

    long_tables.append(

        pd.DataFrame({

            "Feature":

                feature,

            "Feature_value":

                X_plot[:, i],

            "SHAP_value":

                SHAP_plot[:, i]

        })

    )



long_df = pd.concat(
    long_tables,
    ignore_index=True
)



long_df.to_csv(

    os.path.join(
        output_dir,
        "Field_SHAP_beeswarm_plot_data.csv"
    ),

    index=False

)



# ============================================================
# 11. Publication style
# ============================================================


mpl.rcParams.update({

    "font.family":
        "Arial",

    "font.size":
        13,

    "font.weight":
        "bold",


    "axes.labelsize":
        18,


    "axes.labelweight":
        "bold",


    "xtick.labelsize":
        14,


    "ytick.labelsize":
        15,


    "axes.linewidth":
        1.4,


    "xtick.major.width":
        1.4,


    "ytick.major.width":
        1.4,


    "xtick.major.size":
        7,


    "ytick.major.size":
        7,


    "xtick.direction":
        "out",


    "ytick.direction":
        "out",


    "pdf.fonttype":
        42,


    "ps.fonttype":
        42,


    "svg.fonttype":
        "none",

})



cmap = mpl.colormaps["coolwarm"]


rng = np.random.default_rng(2026)



# ============================================================
# 12. Draw SHAP beeswarm
# ============================================================


fig, ax = plt.subplots(

    figsize=(8.4, 5.9)

)



n_features = len(ordered_features)



for row_index, feature in enumerate(ordered_features):


    y_base = (
        n_features - 1 - row_index
    )


    shap_feature = (
        SHAP_plot[:, row_index]
    )


    feature_values = (
        X_plot[:, row_index]
    )



    low, high = np.nanpercentile(

        feature_values,

        [2, 98]

    )



    if np.isclose(low, high):

        normalized_values = np.full_like(

            feature_values,

            0.5,

            dtype=float

        )

    else:

        normalized_values = np.clip(

            (feature_values-low)
            /
            (high-low),

            0,

            1

        )



    jitter = rng.normal(

        loc=0,

        scale=0.10,

        size=len(shap_feature)

    )



    ax.scatter(

        shap_feature,

        y_base + jitter,


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





# zero line

ax.axvline(

    x=0,

    color="#6B6B6B",

    linestyle="--",

    linewidth=1.5,

    zorder=1

)



# ============================================================
# 13. Axes
# ============================================================


all_values = SHAP_plot.flatten()


xmin = np.nanmin(all_values)

xmax = np.nanmax(all_values)



padding = max(

    (xmax-xmin)*0.08,

    0.03

)



ax.set_xlim(

    xmin-padding,

    xmax+padding

)



ax.set_yticks(

    np.arange(n_features)

)



ax.set_yticklabels(

    [

        display_labels[f]

        for f in ordered_features[::-1]

    ],

    fontsize=17,

    fontweight="bold"

)



ax.set_xlabel(

    "SHAP value",

    fontsize=20,

    fontweight="bold"

)


ax.set_ylabel("")



ax.set_ylim(

    -0.55,

    n_features-0.45

)



ax.grid(False)



ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)



ax.tick_params(

    axis="y",

    labelsize=17,

    width=1.5,

    length=7,

    pad=7

)



ax.tick_params(

    axis="x",

    labelsize=15,

    width=1.5,

    length=7,

    pad=7

)



for tick in ax.get_xticklabels():

    tick.set_fontweight("bold")



for tick in ax.get_yticklabels():

    tick.set_fontweight("bold")



# ============================================================
# 14. Colorbar
# ============================================================


sm = ScalarMappable(

    norm=Normalize(

        vmin=0,

        vmax=1

    ),

    cmap=cmap

)


sm.set_array([])



cax = fig.add_axes(

    [

        0.79,

        0.38,

        0.018,

        0.25

    ]

)



cbar = fig.colorbar(

    sm,

    cax=cax

)



cbar.set_ticks(

    [0,1]

)


cbar.set_ticklabels(

    ["Low","High"]

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

    tick.set_fontweight("bold")



# ============================================================
# 15. Layout
# ============================================================


fig.subplots_adjust(

    left=0.30,

    right=0.74,

    top=0.95,

    bottom=0.16

)



# ============================================================
# 16. Save
# ============================================================


output_stem = os.path.join(

    output_dir,

    "Fig_Field_SHAP_beeswarm_optimized"

)



fig.savefig(

    output_stem+".svg",

    bbox_inches="tight",

    facecolor="white"

)


fig.savefig(

    output_stem+".pdf",

    bbox_inches="tight",

    facecolor="white"

)


fig.savefig(

    output_stem+".tiff",

    dpi=600,

    bbox_inches="tight",

    facecolor="white",

    pil_kwargs={
        "compression":"tiff_lzw"
    }

)


fig.savefig(

    output_stem+".png",

    dpi=300,

    bbox_inches="tight",

    facecolor="white"

)



print("\nSaved files:")

for suffix in [

    ".svg",

    ".pdf",

    ".tiff",

    ".png"

]:

    print(output_stem + suffix)



plt.show()

plt.close(fig)
