# -*- coding: utf-8 -*-

"""
Field-model validation figures for Supporting Information

Output 1:
    Field_model_comparison_OOF
    (a) OOF R² comparison among four field models
    (b) OOF RMSE comparison among four field models

Output 2:
    Field_XGBoost_foldwise_performance
    (a) fold-wise R² of final XGBoost
    (b) fold-wise RMSE of final XGBoost

Notes
-----
1. Model_comparison.xlsx is NOT required.
2. The script automatically searches for:
       Field_Nested_OOF_predictions.csv
3. If a Fold column exists in the OOF file, it is used directly.
4. Otherwise, folds are reconstructed using:
       GroupKFold(n_splits=5)
       groups = Study_ID + PFAS
5. The script verifies that the loaded OOF file reproduces:
       R²   = 0.7574462342
       RMSE = 0.5446319655
"""
from project_paths import DATA_DIR, RESULTS_DIR
import json

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

# ============================================================
# 1. BASIC PATHS
# ============================================================

SCRIPT_DIR = RESULTS_DIR / 'Field_Nested_ML_results'

# Search only the selected result directory; do not mix historical copies.
SEARCH_ROOTS = [RESULTS_DIR / 'Field_Nested_ML_results']

OUT = RESULTS_DIR / 'Field_validation_SI'

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. FINAL SAVED FIELD-MODEL RESULTS
#
# These are the final overall OOF results from your previous
# four-model field comparison.
#
# No Model_comparison.xlsx is required.
# ============================================================

MODEL_RESULTS = pd.DataFrame(json.loads((RESULTS_DIR / 'Field_Nested_ML_results' / 'summary.json').read_text(encoding='utf-8'))['Results'])


# Expected final XGBoost values.
EXPECTED_XGB_R2 = float(MODEL_RESULTS.set_index('Model').loc['XGBoost', 'OOF_R2'])
EXPECTED_XGB_RMSE = float(MODEL_RESULTS.set_index('Model').loc['XGBoost', 'OOF_RMSE'])

# Loose enough for CSV floating-point roundoff,
# strict enough to avoid using the wrong OOF file.
VERIFY_ATOL = 1e-6


# ============================================================
# 3. MODEL ORDER / DISPLAY NAMES / COLORS
#
# Match your previous validation-figure colour family:
# RF blue
# boosting orange
# LightGBM green
# XGBoost red
# ============================================================

MODEL_ORDER = [
    "RF",
    "GB",
    "LightGBM",
    "XGBoost",
]

DISPLAY_NAMES = {

    "RF":
        "Random\nForest",

    "GB":
        "Gradient\nBoosting",

    "LightGBM":
        "LightGBM",

    "XGBoost":
        "XGBoost",
}

MODEL_COLORS = {

    "RF":
        "#4C93C3",

    "GB":
        "#F28E2B",

    "LightGBM":
        "#59A14F",

    "XGBoost":
        "#E15759",
}


# ============================================================
# 4. GLOBAL PLOTTING STYLE
#
# Consistent with your previous manuscript/SI figures:
# Arial
# bold
# white background
# no grid
# left/bottom spines only
# publication vector fonts
# ============================================================

plt.rcParams.update({

    "font.family":
        "Arial",

    "font.sans-serif":
        [
            "Arial",
            "DejaVu Sans",
        ],

    "font.size":
        12,

    "font.weight":
        "bold",

    "axes.labelsize":
        17,

    "axes.labelweight":
        "bold",

    "axes.titleweight":
        "bold",

    "xtick.labelsize":
        11,

    "ytick.labelsize":
        12,

    "axes.linewidth":
        1.30,

    "xtick.major.width":
        1.20,

    "ytick.major.width":
        1.20,

    "xtick.major.size":
        6,

    "ytick.major.size":
        6,

    "xtick.direction":
        "out",

    "ytick.direction":
        "out",

    "axes.grid":
        False,

    "legend.frameon":
        False,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,

    "svg.fonttype":
        "none",
})


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def calc_rmse(y_true, y_pred):

    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )
    )


def style_axis(ax):

    """
    Match previous validation SI plots:
    left + bottom axes only.
    """

    ax.grid(False)

    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#30343A")
    ax.spines["bottom"].set_color("#30343A")

    ax.spines["left"].set_linewidth(1.25)
    ax.spines["bottom"].set_linewidth(1.25)

    ax.tick_params(
        axis="both",
        direction="out",
        width=1.20,
        length=6
    )

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")


def save_figure(fig, stem):

    """
    Save four publication formats.
    """

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={
            "compression": "tiff_lzw"
        }
    )

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white"
    )


def test_oof_candidate(path):

    """
    Check whether a CSV is the correct final XGBoost OOF file.

    Returns
    -------
    tuple:
        (valid, dataframe, r2, rmse)
    """

    try:

        data = pd.read_csv(
            path
        )

    except Exception:

        return False, None, None, None


    required = {
        "Observed",
        "Predicted",
        "Study_ID",
        "PFAS",
    }


    if not required.issubset(
        data.columns
    ):

        return False, None, None, None


    data = data.dropna(
        subset=[
            "Observed",
            "Predicted",
            "Study_ID",
            "PFAS",
        ]
    ).copy()


    try:

        data["Observed"] = pd.to_numeric(
            data["Observed"],
            errors="raise"
        )

        data["Predicted"] = pd.to_numeric(
            data["Predicted"],
            errors="raise"
        )

    except Exception:

        return False, None, None, None


    current_r2 = r2_score(
        data["Observed"],
        data["Predicted"]
    )

    current_rmse = calc_rmse(
        data["Observed"],
        data["Predicted"]
    )


    is_correct = (

        np.isclose(
            current_r2,
            EXPECTED_XGB_R2,
            atol=VERIFY_ATOL,
            rtol=0
        )

        and

        np.isclose(
            current_rmse,
            EXPECTED_XGB_RMSE,
            atol=VERIFY_ATOL,
            rtol=0
        )

    )


    return (
        is_correct,
        data,
        current_r2,
        current_rmse
    )


def find_oof_file():

    """
    Automatically locate the correct
    Field_Nested_OOF_predictions.csv.

    If multiple copies exist, identify the correct one by
    reproducing the known final XGBoost OOF metrics.
    """

    filename = "Field_Nested_OOF_predictions.csv"

    candidates = []

    seen = set()


    # --------------------------------------------------------
    # Exact same-folder candidate first
    # --------------------------------------------------------

    direct = (
        SCRIPT_DIR
        /
        filename
    )

    if direct.exists():

        candidates.append(
            direct
        )

        seen.add(
            str(
                direct.resolve()
            )
        )


    # --------------------------------------------------------
    # Recursive search
    # --------------------------------------------------------

    for root in SEARCH_ROOTS:

        if not root.exists():
            continue


        try:

            found = list(
                root.rglob(
                    filename
                )
            )

        except PermissionError:

            found = []


        for path in found:

            resolved = str(
                path.resolve()
            )

            if resolved not in seen:

                candidates.append(
                    path
                )

                seen.add(
                    resolved
                )


    # --------------------------------------------------------
    # Nothing found
    # --------------------------------------------------------

    if len(candidates) == 0:

        raise FileNotFoundError(

            "\n\n"
            "Could not find:\n"
            "Field_Nested_OOF_predictions.csv\n\n"
            "Please copy this CSV into the same folder as:\n"
            f"{Path(__file__).name}\n"
            "\n"
            f"Current script folder:\n{SCRIPT_DIR}\n"
        )


    # --------------------------------------------------------
    # Test candidates
    # --------------------------------------------------------

    tested = []


    for path in candidates:

        (
            valid,
            data,
            current_r2,
            current_rmse

        ) = test_oof_candidate(
            path
        )


        tested.append(
            (
                path,
                current_r2,
                current_rmse
            )
        )


        if valid:

            print(
                "\n============================================"
            )

            print(
                "Correct OOF file automatically identified:"
            )

            print(
                path
            )

            print(
                "============================================"
            )

            print(
                f"N = {len(data)}"
            )

            print(
                f"Overall OOF R2   = {current_r2:.9f}"
            )

            print(
                f"Overall OOF RMSE = {current_rmse:.9f}"
            )


            return path, data


    # --------------------------------------------------------
    # Files were found, but none matched final model
    # --------------------------------------------------------

    print(
        "\nCandidate OOF files found:"
    )


    for path, current_r2, current_rmse in tested:

        print(
            "\n",
            path
        )

        print(
            "R2   =",
            current_r2
        )

        print(
            "RMSE =",
            current_rmse
        )


    raise RuntimeError(

        "\nA Field_Nested_OOF_predictions.csv file was found, "
        "but none reproduced the final XGBoost metrics:\n"
        f"Expected R2   = {EXPECTED_XGB_R2:.9f}\n"
        f"Expected RMSE = {EXPECTED_XGB_RMSE:.9f}\n\n"
        "This prevents the wrong prediction file from being "
        "used silently."
    )


# ============================================================
# 6. FIND AND VERIFY OOF FILE
# ============================================================

OOF_FILE, oof = find_oof_file()


# ============================================================
# 7. PRINT FINAL MODEL COMPARISON
# ============================================================

model_df = (

    MODEL_RESULTS
    .set_index("Model")
    .loc[MODEL_ORDER]
    .reset_index()

)


print(
    "\n\n============================================"
)

print(
    "Final field-model OOF comparison"
)

print(
    "============================================"
)


print(

    model_df[
        [
            "Model",
            "OOF_R2",
            "OOF_RMSE",
            "OOF_MAE",
        ]
    ].to_string(
        index=False
    )

)


# ============================================================
# 8. FIGURE 1
#    FOUR-MODEL OOF COMPARISON
#

# ============================================================

fig1, (
    ax_r2,
    ax_rmse

) = plt.subplots(

    1,
    2,

    figsize=(
        9.6,
        4.25
    )
)


x = np.arange(
    len(MODEL_ORDER)
)


# ------------------------------------------------------------
# 8A. R² PANEL
# ------------------------------------------------------------

for i, model in enumerate(
    MODEL_ORDER
):


    value = float(

        model_df.loc[
            model_df[
                "Model"
            ].eq(
                model
            ),
            "OOF_R2"

        ].iloc[0]

    )


    color = MODEL_COLORS[
        model
    ]


    # Light vertical guide
    ax_r2.plot(

        [
            i,
            i
        ],

        [
            0.60,
            value
        ],

        color="#D0D0D0",

        linewidth=1.35,

        zorder=1
    )


    # Main OOF diamond
    ax_r2.scatter(

        i,

        value,

        s=95,

        marker="D",

        facecolor=color,

        edgecolor="#263238",

        linewidth=1.0,

        zorder=3
    )


    # Exact value
    ax_r2.text(

        i,

        value + 0.010,

        f"{value:.3f}",

        ha="center",

        va="bottom",

        fontsize=10.5,

        fontweight="bold"

    )


ax_r2.set_xticks(
    x
)


ax_r2.set_xticklabels(

    [
        DISPLAY_NAMES[
            model
        ]
        for model
        in MODEL_ORDER
    ],

    fontsize=10.5,

    fontweight="bold",

    linespacing=0.90
)


ax_r2.set_ylabel(

    r"OOF $R^2$",

    fontsize=17,

    fontweight="bold"
)


ax_r2.set_ylim(
    0.60,
    0.79
)


ax_r2.set_yticks(
    [
        0.60,
        0.65,
        0.70,
        0.75,
    ]
)


ax_r2.text(

    -0.14,

    1.035,

    "(a)",

    transform=ax_r2.transAxes,

    fontsize=18,

    fontweight="bold",

    ha="left",

    va="bottom"
)


style_axis(
    ax_r2
)


# ------------------------------------------------------------
# 8B. RMSE PANEL
# ------------------------------------------------------------

for i, model in enumerate(
    MODEL_ORDER
):


    value = float(

        model_df.loc[
            model_df[
                "Model"
            ].eq(
                model
            ),
            "OOF_RMSE"

        ].iloc[0]

    )


    color = MODEL_COLORS[
        model
    ]


    ax_rmse.plot(

        [
            i,
            i
        ],

        [
            0.50,
            value
        ],

        color="#D0D0D0",

        linewidth=1.35,

        zorder=1
    )


    ax_rmse.scatter(

        i,

        value,

        s=95,

        marker="D",

        facecolor=color,

        edgecolor="#263238",

        linewidth=1.0,

        zorder=3
    )


    ax_rmse.text(

        i,

        value + 0.008,

        f"{value:.3f}",

        ha="center",

        va="bottom",

        fontsize=10.5,

        fontweight="bold"

    )


ax_rmse.set_xticks(
    x
)


ax_rmse.set_xticklabels(

    [
        DISPLAY_NAMES[
            model
        ]
        for model
        in MODEL_ORDER
    ],

    fontsize=10.5,

    fontweight="bold",

    linespacing=0.90
)


ax_rmse.set_ylabel(

    "OOF RMSE",

    fontsize=17,

    fontweight="bold"
)


ax_rmse.set_ylim(
    0.50,
    0.65
)


ax_rmse.set_yticks(
    [
        0.50,
        0.55,
        0.60,
        0.65,
    ]
)


ax_rmse.text(

    -0.14,

    1.035,

    "(b)",

    transform=ax_rmse.transAxes,

    fontsize=18,

    fontweight="bold",

    ha="left",

    va="bottom"
)


style_axis(
    ax_rmse
)


fig1.subplots_adjust(

    left=0.085,

    right=0.985,

    bottom=0.22,

    top=0.92,

    wspace=0.31
)


FIG1_STEM = (
    OUT
    /
    "Field_model_comparison_OOF"
)


save_figure(
    fig1,
    FIG1_STEM
)


# ============================================================
# 9. PREPARE XGBOOST OOF DATA
# ============================================================

oof = oof.copy()


# Drop common exported CSV index columns if present
drop_cols = [

    c

    for c in oof.columns

    if str(c).lower().startswith(
        "unnamed:"
    )

]


if drop_cols:

    oof = oof.drop(
        columns=drop_cols
    )


oof["Observed"] = pd.to_numeric(
    oof["Observed"],
    errors="raise"
)


oof["Predicted"] = pd.to_numeric(
    oof["Predicted"],
    errors="raise"
)


oof["Study_ID"] = (
    oof["Study_ID"]
    .astype(str)
    .str.strip()
)


oof["PFAS"] = (
    oof["PFAS"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 10. CREATE GROUP VARIABLE
#
# This matches your field-model grouping:
# Study_ID + PFAS
# ============================================================

oof["Group"] = (

    oof["Study_ID"]

    +

    "__"

    +

    oof["PFAS"]

)


print(
    "\nN OOF observations =",
    len(oof)
)


print(
    "Number of Study_ID + PFAS groups =",
    oof["Group"].nunique()
)


# ============================================================
# 11. OBTAIN FOLD LABELS
#
# Priority:
# 1. Existing Fold column
# 2. Existing fold column
# 3. Reconstruct using GroupKFold(5)
# ============================================================

existing_fold_col = None


for candidate in [

    "Fold",
    "fold",
    "Outer_Fold",
    "outer_fold",
    "CV_Fold",
    "cv_fold",

]:

    if candidate in oof.columns:

        existing_fold_col = candidate

        break


if existing_fold_col is not None:

    print(
        "\nUsing existing fold column:",
        existing_fold_col
    )


    oof["Fold"] = pd.to_numeric(

        oof[
            existing_fold_col
        ],

        errors="raise"

    ).astype(int)


else:

    print(
        "\nNo saved Fold column found."
    )

    print(
        "Reconstructing folds with "
        "GroupKFold(n_splits=5), "
        "groups = Study_ID + PFAS."
    )


    N_SPLITS = 5


    if (
        oof["Group"].nunique()
        <
        N_SPLITS
    ):

        raise ValueError(

            "Not enough unique groups for "
            "5-fold GroupKFold."

        )


    splitter = GroupKFold(
        n_splits=N_SPLITS
    )


    fold_array = np.full(

        len(oof),

        -1,

        dtype=int

    )


    dummy_X = np.zeros(
        (
            len(oof),
            1
        )
    )


    for fold_number, (
        train_index,
        test_index

    ) in enumerate(

        splitter.split(

            dummy_X,

            oof[
                "Observed"
            ].to_numpy(),

            groups=oof[
                "Group"
            ].to_numpy()

        ),

        start=1

    ):

        fold_array[
            test_index
        ] = fold_number


    if (
        fold_array
        <
        1
    ).any():

        raise AssertionError(
            "Some observations were not assigned to a fold."
        )


    oof["Fold"] = (
        fold_array
    )


# ============================================================
# 12. CHECK GROUP INTEGRITY
# ============================================================

group_fold_check = (

    oof
    .groupby(
        "Group"
    )[
        "Fold"
    ]
    .nunique()

)


if not (
    group_fold_check
    ==
    1
).all():

    bad_groups = (

        group_fold_check[
            group_fold_check
            >
            1
        ]
        .index
        .tolist()

    )


    raise AssertionError(

        "Group leakage detected. "
        "The following Study_ID + PFAS groups "
        "appear in multiple folds:\n"
        +
        str(
            bad_groups[:20]
        )

    )


# ============================================================
# 13. CALCULATE FOLD-WISE METRICS
# ============================================================

fold_rows = []


fold_numbers = sorted(
    oof[
        "Fold"
    ].unique()
)


for fold in fold_numbers:


    subset = oof.loc[
        oof[
            "Fold"
        ].eq(
            fold
        )
    ]


    y_true = subset[
        "Observed"
    ].to_numpy(
        dtype=float
    )


    y_pred = subset[
        "Predicted"
    ].to_numpy(
        dtype=float
    )


    if len(y_true) < 2:

        raise ValueError(
            f"Fold {fold} contains fewer than 2 observations."
        )


    current_r2 = r2_score(
        y_true,
        y_pred
    )


    current_rmse = calc_rmse(
        y_true,
        y_pred
    )


    fold_rows.append({

        "Fold":
            int(
                fold
            ),

        "n":
            int(
                len(
                    subset
                )
            ),

        "R2":
            float(
                current_r2
            ),

        "RMSE":
            float(
                current_rmse
            ),

    })


fold_metrics = pd.DataFrame(
    fold_rows
)


# ============================================================
# 14. OVERALL OOF METRICS
# ============================================================

overall_r2 = r2_score(

    oof[
        "Observed"
    ],

    oof[
        "Predicted"
    ]

)


overall_rmse = calc_rmse(

    oof[
        "Observed"
    ],

    oof[
        "Predicted"
    ]

)


# ============================================================
# 15. STRICT VALIDATION
# ============================================================

if not np.isclose(

    overall_r2,

    EXPECTED_XGB_R2,

    atol=VERIFY_ATOL,

    rtol=0

):

    raise AssertionError(

        "\nOverall OOF R2 check failed.\n"
        f"Calculated: {overall_r2:.9f}\n"
        f"Expected:   {EXPECTED_XGB_R2:.9f}\n"
    )


if not np.isclose(

    overall_rmse,

    EXPECTED_XGB_RMSE,

    atol=VERIFY_ATOL,

    rtol=0

):

    raise AssertionError(

        "\nOverall OOF RMSE check failed.\n"
        f"Calculated: {overall_rmse:.9f}\n"
        f"Expected:   {EXPECTED_XGB_RMSE:.9f}\n"
    )


print(
    "\n============================================"
)

print(
    "XGBoost fold-wise performance"
)

print(
    "============================================"
)


print(
    fold_metrics.to_string(
        index=False
    )
)


print(
    "\nOverall OOF:"
)

print(
    f"R2   = {overall_r2:.6f}"
)

print(
    f"RMSE = {overall_rmse:.6f}"
)


print(
    "\nFold sample-size total =",
    fold_metrics[
        "n"
    ].sum()
)


if (
    fold_metrics[
        "n"
    ].sum()
    !=
    len(oof)
):

    raise AssertionError(
        "Fold sample sizes do not sum to total OOF N."
    )


# ============================================================
# 16. SAVE SOURCE DATA FOR SI FIGURE
# ============================================================

fold_metrics.to_csv(

    OUT
    /
    "Field_XGBoost_fold_metrics.csv",

    index=False,

    encoding="utf-8-sig"

)


oof.to_csv(

    OUT
    /
    "Field_XGBoost_OOF_with_fold.csv",

    index=False,

    encoding="utf-8-sig"

)


MODEL_RESULTS.to_csv(

    OUT
    /
    "Field_model_overall_metrics.csv",

    index=False,

    encoding="utf-8-sig"

)


# ============================================================
# 17. FIGURE 2
#     XGBOOST FOLD-WISE PERFORMANCE
#
# Matches your previous SI style:
#
# - one point per fold
# - grey vertical stem to overall OOF
# - dashed overall OOF line
# - numeric fold value
# - Fold + n below
# ============================================================

fig2, (
    ax1,
    ax2

) = plt.subplots(

    1,
    2,

    figsize=(
        9.7,
        4.35
    )
)


fold_x = np.arange(
    1,
    len(
        fold_metrics
    )
    +
    1
)


# ============================================================
# 18. PANEL A — R²
# ============================================================

R2_COLOR = "#6F9BC1"


for x_i, row in zip(

    fold_x,

    fold_metrics.itertuples(
        index=False
    )

):


    value = float(
        row.R2
    )


    # Vertical grey line linking fold result to overall OOF
    ax1.plot(

        [
            x_i,
            x_i
        ],

        [
            overall_r2,
            value
        ],

        color="#B7B7B7",

        linewidth=1.7,

        zorder=1

    )


    # Fold point
    ax1.scatter(

        x_i,

        value,

        s=105,

        facecolor=R2_COLOR,

        edgecolor="#202A30",

        linewidth=1.0,

        zorder=3

    )


    # Value placement
    if value >= overall_r2:

        y_text = (
            value
            +
            0.018
        )

        vertical_align = "bottom"

    else:

        y_text = (
            value
            -
            0.018
        )

        vertical_align = "top"


    ax1.text(

        x_i,

        y_text,

        f"{value:.3f}",

        ha="center",

        va=vertical_align,

        fontsize=10.5,

        fontweight="bold"

    )


# Overall line
ax1.axhline(

    overall_r2,

    color="#666666",

    linewidth=1.45,

    linestyle=(
        0,
        (
            5,
            4
        )
    ),

    zorder=0

)


# Overall label
ax1.text(

    0.97,

    0.965,

    f"Overall OOF = {overall_r2:.3f}",

    transform=ax1.transAxes,

    ha="right",

    va="top",

    fontsize=10.5,

    fontweight="bold",

    color="#202020"

)


ax1.set_xticks(
    fold_x
)


ax1.set_xticklabels(

    [

        (
            f"Fold {int(row.Fold)}\n"
            f"(n={int(row.n)})"
        )

        for row
        in fold_metrics.itertuples(
            index=False
        )

    ],

    fontsize=10,

    fontweight="bold",

    linespacing=0.95

)


ax1.set_ylabel(

    r"$R^2$",

    fontsize=18,

    fontweight="bold"

)


# ------------------------------------------------------------
# Adaptive but visually tidy R² range
# ------------------------------------------------------------

r2_values = np.r_[

    fold_metrics[
        "R2"
    ].to_numpy(),

    overall_r2

]


r2_min = float(
    np.min(
        r2_values
    )
)


r2_max = float(
    np.max(
        r2_values
    )
)


r2_range = max(

    r2_max
    -
    r2_min,

    0.10

)


r2_lower = (
    r2_min
    -
    0.22
    *
    r2_range
)


r2_upper = (
    r2_max
    +
    0.25
    *
    r2_range
)


# Give sensible limits without forcing R² to start at zero
ax1.set_ylim(
    r2_lower,
    r2_upper
)


ax1.set_xlim(
    0.65,
    len(
        fold_metrics
    )
    +
    0.35
)


ax1.text(

    -0.15,

    1.035,

    "(a)",

    transform=ax1.transAxes,

    fontsize=18,

    fontweight="bold",

    ha="left",

    va="bottom"

)


style_axis(
    ax1
)


# ============================================================
# 19. PANEL B — RMSE
# ============================================================

RMSE_COLOR = "#D98B55"


for x_i, row in zip(

    fold_x,

    fold_metrics.itertuples(
        index=False
    )

):


    value = float(
        row.RMSE
    )


    ax2.plot(

        [
            x_i,
            x_i
        ],

        [
            overall_rmse,
            value
        ],

        color="#B7B7B7",

        linewidth=1.7,

        zorder=1

    )


    ax2.scatter(

        x_i,

        value,

        s=105,

        facecolor=RMSE_COLOR,

        edgecolor="#202A30",

        linewidth=1.0,

        zorder=3

    )


    if value >= overall_rmse:

        y_text = (
            value
            +
            0.018
        )

        vertical_align = "bottom"

    else:

        y_text = (
            value
            -
            0.018
        )

        vertical_align = "top"


    ax2.text(

        x_i,

        y_text,

        f"{value:.3f}",

        ha="center",

        va=vertical_align,

        fontsize=10.5,

        fontweight="bold"

    )


ax2.axhline(

    overall_rmse,

    color="#666666",

    linewidth=1.45,

    linestyle=(
        0,
        (
            5,
            4
        )
    ),

    zorder=0

)


ax2.text(

    0.97,

    0.965,

    f"Overall OOF = {overall_rmse:.3f}",

    transform=ax2.transAxes,

    ha="right",

    va="top",

    fontsize=10.5,

    fontweight="bold",

    color="#202020"

)


ax2.set_xticks(
    fold_x
)


ax2.set_xticklabels(

    [

        (
            f"Fold {int(row.Fold)}\n"
            f"(n={int(row.n)})"
        )

        for row
        in fold_metrics.itertuples(
            index=False
        )

    ],

    fontsize=10,

    fontweight="bold",

    linespacing=0.95

)


ax2.set_ylabel(

    "RMSE",

    fontsize=18,

    fontweight="bold"

)


# ------------------------------------------------------------
# Adaptive RMSE range
# ------------------------------------------------------------

rmse_values = np.r_[

    fold_metrics[
        "RMSE"
    ].to_numpy(),

    overall_rmse

]


rmse_min = float(
    np.min(
        rmse_values
    )
)


rmse_max = float(
    np.max(
        rmse_values
    )
)


rmse_range = max(

    rmse_max
    -
    rmse_min,

    0.10

)


rmse_lower = max(

    0,

    rmse_min
    -
    0.22
    *
    rmse_range

)


rmse_upper = (

    rmse_max
    +
    0.25
    *
    rmse_range

)


ax2.set_ylim(
    rmse_lower,
    rmse_upper
)


ax2.set_xlim(
    0.65,
    len(
        fold_metrics
    )
    +
    0.35
)


ax2.text(

    -0.15,

    1.035,

    "(b)",

    transform=ax2.transAxes,

    fontsize=18,

    fontweight="bold",

    ha="left",

    va="bottom"

)


style_axis(
    ax2
)


# ============================================================
# 20. FIGURE 2 LAYOUT
# ============================================================

fig2.subplots_adjust(

    left=0.085,

    right=0.985,

    bottom=0.22,

    top=0.92,

    wspace=0.30
)


FIG2_STEM = (
    OUT
    /
    "Field_XGBoost_foldwise_performance"
)


save_figure(
    fig2,
    FIG2_STEM
)


# ============================================================
# 21. FINAL CONSOLE REPORT
# ============================================================

print(
    "\n\n============================================"
)

print(
    "ALL CHECKS PASSED"
)

print(
    "============================================"
)


print(
    "\nCorrect OOF file:"
)

print(
    OOF_FILE
)


print(
    "\nOutput directory:"
)

print(
    OUT
)


print(
    "\nFigure 1:"
)

print(
    FIG1_STEM.with_suffix(
        ".png"
    )
)


print(
    "\nFigure 2:"
)

print(
    FIG2_STEM.with_suffix(
        ".png"
    )
)


print(
    "\nSource-data files:"
)

print(
    OUT
    /
    "Field_model_overall_metrics.csv"
)

print(
    OUT
    /
    "Field_XGBoost_fold_metrics.csv"
)

print(
    OUT
    /
    "Field_XGBoost_OOF_with_fold.csv"
)


# ============================================================
# 22. DISPLAY
# ============================================================

plt.show()


plt.close(
    fig1
)

plt.close(
    fig2
)
