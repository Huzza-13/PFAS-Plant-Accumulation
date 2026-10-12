from project_paths import DATA_DIR, RESULTS_DIR
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

from pathlib import Path

# ============================================================
# 1. Output path
# ============================================================

output_dir = Path(
    str(RESULTS_DIR / 'Field_Nested_ML_results')
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

print("Output directory:")
print(output_dir)
print("repr:", repr(str(output_dir)))
# ============================================================
# 2. Input SHAP importance tables
# ============================================================

# ------------------------------------------------------------
# Laboratory model
# LEFT PANEL:
# Display aliases retain the historical plot labels, not rounded values.
# ------------------------------------------------------------

lab_data = pd.read_csv(RESULTS_DIR / 'Overall_aggregated_SHAP_importance.csv')

lab_df = lab_data.rename(columns={'Domain':'Class'})[['Feature','Mean_abs_SHAP','Class']].replace({'Feature': {'Exposure concentration':'PFAS concentration', 'MinPartialCharge':'Min partial charge'}})


# ------------------------------------------------------------
# Field model
# RIGHT PANEL:
# Exact field table is exported by the reviewed 003 script.
# ------------------------------------------------------------

field_data = pd.read_csv(RESULTS_DIR / 'Field_Nested_ML_results' / 'Field_aggregated_SHAP_importance.csv')

field_df = field_data.rename(columns={'display_name':'Feature', 'mean_abs_shap':'Mean_abs_SHAP', 'class':'Class'})[['Feature','Mean_abs_SHAP','Class']].replace({'Feature': {'Soil PFAS concentration':'PFAS concentration'}})

for table, expected_count in [(lab_df, 13), (field_df, 11)]:
    if len(table) != expected_count or table["Feature"].duplicated().any():
        raise ValueError("Importance table has an unexpected feature set.")
    if not np.isfinite(table["Mean_abs_SHAP"]).all() or (table["Mean_abs_SHAP"] < 0).any():
        raise ValueError("Mean absolute SHAP values must be finite and nonnegative.")
    if not table["Class"].isin(["Plant", "Environment", "Exposure", "Molecular"]).all():
        raise ValueError("Unrecognized SHAP feature domain.")


# ============================================================
# 3. Color map
# ============================================================

color_map = {
    "Plant": "#2E8B57",
    "Environment": "#2C7FB8",
    "Exposure": "#7A4FA3",
    "Molecular": "#D95F02"
}


# ============================================================
# 4. Global plot style
# ============================================================

mpl.rcParams.update({

    "font.family": "Arial",

    "font.size": 14,

    "font.weight": "bold",

    # axis title
    "axes.labelsize": 20,
    "axes.labelweight": "bold",

    # tick labels
    "xtick.labelsize": 15,
    "ytick.labelsize": 16,

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
# 5. Plotting function
# ============================================================

def plot_shap_panel(
    ax,
    df,
    xlim_max,
    show_legend=False
):

    # Highest importance at top
    plot_df = (
        df
        .sort_values(
            "Mean_abs_SHAP",
            ascending=False
        )
        .iloc[::-1]
        .reset_index(drop=True)
    )

    y_pos = np.arange(
        len(plot_df)
    )

    plot_df["Color"] = (
        plot_df["Class"]
        .map(color_map)
    )


    # --------------------------------------------------------
    # Vertical dashed grid
    # --------------------------------------------------------

    ax.set_axisbelow(True)

    ax.xaxis.grid(
        True,
        linestyle="--",
        linewidth=0.8,
        color="#d9d9d9"
    )

    ax.yaxis.grid(False)


    # --------------------------------------------------------
    # Lollipop lines
    # --------------------------------------------------------

    for i, row in plot_df.iterrows():

        ax.hlines(
            y=i,
            xmin=0,
            xmax=row["Mean_abs_SHAP"],
            color=row["Color"],
            linewidth=2.3,
            alpha=0.23,
            zorder=1
        )


    # --------------------------------------------------------
    # Points
    # --------------------------------------------------------

    ax.scatter(
        plot_df["Mean_abs_SHAP"],
        y_pos,
        s=220,
        c=plot_df["Color"],
        edgecolors="none",
        zorder=3
    )


    # --------------------------------------------------------
    # Y labels
    # --------------------------------------------------------

    ax.set_yticks(
        y_pos
    )

    ax.set_yticklabels(
        plot_df["Feature"],
        fontsize=16,
        fontweight="bold"
    )


    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        xlim_max
    )

    ax.set_xlabel(
        "Mean absolute SHAP value",
        fontsize=20,
        fontweight="bold",
        labelpad=9
    )


    # --------------------------------------------------------
    # Frame
    # --------------------------------------------------------

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


    # --------------------------------------------------------
    # Tick style
    # --------------------------------------------------------

    ax.tick_params(
        axis="x",
        labelsize=15,
        width=1.4,
        length=7,
        pad=6
    )

    ax.tick_params(
        axis="y",
        labelsize=16,
        width=1.4,
        length=7,
        pad=5
    )


    for tick in ax.get_xticklabels():
        tick.set_fontweight(
            "bold"
        )

    for tick in ax.get_yticklabels():
        tick.set_fontweight(
            "bold"
        )


    # --------------------------------------------------------
    # Legend on right panel only
    # --------------------------------------------------------

    if show_legend:

        legend_handles = [

            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markersize=10,
                markerfacecolor=color_map["Plant"],
                markeredgecolor="none",
                label="Plant"
            ),

            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markersize=10,
                markerfacecolor=color_map["Environment"],
                markeredgecolor="none",
                label="Environment"
            ),

            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markersize=10,
                markerfacecolor=color_map["Exposure"],
                markeredgecolor="none",
                label="Exposure"
            ),

            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markersize=10,
                markerfacecolor=color_map["Molecular"],
                markeredgecolor="none",
                label="Molecular"
            )
        ]


        legend = ax.legend(

            handles=legend_handles,

            loc="lower right",

            frameon=False,

            fontsize=13,

            handletextpad=0.6,

            labelspacing=0.6,

            borderpad=0.2
        )


        for text in legend.get_texts():

            text.set_fontweight(
                "bold"
            )


# ============================================================
# 6. Unified x-axis maximum
# ============================================================

xmax = max(

    lab_df[
        "Mean_abs_SHAP"
    ].max(),

    field_df[
        "Mean_abs_SHAP"
    ].max()

) * 1.10


# ============================================================
# 7. Create side-by-side figure
#
# Equal panel widths
# Moderate spacing
# ============================================================

fig, axes = plt.subplots(

    nrows=1,

    ncols=2,

    figsize=(14.7, 5.8),

    gridspec_kw={
        "width_ratios": [
            1.0,
            1.0
        ]
    }
)


# ============================================================
# 8. Left panel
# ============================================================

plot_shap_panel(

    axes[0],

    lab_df,

    xlim_max=xmax,

    show_legend=False
)


# ============================================================
# 9. Right panel
# ============================================================

plot_shap_panel(

    axes[1],

    field_df,

    xlim_max=xmax,

    show_legend=True
)


# ============================================================
# 10. Panel labels
# ============================================================

axes[0].text(

    -0.22,
    1.025,

    "(a)",

    transform=axes[0].transAxes,

    fontsize=23,

    fontweight="bold",

    ha="left",

    va="bottom"
)


axes[1].text(

    -0.22,
    1.025,

    "(b)",

    transform=axes[1].transAxes,

    fontsize=23,

    fontweight="bold",

    ha="left",

    va="bottom"
)


# ============================================================
# 11. Layout
#
# Main change:
# wspace reduced from 0.62 -> 0.42
# ============================================================

plt.subplots_adjust(

    left=0.105,

    right=0.985,

    bottom=0.16,

    top=0.92,

    wspace=0.42
)


# ============================================================
# ============================================================
# Save
# ============================================================

output_stem = (
    output_dir /
    "Fig_SHAP_Lab_vs_Field_side_by_side_refined"
)

tiff_file = str(output_stem) + ".tiff"
pdf_file = str(output_stem) + ".pdf"
png_file = str(output_stem) + ".png"
svg_file = str(output_stem) + ".svg"


fig.savefig(
    tiff_file,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    pil_kwargs={
        "compression": "tiff_lzw"
    }
)

fig.savefig(
    pdf_file,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    png_file,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    svg_file,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)

print("\nFigure saved to:")
print(tiff_file)
print(pdf_file)
print(png_file)
print(svg_file)
