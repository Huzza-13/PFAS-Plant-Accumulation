from project_paths import DATA_DIR, RESULTS_DIR
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

SCRIPT_DIR = RESULTS_DIR
BASE_DIR = Path(os.environ.get("PFAS_FEATURE_BLOCK_DIR", str(SCRIPT_DIR)))
OUTPUT_DIR = Path(os.environ.get("PFAS_FIG_OUTPUT_DIR", str(BASE_DIR)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = BASE_DIR / "Feature_block_model_results_final15.csv"
OUTPUT_STEM = OUTPUT_DIR / "Fig_S_Predictor_domain_sensitivity_compact"


# ============================================================
# 2. Read and validate results
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()

required_columns = {"Model", "R2", "RMSE", "Number_of_features"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}\n"
        f"Available columns: {df.columns.tolist()}"
    )

df["Model"] = df["Model"].astype(str).str.strip()
model_order = ["E", "E+M"]

df = df[df["Model"].isin(model_order)].copy()
df = df.set_index("Model").reindex(model_order)

if df[["R2", "RMSE", "Number_of_features"]].isna().any().any():
    raise ValueError(
        "The results file must contain exactly the E and E+M model results."
    )

base_r2 = float(df.loc["E", "R2"])
full_r2 = float(df.loc["E+M", "R2"])
base_rmse = float(df.loc["E", "RMSE"])
full_rmse = float(df.loc["E+M", "RMSE"])

base_n = int(df.loc["E", "Number_of_features"])
full_n = int(df.loc["E+M", "Number_of_features"])


# ============================================================
# 3. Publication style: larger and bold typography
# ============================================================

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5,
    "font.weight": "bold",
    "axes.labelsize": 10.5,
    "axes.labelweight": "bold",
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

BASE_COLOR = "#6F8FAF"
MOLECULAR_COLOR = "#D49A3A"
CONNECTOR_COLOR = "#B8B8B8"
TEXT_COLOR = "#252525"


# ============================================================
# 4. Helpers
# ============================================================

def metric_limits(value_1, value_2, minimum_padding):
    lower = min(value_1, value_2)
    upper = max(value_1, value_2)
    difference = upper - lower
    padding = max(difference * 0.55, minimum_padding)
    return lower - padding, upper + padding


def draw_dumbbell(ax, base_value, molecular_value, xlabel, delta_text):
    ax.plot(
        [base_value, molecular_value],
        [0, 0],
        color=CONNECTOR_COLOR,
        linewidth=2.3,
        solid_capstyle="round",
        zorder=1,
    )

    ax.scatter(
        base_value,
        0,
        s=82,
        color=BASE_COLOR,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )

    ax.scatter(
        molecular_value,
        0,
        s=82,
        color=MOLECULAR_COLOR,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )

    ax.annotate(
        f"{base_value:.3f}",
        xy=(base_value, 0),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=TEXT_COLOR,
    )

    ax.annotate(
        f"{molecular_value:.3f}",
        xy=(molecular_value, 0),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=MOLECULAR_COLOR,
    )

    ax.text(
        0.5,
        0.12,
        delta_text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="#4A4A4A",
    )

    ax.set_ylim(-0.42, 0.50)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, labelpad=6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", direction="out", length=3.8, width=1.0)

    for tick_label in ax.get_xticklabels():
        tick_label.set_fontweight("bold")


# ============================================================
# 5. Draw compact two-panel figure
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.45))

r2_limits = metric_limits(base_r2, full_r2, minimum_padding=0.018)
rmse_limits = metric_limits(base_rmse, full_rmse, minimum_padding=0.018)

draw_dumbbell(
    axes[0],
    base_r2,
    full_r2,
    xlabel="OOF R² (higher is better)",
    delta_text=f"ΔR² = {full_r2 - base_r2:+.3f}",
)
axes[0].set_xlim(*r2_limits)

draw_dumbbell(
    axes[1],
    base_rmse,
    full_rmse,
    xlabel="OOF RMSE (lower is better)",
    delta_text=f"ΔRMSE = {full_rmse - base_rmse:+.3f}",
)
axes[1].set_xlim(*rmse_limits)

for panel_label, ax in zip(["(a)", "(b)"], axes):
    ax.text(
        -0.09,
        1.05,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

legend_handles = [
    plt.Line2D(
        [0], [0], marker="o", linestyle="none", markersize=6.5,
        markerfacecolor=BASE_COLOR, markeredgecolor="white",
        label=f"Base variables ({base_n} predictors)",
    ),
    plt.Line2D(
        [0], [0], marker="o", linestyle="none", markersize=6.5,
        markerfacecolor=MOLECULAR_COLOR, markeredgecolor="white",
        label=f"Base + molecular descriptors ({full_n} predictors)",
    ),
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    frameon=False,
    prop={"family": "Arial", "size": 9.0, "weight": "bold"},
    handletextpad=0.5,
    columnspacing=1.6,
)

fig.subplots_adjust(
    left=0.075,
    right=0.985,
    bottom=0.28,
    top=0.70,
    wspace=0.30,
)


# ============================================================
# 6. Export
# ============================================================

fig.savefig(
    OUTPUT_STEM.with_suffix(".svg"),
    bbox_inches="tight",
    facecolor="white",
)
fig.savefig(
    OUTPUT_STEM.with_suffix(".pdf"),
    bbox_inches="tight",
    facecolor="white",
)
fig.savefig(
    OUTPUT_STEM.with_suffix(".tiff"),
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    pil_kwargs={"compression": "tiff_lzw"},
)
fig.savefig(
    OUTPUT_STEM.with_suffix(".png"),
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

print("Input:", INPUT_FILE)
print("Saved figure stem:", OUTPUT_STEM)
print(f"R2:   {base_r2:.3f} -> {full_r2:.3f}")
print(f"RMSE: {base_rmse:.3f} -> {full_rmse:.3f}")

plt.show()
plt.close(fig)
