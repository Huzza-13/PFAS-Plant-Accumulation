from project_paths import DATA_DIR, RESULTS_DIR
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.lines import Line2D

# ============================================================
# 1. File paths
# ============================================================

output_dir = str(RESULTS_DIR / 'Field_Nested_ML_results')

input_file = os.path.join(
    output_dir,
    "Field_Nested_OOF_predictions.csv"
)

# ============================================================
# 2. Read OOF predictions
# ============================================================

data = pd.read_csv(input_file)

y_obs = data["Observed"].values
y_pred = data["Predicted"].values

print(data.columns)
print(data["Study_ID"].value_counts())

# ============================================================
# 3. Performance metrics
# ============================================================

r2 = r2_score(y_obs, y_pred)
rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
n = len(y_obs)

print(f"OOF R²   = {r2:.3f}")
print(f"OOF RMSE = {rmse:.3f}")
print(f"n        = {n}")

# ============================================================
# 4. Study labels and colors
# ============================================================

study_label_map = {
    "S218": "Liu et al. (2019)",
    "S157": "Liu et al. (2022)",
    "S399": "Wen et al. (2014)",
    218: "Liu et al. (2019)",
    157: "Liu et al. (2022)",
    399: "Wen et al. (2014)"
}

data["Study_label"] = data["Study_ID"].map(study_label_map)
data["Study_label"] = data["Study_label"].fillna(data["Study_ID"].astype(str))

plot_order = [
    "Liu et al. (2019)",
    "Liu et al. (2022)",
    "Wen et al. (2014)"
]

study_color_map = {
    "Liu et al. (2019)": "#9ec3de",   # light blue
    "Liu et al. (2022)": "#e7c29f",   # light orange
    "Wen et al. (2014)": "#b8d4bd"    # light green
}

fit_color = "#d95f02"

# ============================================================
# 5. Plot range
# ============================================================

global_min = min(y_obs.min(), y_pred.min())
global_max = max(y_obs.max(), y_pred.max())

padding = (global_max - global_min) * 0.055

plot_min = global_min - padding
plot_max = global_max + padding

x_line = np.linspace(plot_min, plot_max, 300)

# ============================================================
# 6. Calibration regression
# Predicted = a + b × Observed
# ============================================================

reg = LinearRegression()
reg.fit(y_obs.reshape(-1, 1), y_pred)

calibration_fit = reg.predict(x_line.reshape(-1, 1))

# ============================================================
# 7. 95% confidence interval
# ============================================================

fitted_values = reg.predict(y_obs.reshape(-1, 1))
residuals = y_pred - fitted_values

dof = n - 2
residual_se = np.sqrt(np.sum(residuals ** 2) / dof)
t_value = stats.t.ppf(0.975, dof)

x_mean = np.mean(y_obs)
ss_x = np.sum((y_obs - x_mean) ** 2)

confidence_interval = (
    t_value
    * residual_se
    * np.sqrt(
        1 / n + ((x_line - x_mean) ** 2 / ss_x)
    )
)

# ============================================================
# 8. Global figure style
# ============================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "font.weight": "bold",

    "axes.labelsize": 15,   # 比之前小一点
    "axes.labelweight": "bold",

    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    "axes.linewidth": 1.4,

    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,

    "xtick.major.size": 7,
    "ytick.major.size": 7,

    "xtick.direction": "out",
    "ytick.direction": "out"
})

# ============================================================
# 9. Create figure
# ============================================================

fig, ax = plt.subplots(figsize=(5.6, 5.4))

# ============================================================
# 10. Scatter by study
# ============================================================

for study in plot_order:
    sub = data[data["Study_label"] == study]
    if len(sub) == 0:
        continue

    ax.scatter(
        sub["Observed"],
        sub["Predicted"],
        s=34,
        alpha=0.62,
        color=study_color_map[study],
        edgecolors="none",
        rasterized=True,
        zorder=3,
        label=study
    )

# ============================================================
# 11. 95% confidence band
# 不进图例
# ============================================================

ax.fill_between(
    x_line,
    calibration_fit - confidence_interval,
    calibration_fit + confidence_interval,
    color=fit_color,
    alpha=0.12,
    linewidth=0,
    zorder=1
)

# ============================================================
# 12. 1:1 line
# ============================================================

ax.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle="--",
    color="#7f7f7f",
    linewidth=1.8,
    zorder=2
)

# ============================================================
# 13. Calibration fit
# ============================================================

ax.plot(
    x_line,
    calibration_fit,
    color=fit_color,
    linewidth=2.3,
    zorder=4
)

# ============================================================
# 14. Metrics annotation
# ============================================================

metrics_text = (
    f"$R^2$ = {r2:.3f}\n"
    f"RMSE = {rmse:.3f}\n"
    f"$n$ = {n}"
)

ax.text(
    0.05,
    0.92,
    metrics_text,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=14,
    fontweight="bold",
    linespacing=1.15
)

# ============================================================
# 15. Axes
# ============================================================

ax.set_xlim(plot_min, plot_max)
ax.set_ylim(plot_min, plot_max)
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel(
    r"Observed log$_{10}$ concentration",
    fontsize=18,
    fontweight="bold",
    labelpad=12
)

ax.set_ylabel(
    r"Predicted log$_{10}$ concentration",
    fontsize=18,
    fontweight="bold",
    labelpad=12
)

# ============================================================
# 16. Clean frame
# ============================================================

ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for tick in ax.get_xticklabels():
    tick.set_fontweight("bold")

for tick in ax.get_yticklabels():
    tick.set_fontweight("bold")

ax.tick_params(axis="both", which="major", pad=7)

# ============================================================
# 17. Panel label
# ============================================================

ax.text(
    -0.14,
    1.03,
    "(a)",
    transform=ax.transAxes,
    fontsize=22,
    fontweight="bold",
    ha="left",
    va="bottom"
)

# ============================================================
# 18. Legend 1: line legend inside lower right
# ============================================================

line_handles = [
    Line2D(
        [0], [0],
        linestyle="--",
        color="#7f7f7f",
        linewidth=1.8,
        label="1:1 line"
    ),
    Line2D(
        [0], [0],
        linestyle="-",
        color=fit_color,
        linewidth=2.3,
        label="Calibration fit"
    )
]

legend1 = ax.legend(
    handles=line_handles,
    loc="lower right",
    frameon=False,
    fontsize=10.5,
    handlelength=2.4,
    handletextpad=0.6,
    borderpad=0.2,
    labelspacing=0.5
)

for text in legend1.get_texts():
    text.set_fontweight("bold")

ax.add_artist(legend1)

# ============================================================
# 19. Legend 2: study legend below the plot
# ============================================================

study_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markersize=8.5,
        markerfacecolor=study_color_map[study],
        markeredgecolor="none",
        alpha=0.9,
        label=study
    )
    for study in plot_order
]

legend2 = ax.legend(
    handles=study_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=3,
    frameon=False,
    fontsize=10.5,
    handletextpad=0.6,
    columnspacing=1.6
)

for text in legend2.get_texts():
    text.set_fontweight("bold")

# ============================================================
# 20. Layout
# ============================================================

plt.subplots_adjust(
    left=0.19,
    right=0.98,
    bottom=0.30,
    top=0.95
)

# ============================================================
# 21. Save
# ============================================================

tiff_file = os.path.join(
    output_dir,
    "Fig_Field_performance_final_style.tiff"
)

pdf_file = os.path.join(
    output_dir,
    "Fig_Field_performance_final_style.pdf"
)

png_file = os.path.join(
    output_dir,
    "Fig_Field_performance_final_style.png"
)

plt.savefig(
    tiff_file,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    pdf_file,
    bbox_inches="tight"
)

plt.savefig(
    png_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFigure saved to:")
print(tiff_file)
print(pdf_file)
print(png_file)