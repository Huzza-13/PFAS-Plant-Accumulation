"""Descriptive tissue-stratified displays of an existing global model.

No training, parameter search, or modification of the source results.
All observations are used. These are final-model SHAP values, not OOF SHAP.
"""
from project_paths import DATA_DIR, RESULTS_DIR
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

RESULTS_DIR = Path(str(RESULTS_DIR))
OUTPUT_DIR = RESULTS_DIR / 'Global_model_tissue_SHAP'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TISSUES = ["Root", "Leaf", "Stem", "Fruit/seed"]
ENV = ["SOM", "Exposure Concentration", "Soil pH", "Growth Temperature", "Growth Duration"]
MOL = ["ALogP", "Chi3v", "GATS3c", "MinPartialCharge", "SpMin8_Bhi", "TPSA"]
FEATURES = ENV + MOL
LABELS = {"Exposure Concentration": "Exposure concentration",
          "Growth Temperature": "Growth temperature", "Growth Duration": "Growth duration",
          "MinPartialCharge": "Min partial charge"}
COLORS = ["#21867A", "#668EB5", "#8875AD", "#CA829A"]
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.labelweight": "bold",
    "axes.linewidth": .7, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", dpi=600, facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{name}.svg", dpi=600, facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{name}.tiff", dpi=600, facecolor="white")
    plt.close(fig)


def swarm_offsets(values, limit, bins=85):
    """Deterministic vertical spreading only; original SHAP x values are preserved."""
    bin_id = np.clip(((values + limit) / (2 * limit) * bins).astype(int), 0, bins - 1)
    offsets = np.zeros(len(values))
    for b in np.unique(bin_id):
        indices = np.flatnonzero(bin_id == b)
        # Interleave around the centre, with a fixed maximum half-height.
        sequence = np.arange(len(indices))
        levels = np.ceil(sequence / 2) * np.where(sequence % 2, 1, -1)
        offsets[indices] = levels * min(.075, .32 / max(1, np.max(np.abs(levels))))
    return offsets


def main():
    s = pd.read_csv(RESULTS_DIR / "Final_XGBoost_raw_SHAP_values.csv")
    x = pd.read_csv(RESULTS_DIR / "Final_XGBoost_transformed_X.csv")
    meta = pd.read_csv(RESULTS_DIR / "Final_XGBoost_SHAP_metadata.csv")
    assert s.shape == x.shape and s.columns.equals(x.columns), "Saved SHAP / X columns differ"
    assert len(s) == len(meta), "Metadata row count differs"
    assert np.isfinite(s.to_numpy()).all() and np.isfinite(x.to_numpy()).all()
    tissue_cols = [f"Plant tissue group classify_{i}" for i in range(4)]
    one_hot = x[tissue_cols].to_numpy()
    assert np.isin(one_hot, [0, 1]).all() and (one_hot.sum(axis=1) == 1).all()
    group = one_hot.argmax(axis=1)
    # Verify saved SHAP really belongs to this fitted model and this transformed X.
    import joblib
    import shap
    pipeline = joblib.load(RESULTS_DIR / "Final_XGBoost_pipeline.joblib")
    recalculated = shap.TreeExplainer(pipeline.named_steps["model"]).shap_values(x)
    np.testing.assert_allclose(s.to_numpy(), recalculated, rtol=2e-5, atol=2e-6)

    rows = []
    for k, tissue in enumerate(TISSUES):
        mask = group == k
        for feature in FEATURES:
            values = s.loc[mask, feature].to_numpy()
            rows.append({"Tissue": tissue, "Feature": feature,
                "n_observations": int(mask.sum()), "n_studies": int(meta.loc[mask, "Reference"].nunique()),
                "Mean_absolute_SHAP": float(np.mean(np.abs(values))),
                "Mean_signed_SHAP": float(np.mean(values)),
                "Median_SHAP": float(np.median(values)),
                "Q25_SHAP": float(np.quantile(values, .25)), "Q75_SHAP": float(np.quantile(values, .75))})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "Tissue_stratified_global_SHAP_summary.csv", index=False)
    source = s[FEATURES].copy()
    source.insert(0, "Tissue", np.array(TISSUES)[group])
    source.insert(0, "Source_row_zero_based", np.arange(len(s)))
    source.to_csv(OUTPUT_DIR / "Tissue_stratified_global_SHAP_observations.csv", index=False)
    counts = [int((group == k).sum()) for k in range(4)]
    matrix = summary.pivot(index="Feature", columns="Tissue", values="Mean_absolute_SHAP").loc[FEATURES, TISSUES].to_numpy()

    # One common, unnormalized scale preserves absolute cross-tissue comparisons.
    cmap = LinearSegmentedColormap.from_list("importance_teal", ["#F6FAF9", "#C8E0DA", "#70AEA2", "#176C60"])
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    fig.subplots_adjust(left=.34, right=.86, top=.83, bottom=.17)
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=matrix.max(), aspect="auto")
    for i in range(len(FEATURES)):
        for j in range(4):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if matrix[i,j] > .65 * matrix.max() else "#223A36", fontweight="bold")
    ax.set_yticks(range(len(FEATURES)), [LABELS.get(f, f) for f in FEATURES], fontweight="bold")
    ax.set_xticks(range(4), [f"{t}\n(n = {n})" for t,n in zip(TISSUES, counts)], fontweight="bold")
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=8)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(FEATURES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.axhline(4.5, color="#647873", linewidth=1.1)
    for spine in ax.spines.values(): spine.set_visible(False)
    # Group annotations at the left keep the entire feature list explicit.
    ax.text(-.60, 1.015, "Environment / exposure", transform=ax.transAxes,
            fontsize=8, color="#566C67", fontweight="bold", ha="left")
    ax.text(-.60, .535, "Molecular descriptors", transform=ax.transAxes,
            fontsize=8, color="#566C67", fontweight="bold", ha="left")
    cax = fig.add_axes([.89, .17, .018, .66])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Mean absolute SHAP value", fontsize=9, labelpad=10)
    cb.outline.set_visible(False)
    fig.text(.50, .965, "Feature contributions by plant tissue", ha="center", fontsize=12, fontweight="bold")
    fig.text(.50, .925, "One fitted global model · tissue-stratified observations", ha="center", fontsize=8, color="#59635F")
    fig.text(.5, .055, "Darker cells indicate larger average contributions; all cells use the same scale.",
             ha="center", fontsize=8, color="#59635F")
    save(fig, "Global_model_tissue_SHAP_heatmap")

    # All points are retained. Feature colors use identical global scales in all four panels.
    color_map = LinearSegmentedColormap.from_list("feature_values", ["#397AB7", "#D9DCE5", "#C14A71"])
    limit = float(np.max(np.abs(s[FEATURES].to_numpy()))) * 1.08
    feature_colors = {}
    limits = {}
    for f in FEATURES:
        low, high = np.quantile(x[f], [.05, .95])
        if high <= low: low, high = float(x[f].min()), float(x[f].max())
        feature_colors[f] = np.clip((x[f].to_numpy() - low) / (high-low), 0, 1) if high>low else np.full(len(x), .5)
        limits[f] = [float(low), float(high)]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=.23, right=.86, top=.89, bottom=.13, hspace=.20, wspace=.15)
    for k, ax in enumerate(axes.flat):
        mask = group == k
        ax.axvline(0, color="#9FA7AB", linewidth=.8, zorder=0)
        for i, f in enumerate(FEATURES):
            values = s.loc[mask, f].to_numpy()
            y = len(FEATURES)-1-i
            ax.axhline(y, color="#EDF0F1", lw=.6, zorder=0)
            ax.scatter(values, y + swarm_offsets(values, limit), s=5.5,
                       c=feature_colors[f][mask], cmap=color_map, vmin=0, vmax=1,
                       linewidths=0, alpha=.8, rasterized=True)
        ax.axhline(5.5, color="#B9C2C6", lw=.7, linestyle=(0,(3,3)))
        ax.set_title(f"({chr(97+k)}) {TISSUES[k]}  (n = {counts[k]})", fontsize=10, fontweight="bold", pad=8)
        ax.set_yticks(range(len(FEATURES)), [LABELS.get(f,f) for f in FEATURES[::-1]], fontsize=8, fontweight="bold")
        ax.tick_params(axis="y", length=0, pad=7)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-.65, len(FEATURES)-.35)
        ax.spines["left"].set_visible(False)
        if k >= 2: ax.set_xlabel("SHAP value")
    cax = fig.add_axes([.89, .35, .013, .32])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0,1), cmap=color_map), cax=cax)
    cb.set_ticks([0,1], labels=["Low", "High"])
    cb.set_label("Feature value", fontsize=8)
    cb.outline.set_visible(False)
    fig.text(.52, .97, "Within-tissue SHAP distributions", ha="center", fontsize=12, fontweight="bold")
    fig.text(.52, .935, "One fitted global model · all 11 displayed predictors", ha="center", fontsize=8, color="#59635F")
    fig.text(.52, .055, "Each point is one observation. Feature-color scales are shared across tissues.", ha="center", fontsize=7.5, color="#59635F")
    save(fig, "Global_model_tissue_SHAP_distributions")
    qa = {"model": "Existing final global XGBoost; no refitting", "shap_alignment": "Recalculated against saved model and transformed X; allclose passed",
          "observations": len(s), "tissue_counts": dict(zip(TISSUES, counts)),
          "features_displayed": FEATURES, "features_hidden": ["Tissue group", "Morphotype"],
          "hidden_reason": "Requested tissue-stratified display of environment/exposure and all six molecular predictors; both remain in fitted model",
          "statistics": "Observation-weighted descriptive mean absolute SHAP; no tests or confidence intervals. Repeated observations are not independent studies.",
          "provenance": "Final fitted-model explanations, not OOF explanations or four separately fitted models",
          "interpretation": "Differences reflect observed tissue subsets and covariate compositions; not isolated causal tissue effects. Absolute SHAP shows magnitude, not sign.",
          "color_limits_global_5th_95th_percentiles": limits, "points_removed": 0}
    (OUTPUT_DIR / "Figure_QA.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
