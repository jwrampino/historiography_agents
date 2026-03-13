"""
visualization.py
================
All visualizations for the multi-agent historical synthesis experiment.
Reads directly from CSVs exported by ExperimentStorage.export_to_csv().

Expected files in --data-dir:
  triads.csv
  proposals.csv
  synthesis.csv
  convergence_results.csv
  feature_importance.csv          (optional)
  prediction_model.json           (optional)
  inference_results.json          (optional)

Usage:
  python visualization.py --data-dir data/agent_experiments --out-dir figures/
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.colors as mcolors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Global style ──────────────────────────────────────────────────────────────
PALETTE   = sns.color_palette("husl", 8)
NEUTRAL_COLOR     = "#3498db"
ACCENT_COLOR      = "#9b59b6"
BIAS_COLOR        = "#e67e22"

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path, label: str) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {label:55s} → {path.name}")


def _load_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in ["triads", "proposals", "synthesis", "convergence_results", "feature_importance"]:
        p = data_dir / f"{name}.csv"
        if p.exists():
            tables[name] = pd.read_csv(p)
    return tables


def _merged(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return triads LEFT-JOINed with convergence_results."""
    if "triads" not in t or "convergence_results" not in t:
        return pd.DataFrame()
    return t["triads"].merge(t["convergence_results"], on="triad_id", how="left")


def _delta_color(delta_series: pd.Series) -> list[str]:
    """Map convergence_delta to RdYlGn — green=high delta, red=low."""
    vals = delta_series.fillna(0).values
    vmin, vmax = vals.min(), vals.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-9)
    cmap = plt.get_cmap("RdYlGn")
    return [mcolors.to_hex(cmap(norm(v))) for v in vals]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – GEOMETRY (plots 01-08)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_01_geometry_distributions(df: pd.DataFrame, out: Path) -> None:
    """Histograms for all 8 triangle geometry features."""
    cols = ["side_1","side_2","side_3","perimeter","area","min_angle","max_angle","angle_variance"]
    labels = ["Side 1","Side 2","Side 3","Perimeter","Area","Min Angle","Max Angle","Angle Variance"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("Triangle Geometry Feature Distributions", fontsize=15, fontweight="bold")
    for ax, col, lbl in zip(axes.flat, cols, labels):
        if col not in df.columns:
            ax.set_visible(False); continue
        data = df[col].dropna()
        ax.hist(data, bins=25, color=NEUTRAL_COLOR, edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(),   color="red",   ls="--", lw=1.5, label=f"μ={data.mean():.3f}")
        ax.axvline(data.median(), color="green", ls=":",  lw=1.5, label=f"med={data.median():.3f}")
        ax.set_title(lbl, fontsize=11); ax.set_xlabel(lbl); ax.set_ylabel("Count")
        ax.legend(fontsize=7)
    plt.tight_layout()
    _save(fig, out / "01_geometry_distributions.png", "Geometry distributions")


def plot_02_perimeter_vs_area(df: pd.DataFrame, out: Path) -> None:
    """Perimeter vs area scatter coloured by angle variance."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["perimeter"], df["area"], c=df["angle_variance"],
                    cmap="viridis", s=60, alpha=0.8, edgecolors="white", lw=0.4)
    plt.colorbar(sc, ax=ax, label="Angle Variance")
    ax.set_xlabel("Perimeter"); ax.set_ylabel("Area")
    ax.set_title("Triangle Shape Space: Perimeter vs Area", fontweight="bold")
    _save(fig, out / "02_perimeter_vs_area.png", "Perimeter vs area")


def plot_03_side_lengths(df: pd.DataFrame, out: Path) -> None:
    """Pairwise side-length distributions as overlapping KDE."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for col, lbl, c in zip(["side_1","side_2","side_3"],
                            ["Side 1","Side 2","Side 3"],
                            PALETTE[:3]):
        if col in df.columns:
            sns.kdeplot(df[col].dropna(), ax=ax, label=lbl, color=c, fill=True, alpha=0.3)
    ax.set_xlabel("Cosine Distance"); ax.set_ylabel("Density")
    ax.set_title("Pairwise Side-Length Distributions", fontweight="bold")
    ax.legend()
    _save(fig, out / "03_side_length_kde.png", "Side length KDE")


def plot_04_angle_analysis(df: pd.DataFrame, out: Path) -> None:
    """Min angle vs max angle scatter + angle variance histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Scatter
    sc = axes[0].scatter(df["min_angle"], df["max_angle"], c=df["area"],
                         cmap="plasma", s=60, alpha=0.8, edgecolors="white", lw=0.3)
    plt.colorbar(sc, ax=axes[0], label="Area")
    axes[0].set_xlabel("Min Angle (rad)"); axes[0].set_ylabel("Max Angle (rad)")
    axes[0].set_title("Angle Range by Triangle Area", fontweight="bold")
    # Variance histogram
    axes[1].hist(df["angle_variance"].dropna(), bins=25, color=ACCENT_COLOR,
                 edgecolor="white", alpha=0.85)
    axes[1].axvline(df["angle_variance"].mean(), color="red", ls="--",
                    label=f"μ={df['angle_variance'].mean():.4f}")
    axes[1].set_xlabel("Angle Variance"); axes[1].set_ylabel("Count")
    axes[1].set_title("Triangle Regularity (Angle Variance)", fontweight="bold")
    axes[1].legend()
    plt.tight_layout()
    _save(fig, out / "04_angle_analysis.png", "Angle analysis")


def plot_05_geometry_correlation_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Correlation heatmap of all geometry columns."""
    geo_cols = ["side_1","side_2","side_3","perimeter","area","min_angle","max_angle","angle_variance"]
    geo_cols = [c for c in geo_cols if c in df.columns]
    corr = df[geo_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.4, ax=ax)
    ax.set_title("Geometry Feature Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "05_geometry_correlation_heatmap.png", "Geometry correlation heatmap")


def plot_06_triangle_regularity(df: pd.DataFrame, out: Path) -> None:
    """Regularity score = 1 - side_variance / mean_side². Distribution + vs area."""
    avg  = (df["side_1"] + df["side_2"] + df["side_3"]) / 3
    var  = df[["side_1","side_2","side_3"]].var(axis=1)
    reg  = 1 - var / (avg**2 + 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(reg, bins=25, color=NEUTRAL_COLOR, edgecolor="white", alpha=0.85)
    axes[0].axvline(reg.mean(), color="red", ls="--", label=f"μ={reg.mean():.3f}")
    axes[0].set_xlabel("Regularity (1 = equilateral)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Triangle Regularity Distribution", fontweight="bold")
    axes[0].legend()
    axes[1].scatter(reg, df["area"], alpha=0.7, color=ACCENT_COLOR, edgecolors="white", lw=0.3)
    axes[1].set_xlabel("Regularity"); axes[1].set_ylabel("Area")
    axes[1].set_title("Regularity vs Triangle Area", fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "06_triangle_regularity.png", "Triangle regularity")


def plot_07_side_balance(df: pd.DataFrame, out: Path) -> None:
    """Box plots of each side length to compare spread."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [df[c].dropna().values for c in ["side_1","side_2","side_3"] if c in df.columns]
    bp = ax.boxplot(data, patch_artist=True, labels=["Side 1","Side 2","Side 3"])
    for patch, color in zip(bp["boxes"], PALETTE[:3]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_ylabel("Cosine Distance"); ax.set_title("Side Length Balance Across Triads", fontweight="bold")
    _save(fig, out / "07_side_balance_boxplot.png", "Side balance boxplot")


def plot_08_geometry_pairplot(df: pd.DataFrame, out: Path) -> None:
    """Seaborn pairplot of key geometry features."""
    cols = ["perimeter","area","angle_variance","min_angle","max_angle"]
    cols = [c for c in cols if c in df.columns]
    g = sns.pairplot(df[cols].dropna(), diag_kind="kde", plot_kws={"alpha":0.5, "s":30})
    g.figure.suptitle("Geometry Feature Pairplot", y=1.02, fontweight="bold")
    _save(g.figure, out / "08_geometry_pairplot.png", "Geometry pairplot")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – CONVERGENCE (plots 09-17)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_09_convergence_outcome_bar(df: pd.DataFrame, out: Path) -> None:
    """Triad counts by convergence_delta quartile."""
    if "convergence_delta" not in df.columns: return
    df2 = df.copy()
    df2["delta_q"] = pd.qcut(df2["convergence_delta"], q=4,
                              labels=["Q1 (low Δ)","Q2","Q3","Q4 (high Δ)"])
    counts = df2["delta_q"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=PALETTE[:4], edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(int(val)), ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_title("Triad Counts by Convergence Delta Quartile", fontweight="bold")
    ax.set_xlabel("Delta Quartile (Q1=least movement toward centroid, Q4=most)")
    _save(fig, out / "09_convergence_outcome_bar.png", "Delta quartile bar")


def plot_10_delta_distribution(df: pd.DataFrame, out: Path) -> None:
    """Histogram of convergence_delta with zero line and mean annotated."""
    fig, ax = plt.subplots(figsize=(9, 5))
    data = df["convergence_delta"].dropna()
    ax.hist(data, bins=25, color=NEUTRAL_COLOR, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", ls="--", lw=1.5, label="Zero (no movement)")
    ax.axvline(data.mean(),   color="red",   ls="--", lw=1.5, label=f"Mean={data.mean():.3f}")
    ax.axvline(data.median(), color="green", ls=":",  lw=1.5, label=f"Median={data.median():.3f}")
    ax.set_xlabel("Convergence Delta (positive = synthesis closer to centroid)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Convergence Delta", fontweight="bold")
    ax.legend()
    _save(fig, out / "10_delta_distribution.png", "Delta distribution")


def plot_11_delta_by_geometry(df: pd.DataFrame, out: Path) -> None:
    """4-panel scatter: perimeter/area/angle_variance/mean_historian_distance vs delta."""
    feats = ["perimeter","area","angle_variance","mean_historian_distance"]
    feats = [f for f in feats if f in df.columns]
    ncols = 2; nrows = (len(feats) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5*nrows))
    axes_flat = axes.flat
    delta_vals = df["convergence_delta"].fillna(0)
    norm11 = plt.Normalize(vmin=delta_vals.min(), vmax=delta_vals.max())
    cmap11 = plt.get_cmap("RdYlGn")
    sc_last = None
    for ax, feat in zip(axes_flat, feats):
        sc_last = ax.scatter(df[feat], delta_vals, c=delta_vals, cmap=cmap11,
                             norm=norm11, alpha=0.75, edgecolors="white", lw=0.3, s=55)
        mask = df[[feat,"convergence_delta"]].dropna()
        if len(mask) > 2:
            m, b, r, p, _ = stats.linregress(mask[feat], mask["convergence_delta"])
            xs = np.linspace(mask[feat].min(), mask[feat].max(), 100)
            ax.plot(xs, m*xs+b, color="navy", lw=1.5, ls="--",
                    label=f"r={r:.2f}, p={p:.3f}")
            ax.legend(fontsize=8)
        ax.set_xlabel(feat); ax.set_ylabel("Convergence Delta")
        ax.set_title(f"{feat} vs Convergence Delta", fontweight="bold")
        ax.axhline(0, color="gray", ls=":", lw=1)
    for ax in list(axes.flat)[len(feats):]:
        ax.set_visible(False)
    if sc_last is not None:
        fig.colorbar(sc_last, ax=axes.ravel().tolist(), label="Convergence Delta", shrink=0.6)
    plt.tight_layout()
    _save(fig, out / "11_delta_by_geometry.png", "Delta by geometry features")


def plot_12_convergence_rate_by_area_quartile(df: pd.DataFrame, out: Path) -> None:
    """Mean delta per area quartile — tests diversity→convergence hypothesis."""
    df2 = df.copy()
    df2["area_quartile"] = pd.qcut(df2["area"], q=4, labels=["Q1\n(low)","Q2","Q3","Q4\n(high)"])
    means = df2.groupby("area_quartile", observed=True)["convergence_delta"].mean()
    sems  = df2.groupby("area_quartile", observed=True)["convergence_delta"].sem()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(means.index, means.values, yerr=sems.values, capsize=5,
                  color=PALETTE[:4], edgecolor="white", alpha=0.85)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Triangle Area Quartile (diversity proxy)")
    ax.set_ylabel("Mean Convergence Delta ± SE")
    ax.set_title("Convergence Delta by Diversity Quartile", fontweight="bold")
    _save(fig, out / "12_delta_by_area_quartile.png", "Delta by area quartile")


def plot_13_abstract_vs_final_distance(df: pd.DataFrame, out: Path) -> None:
    """Mean abstract distance vs final distance — visualises convergence geometrically."""
    if not all(c in df.columns for c in ["mean_abstract_distance","distance_final_to_centroid"]):
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = _delta_color(df["convergence_delta"].fillna(0))
    ax.scatter(df["mean_abstract_distance"], df["distance_final_to_centroid"],
               c=colors, s=70, alpha=0.8, edgecolors="white", lw=0.4)
    lims = [0, max(df["mean_abstract_distance"].max(), df["distance_final_to_centroid"].max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1.5, label="Equal distance (no convergence)")
    ax.fill_between(lims, [0,0], lims, alpha=0.06, color=NEUTRAL_COLOR)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Mean Individual Abstract Distance to Centroid")
    ax.set_ylabel("Final Synthesis Distance to Centroid")
    ax.set_title("Abstract Distance vs Final Synthesis Distance", fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color="none", label="Below diagonal = moved toward centroid")])
    _save(fig, out / "13_abstract_vs_final_distance.png", "Abstract vs final distance")


def plot_14_historian_distances_to_centroid(df: pd.DataFrame, out: Path) -> None:
    """Per-historian distance to centroid (all 3 historians) as grouped box plots."""
    cols = ["distance_hist1_to_centroid","distance_hist2_to_centroid","distance_hist3_to_centroid"]
    cols = [c for c in cols if c in df.columns]
    if not cols: return
    data  = [df[c].dropna().values for c in cols]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, labels=["Historian 1","Historian 2","Historian 3"])
    for patch, color in zip(bp["boxes"], PALETTE[:3]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_ylabel("Cosine Distance to Centroid")
    ax.set_title("Historian Distance to Triad Centroid", fontweight="bold")
    _save(fig, out / "14_historian_distances_to_centroid.png", "Historian distances to centroid")


def plot_15_abstract_distances_to_centroid(df: pd.DataFrame, out: Path) -> None:
    """Per-abstract distance to centroid overlaid on histogram."""
    cols = ["distance_abstract1_to_centroid","distance_abstract2_to_centroid",
            "distance_abstract3_to_centroid","distance_final_to_centroid"]
    cols = [c for c in cols if c in df.columns]
    if not cols: return
    fig, ax = plt.subplots(figsize=(10, 5))
    lbls = ["Abstract 1","Abstract 2","Abstract 3","Final Synthesis"]
    for col, lbl, c in zip(cols, lbls, PALETTE[:4]):
        sns.kdeplot(df[col].dropna(), ax=ax, label=lbl, color=c, fill=True, alpha=0.25)
    ax.set_xlabel("Cosine Distance to Centroid"); ax.set_ylabel("Density")
    ax.set_title("Distance to Centroid: Individuals vs Synthesis", fontweight="bold")
    ax.legend()
    _save(fig, out / "15_abstract_distances_kde.png", "Abstract distances KDE")


def plot_16_pairwise_abstract_similarity(df: pd.DataFrame, out: Path) -> None:
    """Mean pairwise abstract similarity distribution + convergence overlay."""
    if "mean_pairwise_abstract_similarity" not in df.columns: return
    fig, ax = plt.subplots(figsize=(9, 5))
    med16 = df["convergence_delta"].median() if "convergence_delta" in df.columns else 0
    lo16  = df[df["convergence_delta"] <= med16]["mean_pairwise_abstract_similarity"].dropna()
    hi16  = df[df["convergence_delta"] >  med16]["mean_pairwise_abstract_similarity"].dropna()
    ax.hist(lo16, bins=20, alpha=0.6, color=BIAS_COLOR,    label="Low Δ",  edgecolor="white")
    ax.hist(hi16, bins=20, alpha=0.6, color=NEUTRAL_COLOR, label="High Δ", edgecolor="white")
    ax.set_xlabel("Mean Pairwise Abstract Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Abstract Similarity Distribution by Delta Half", fontweight="bold")
    ax.legend()
    _save(fig, out / "16_pairwise_abstract_similarity.png", "Pairwise abstract similarity")


def plot_17_delta_vs_abstract_variance(df: pd.DataFrame, out: Path) -> None:
    """Abstract distance variance vs convergence delta — tests 'unequal positions' hypothesis."""
    if "abstract_distance_variance" not in df.columns: return
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = _delta_color(df["convergence_delta"].fillna(0))
    ax.scatter(df["abstract_distance_variance"], df["convergence_delta"],
               c=colors, s=65, alpha=0.8, edgecolors="white", lw=0.3)
    mask = df[["abstract_distance_variance","convergence_delta"]].dropna()
    if len(mask) > 2:
        m, b, r, p, _ = stats.linregress(mask["abstract_distance_variance"], mask["convergence_delta"])
        xs = np.linspace(mask["abstract_distance_variance"].min(), mask["abstract_distance_variance"].max(), 100)
        ax.plot(xs, m*xs+b, "navy", ls="--", lw=1.5, label=f"r={r:.2f}, p={p:.3f}")
        ax.legend()
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Abstract Distance Variance (unequal starting positions)")
    ax.set_ylabel("Convergence Delta")
    ax.set_title("Unequal Starting Positions vs Convergence", fontweight="bold")
    _save(fig, out / "17_delta_vs_abstract_variance.png", "Delta vs abstract variance")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – BIAS (plots 18-23)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_18_bias_score_distribution(df: pd.DataFrame, out: Path) -> None:
    """Histogram of bias_score (0=balanced, ~0.67=dominated)."""
    if "bias_score" not in df.columns: return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["bias_score"].dropna(), bins=25, color=BIAS_COLOR, edgecolor="white", alpha=0.85)
    ax.axvline(df["bias_score"].mean(), color="red", ls="--",
               label=f"μ={df['bias_score'].mean():.3f}")
    ax.axvline(0, color="green", ls=":", lw=1.5, label="Perfectly balanced (0)")
    ax.set_xlabel("Bias Score (max_weight − 1/3)")
    ax.set_ylabel("Count")
    ax.set_title("Synthesis Bias Score Distribution", fontweight="bold")
    ax.legend()
    _save(fig, out / "18_bias_score_distribution.png", "Bias score distribution")


def plot_19_dominant_historian_frequency(df: pd.DataFrame, out: Path) -> None:
    """Bar chart: which historian position dominates synthesis most often."""
    if "dominant_historian_position" not in df.columns: return
    counts = df["dominant_historian_position"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([f"Historian {i}" for i in counts.index], counts.values,
                  color=PALETTE[:len(counts)], edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                str(val), ha="center", fontweight="bold")
    expected = len(df) / 3
    ax.axhline(expected, color="gray", ls="--", label=f"Expected if balanced ({expected:.1f})")
    ax.set_ylabel("Times Dominant"); ax.set_title("Which Historian Dominates the Synthesis?", fontweight="bold")
    ax.legend()
    _save(fig, out / "19_dominant_historian_frequency.png", "Dominant historian frequency")


def plot_20_bias_weights_ternary_proxy(df: pd.DataFrame, out: Path) -> None:
    """Scatter of bias_weight_1 vs bias_weight_2 coloured by bias_weight_3."""
    cols = ["bias_weight_1","bias_weight_2","bias_weight_3"]
    if not all(c in df.columns for c in cols): return
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(df["bias_weight_1"], df["bias_weight_2"],
                    c=df["bias_weight_3"], cmap="coolwarm",
                    s=65, alpha=0.8, edgecolors="white", lw=0.3)
    plt.colorbar(sc, ax=ax, label="Bias Weight 3")
    # Equal-weight point
    ax.scatter([1/3], [1/3], s=200, marker="*", color="gold", zorder=5,
               edgecolors="black", lw=0.5, label="Perfect balance (1/3, 1/3, 1/3)")
    ax.set_xlabel("Bias Weight — Historian 1")
    ax.set_ylabel("Bias Weight — Historian 2")
    ax.set_title("Synthesis Bias Weight Space\n(colour = weight of historian 3)", fontweight="bold")
    ax.legend()
    _save(fig, out / "20_bias_weights_scatter.png", "Bias weight scatter")


def plot_21_bias_score_vs_delta(df: pd.DataFrame, out: Path) -> None:
    """Bias score vs convergence delta — tests orthogonality claim."""
    if not all(c in df.columns for c in ["bias_score","convergence_delta"]): return
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = _delta_color(df["convergence_delta"].fillna(0))
    ax.scatter(df["bias_score"], df["convergence_delta"],
               c=colors, s=65, alpha=0.8, edgecolors="white", lw=0.3)
    mask = df[["bias_score","convergence_delta"]].dropna()
    if len(mask) > 2:
        m, b, r, p, _ = stats.linregress(mask["bias_score"], mask["convergence_delta"])
        xs = np.linspace(mask["bias_score"].min(), mask["bias_score"].max(), 100)
        ax.plot(xs, m*xs+b, "navy", ls="--", lw=1.5, label=f"r={r:.2f}, p={p:.3f}")
        ax.legend()
    ax.set_xlabel("Bias Score"); ax.set_ylabel("Convergence Delta")
    ax.set_title("Bias Score vs Convergence Delta\n(testing orthogonality)", fontweight="bold")
    ax.legend()
    _save(fig, out / "21_bias_vs_delta.png", "Bias vs delta")


def plot_22_bias_by_geometry(df: pd.DataFrame, out: Path) -> None:
    """Does triad geometry predict who gets dominated? Area/perimeter vs bias_score."""
    if "bias_score" not in df.columns: return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, feat in zip(axes, ["area","perimeter"]):
        if feat not in df.columns: continue
        ax.scatter(df[feat], df["bias_score"], alpha=0.7,
                   color=BIAS_COLOR, edgecolors="white", lw=0.3, s=55)
        mask = df[[feat,"bias_score"]].dropna()
        if len(mask) > 2:
            m, b, r, p, _ = stats.linregress(mask[feat], mask["bias_score"])
            xs = np.linspace(mask[feat].min(), mask[feat].max(), 100)
            ax.plot(xs, m*xs+b, "navy", ls="--", lw=1.5, label=f"r={r:.2f}, p={p:.3f}")
            ax.legend()
        ax.set_xlabel(feat.capitalize()); ax.set_ylabel("Bias Score")
        ax.set_title(f"{feat.capitalize()} vs Bias Score", fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "22_bias_by_geometry.png", "Bias by geometry")


def plot_23_bias_weight_distributions(df: pd.DataFrame, out: Path) -> None:
    """Box plots of each bias weight — shows if historian position matters."""
    cols = ["bias_weight_1","bias_weight_2","bias_weight_3"]
    cols = [c for c in cols if c in df.columns]
    if not cols: return
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [df[c].dropna().values for c in cols]
    bp = ax.boxplot(data, patch_artist=True, labels=["Hist 1 Weight","Hist 2 Weight","Hist 3 Weight"])
    for patch, color in zip(bp["boxes"], PALETTE[:3]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.axhline(1/3, color="red", ls="--", lw=1.5, label="Equal weight (1/3)")
    ax.set_ylabel("Softmax Bias Weight"); ax.set_title("Bias Weight Distribution by Position", fontweight="bold")
    ax.legend()
    _save(fig, out / "23_bias_weight_distributions.png", "Bias weight distributions")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – PREDICTION MODEL (plots 24-29)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_24_feature_importance(feature_importance_df: pd.DataFrame, out: Path) -> None:
    """Horizontal bar chart of standardised regression coefficients."""
    if feature_importance_df is None or feature_importance_df.empty: return
    df2 = feature_importance_df.sort_values("abs_coefficient", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [NEUTRAL_COLOR if c > 0 else BIAS_COLOR for c in df2["coefficient"]]
    bars = ax.barh(df2["feature"], df2["coefficient"], color=colors, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Standardised Regression Coefficient")
    ax.set_title("Ridge Regression Feature Importance\n(predicting convergence_delta)", fontweight="bold")
    pos_patch = mpatches.Patch(color=NEUTRAL_COLOR, label="Positive → higher delta")
    neg_patch = mpatches.Patch(color=BIAS_COLOR,    label="Negative → lower delta")
    ax.legend(handles=[pos_patch, neg_patch])
    plt.tight_layout()
    _save(fig, out / "24_feature_importance.png", "Feature importance")


def plot_25_actual_vs_predicted(df: pd.DataFrame, model_json: Optional[dict], out: Path) -> None:
    """Reconstruct predictions from saved model JSON and plot actual vs predicted."""
    if model_json is None: return
    feature_names = model_json.get("feature_names", [])
    coef          = np.array(model_json.get("coefficients", []))
    intercept     = model_json.get("intercept", 0.0)
    scaler_mean   = np.array(model_json.get("scaler_mean", []))
    scaler_scale  = np.array(model_json.get("scaler_scale", [1.0]*len(coef)))
    missing = [f for f in feature_names if f not in df.columns]
    if missing: return
    X = df[feature_names].dropna().values
    idx = df[feature_names].dropna().index
    X_scaled = (X - scaler_mean) / scaler_scale
    y_pred = X_scaled @ coef + intercept
    y_true = df.loc[idx, "convergence_delta"].values
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Actual vs predicted
    axes[0].scatter(y_true, y_pred, s=65, alpha=0.8, color=NEUTRAL_COLOR, edgecolors="white", lw=0.3)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lim, lim, "r--", lw=1.5, label=f"Perfect fit (R²={r2:.3f})")
    axes[0].set_xlabel("Actual Delta"); axes[0].set_ylabel("Predicted Delta")
    axes[0].set_title("Actual vs Predicted Convergence Delta", fontweight="bold")
    axes[0].legend()
    # Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, s=65, alpha=0.8, color=ACCENT_COLOR, edgecolors="white", lw=0.3)
    axes[1].axhline(0, color="red", ls="--", lw=1.5)
    axes[1].set_xlabel("Predicted Delta"); axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residual Plot", fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "25_actual_vs_predicted.png", "Actual vs predicted")


def plot_26_coefficient_magnitude_ranked(feature_importance_df: pd.DataFrame, out: Path) -> None:
    """Lollipop chart of |coefficient| ranked — cleaner than bar for many features."""
    if feature_importance_df is None or feature_importance_df.empty: return
    df2 = feature_importance_df.sort_values("abs_coefficient", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hlines(range(len(df2)), 0, df2["abs_coefficient"], color="lightgray", lw=2)
    ax.scatter(df2["abs_coefficient"], range(len(df2)), s=80,
               color=[NEUTRAL_COLOR if c > 0 else BIAS_COLOR for c in df2["coefficient"]],
               zorder=3, edgecolors="white", lw=0.4)
    ax.set_yticks(range(len(df2))); ax.set_yticklabels(df2["feature"])
    ax.set_xlabel("|Coefficient| (standardised)")
    ax.set_title("Feature Importance — Lollipop Ranked", fontweight="bold")
    ax.invert_yaxis()
    _save(fig, out / "26_coefficient_lollipop.png", "Coefficient lollipop")


def plot_27_inference_correlations(inference: Optional[dict], out: Path) -> None:
    """Bar chart of Pearson r values from inference_results.json."""
    if inference is None: return
    corrs = inference.get("correlations_with_delta", {})
    bias_corr = inference.get("bias_score_correlation_with_delta", None)
    if bias_corr is not None:
        corrs["bias_score"] = bias_corr
    if not corrs: return
    features = list(corrs.keys())
    values   = list(corrs.values())
    colors   = [NEUTRAL_COLOR if v > 0 else BIAS_COLOR for v in values]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(features, values, color=colors, edgecolor="white", alpha=0.85)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Pearson r with Convergence Delta")
    ax.set_title("Feature Correlations with Convergence Delta", fontweight="bold")
    ax.set_xticklabels(features, rotation=25, ha="right")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.005 if val >= 0 else -0.015),
                f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    _save(fig, out / "27_inference_correlations.png", "Inference correlations")


def plot_28_ols_area_vs_delta(inference: Optional[dict], df: pd.DataFrame, out: Path) -> None:
    """Area vs delta scatter with OLS fit line and confidence band."""
    if inference is None or "area" not in df.columns or "convergence_delta" not in df.columns: return
    ols = inference.get("ols_area_vs_delta", {})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["area"], df["convergence_delta"], s=65, alpha=0.8,
               color=NEUTRAL_COLOR, edgecolors="white", lw=0.3)
    if ols:
        slope = ols["slope"]; intercept = ols["intercept"]
        r2 = ols["r_squared"]; p = ols["p_value"]
        xs = np.linspace(df["area"].min(), df["area"].max(), 200)
        ax.plot(xs, slope*xs + intercept, "r--", lw=2,
                label=f"OLS: y={slope:.2f}x+{intercept:.3f}\nR²={r2:.3f}, p={p:.3f}")
        ax.legend()
    ax.set_xlabel("Triangle Area (intellectual diversity)")
    ax.set_ylabel("Convergence Delta")
    ax.set_title("OLS: Diversity → Convergence\n(key hypothesis test)", fontweight="bold")
    _save(fig, out / "28_ols_area_vs_delta.png", "OLS area vs delta")


def plot_29_prediction_summary_dashboard(df: pd.DataFrame, inference: Optional[dict],
                                         feature_importance_df: Optional[pd.DataFrame],
                                         out: Path) -> None:
    """4-panel summary dashboard: delta dist, feature imp, correlations, bias vs delta."""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Prediction Model Summary Dashboard", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Panel A: delta distribution
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.hist(df["convergence_delta"].dropna(), bins=20,
              color=NEUTRAL_COLOR, edgecolor="white", alpha=0.85)
    ax_a.axvline(df["convergence_delta"].mean(), color="red", ls="--",
                 label=f"μ={df['convergence_delta'].mean():.3f}")
    ax_a.set_title("A. Delta Distribution"); ax_a.legend(fontsize=8)
    ax_a.set_xlabel("Convergence Delta")

    # Panel B: feature importance
    ax_b = fig.add_subplot(gs[0, 1])
    if feature_importance_df is not None and not feature_importance_df.empty:
        df_fi = feature_importance_df.sort_values("abs_coefficient", ascending=True)
        colors = [NEUTRAL_COLOR if c > 0 else BIAS_COLOR for c in df_fi["coefficient"]]
        ax_b.barh(df_fi["feature"], df_fi["coefficient"], color=colors, edgecolor="white", alpha=0.85)
        ax_b.axvline(0, color="black", lw=1)
        ax_b.set_title("B. Feature Coefficients")
        ax_b.set_xlabel("Coefficient")

    # Panel C: correlations
    ax_c = fig.add_subplot(gs[1, 0])
    if inference:
        corrs = inference.get("correlations_with_delta", {})
        bs    = inference.get("bias_score_correlation_with_delta", None)
        if bs is not None: corrs["bias_score"] = bs
        if corrs:
            vals = list(corrs.values()); fts = list(corrs.keys())
            colors = [NEUTRAL_COLOR if v > 0 else BIAS_COLOR for v in vals]
            ax_c.bar(fts, vals, color=colors, edgecolor="white", alpha=0.85)
            ax_c.axhline(0, color="black", lw=1)
            ax_c.set_xticklabels(fts, rotation=20, ha="right", fontsize=8)
            ax_c.set_title("C. Pearson r with Delta")

    # Panel D: bias vs delta
    ax_d = fig.add_subplot(gs[1, 1])
    if "bias_score" in df.columns and "convergence_delta" in df.columns:
        ax_d.scatter(df["bias_score"], df["convergence_delta"],
                     c=_delta_color(df["convergence_delta"].fillna(0)),
                     s=55, alpha=0.8, edgecolors="white", lw=0.3)
        ax_d.axhline(0, color="gray", ls=":", lw=1)
        ax_d.set_xlabel("Bias Score"); ax_d.set_ylabel("Convergence Delta")
        ax_d.set_title("D. Bias vs Delta (orthogonality)")

    _save(fig, out / "29_prediction_summary_dashboard.png", "Prediction summary dashboard")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – EXPERIMENT OVERVIEW (plots 30-37)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_30_experiment_overview_dashboard(df: pd.DataFrame, out: Path) -> None:
    """High-level experiment dashboard: N triads, convergence rate, delta stats."""
    n = len(df)
    delta_mean = df["convergence_delta"].mean() if "convergence_delta" in df.columns else 0
    delta_std  = df["convergence_delta"].std()  if "convergence_delta" in df.columns else 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Experiment Overview", fontsize=15, fontweight="bold")

    # Quartile bar
    df30 = df.copy()
    df30["delta_q"] = pd.qcut(df30["convergence_delta"], q=4, labels=["Q1","Q2","Q3","Q4"])
    counts30 = df30["delta_q"].value_counts().sort_index()
    axes[0].bar(counts30.index, counts30.values, color=PALETTE[:4], edgecolor="white")
    axes[0].set_title(f"Delta Quartile Counts\n({n} triads)", fontweight="bold")
    axes[0].set_xlabel("Delta Quartile")

    # KPI boxes
    kpis = {"N Triads": n, "Mean Δ": f"{delta_mean:.3f}", "Std Δ": f"{delta_std:.3f}",
            "Min Δ": f"{df['convergence_delta'].min():.3f}" if "convergence_delta" in df.columns else "—",
            "Max Δ": f"{df['convergence_delta'].max():.3f}" if "convergence_delta" in df.columns else "—"}
    axes[1].axis("off")
    y = 0.9
    for k, v in kpis.items():
        axes[1].text(0.1, y, f"{k}:", fontsize=12, fontweight="bold", transform=axes[1].transAxes)
        axes[1].text(0.55, y, str(v), fontsize=12, transform=axes[1].transAxes)
        y -= 0.16
    axes[1].set_title("Key Metrics", fontweight="bold")

    # Delta distribution
    axes[2].hist(df["convergence_delta"].dropna(), bins=20, color=NEUTRAL_COLOR,
                 edgecolor="white", alpha=0.85)
    axes[2].axvline(delta_mean, color="red", ls="--", label=f"μ={delta_mean:.3f}")
    axes[2].set_xlabel("Convergence Delta"); axes[2].set_ylabel("Count")
    axes[2].set_title("Delta Distribution", fontweight="bold")
    axes[2].legend()
    plt.tight_layout()
    _save(fig, out / "30_experiment_overview.png", "Experiment overview dashboard")


def plot_31_historian_participation(proposals_df: pd.DataFrame, out: Path) -> None:
    """How many triads each historian appeared in."""
    if proposals_df is None or proposals_df.empty: return
    counts = proposals_df["historian_name"].value_counts()
    fig, ax = plt.subplots(figsize=(max(10, len(counts)*0.5), 5))
    ax.bar(counts.index, counts.values, color=NEUTRAL_COLOR, edgecolor="white", alpha=0.85)
    ax.axhline(counts.mean(), color="red", ls="--", label=f"Mean={counts.mean():.1f}")
    ax.set_xlabel("Historian"); ax.set_ylabel("Appearances in Triads")
    ax.set_title("Historian Participation Frequency", fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.legend()
    plt.tight_layout()
    _save(fig, out / "31_historian_participation.png", "Historian participation")


def plot_32_proposal_abstract_length(proposals_df: pd.DataFrame, out: Path) -> None:
    """Abstract length distribution per historian position."""
    if proposals_df is None or proposals_df.empty or "abstract" not in proposals_df.columns: return
    proposals_df = proposals_df.copy()
    proposals_df["abstract_len"] = proposals_df["abstract"].str.len()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # By position
    for pos in [1, 2, 3]:
        d = proposals_df[proposals_df["historian_position"]==pos]["abstract_len"].dropna()
        if len(d): sns.kdeplot(d, ax=axes[0], label=f"Position {pos}", fill=True, alpha=0.3)
    axes[0].set_xlabel("Abstract Length (chars)"); axes[0].set_ylabel("Density")
    axes[0].set_title("Abstract Length by Historian Position", fontweight="bold")
    axes[0].legend()
    # Overall
    axes[1].hist(proposals_df["abstract_len"].dropna(), bins=25,
                 color=ACCENT_COLOR, edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("Abstract Length (chars)"); axes[1].set_ylabel("Count")
    axes[1].set_title("Overall Hypothesis Abstract Lengths", fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "32_hypothesis_abstract_length.png", "Hypothesis abstract length")


def plot_33_synthesis_abstract_length(synthesis_df: pd.DataFrame, proposals_df: pd.DataFrame, out: Path) -> None:
    """Compare synthesis abstract length vs individual hypotheses."""
    if synthesis_df is None or synthesis_df.empty: return
    fig, ax = plt.subplots(figsize=(8, 5))
    syn_len = synthesis_df["final_abstract"].str.len().dropna()
    ax.hist(syn_len, bins=20, alpha=0.6, color=ACCENT_COLOR,
            edgecolor="white", label="Synthesis")
    if proposals_df is not None and "abstract" in proposals_df.columns:
        prop_len = proposals_df["abstract"].str.len().dropna()
        ax.hist(prop_len, bins=20, alpha=0.6, color=NEUTRAL_COLOR,
                edgecolor="white", label="Individual Hypotheses")
    ax.set_xlabel("Abstract Length (chars)"); ax.set_ylabel("Count")
    ax.set_title("Synthesis vs Individual Abstract Length", fontweight="bold")
    ax.legend()
    _save(fig, out / "33_synthesis_vs_hypothesis_length.png", "Synthesis vs hypothesis length")


def plot_34_source_usage(proposals_df: pd.DataFrame, out: Path) -> None:
    """Distribution of n_text_sources and n_image_sources per hypothesis."""
    if proposals_df is None or proposals_df.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, lbl, color in zip(axes,
                                    ["n_text_sources","n_image_sources"],
                                    ["Text Sources","Image Sources"],
                                    [NEUTRAL_COLOR, ACCENT_COLOR]):
        if col not in proposals_df.columns: continue
        vc = proposals_df[col].value_counts().sort_index()
        ax.bar(vc.index.astype(str), vc.values, color=color, edgecolor="white", alpha=0.85)
        ax.set_xlabel(f"Number of {lbl}"); ax.set_ylabel("Hypotheses")
        ax.set_title(f"{lbl} per Hypothesis", fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "34_source_usage.png", "Source usage")


def plot_35_convergence_by_historian_pair(df_merged: pd.DataFrame, out: Path) -> None:
    """
    Heatmap-style: for each pair of historian names, mean convergence_delta.
    Only feasible with the triads CSV which has historian_1_name etc.
    """
    needed = ["historian_1_name","historian_2_name","historian_3_name","convergence_delta"]
    if not all(c in df_merged.columns for c in needed): return
    pairs = []
    for _, row in df_merged.iterrows():
        delta = row["convergence_delta"]
        for a, b in [("historian_1_name","historian_2_name"),
                     ("historian_2_name","historian_3_name"),
                     ("historian_1_name","historian_3_name")]:
            h1 = row[a].split()[-1] if isinstance(row[a], str) else str(row[a])
            h2 = row[b].split()[-1] if isinstance(row[b], str) else str(row[b])
            pairs.append({"h1": h1, "h2": h2, "delta": delta})
    pdf = pd.DataFrame(pairs)
    pivot = pdf.pivot_table(index="h1", columns="h2", values="delta", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(8, len(pivot)*0.7), max(6, len(pivot)*0.6)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
                linewidths=0.3, ax=ax, cbar_kws={"label":"Mean Convergence Delta"})
    ax.set_title("Mean Convergence Delta by Historian Pair", fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "35_convergence_by_historian_pair.png", "Convergence by historian pair")


def plot_36_triad_geometry_ranked(df: pd.DataFrame, out: Path) -> None:
    """All triads ranked by convergence_delta, coloured by converged."""
    if "convergence_delta" not in df.columns: return
    df2 = df.sort_values("convergence_delta").reset_index(drop=True)
    colors = _delta_color(df2["convergence_delta"].fillna(0))
    fig, ax = plt.subplots(figsize=(max(10, len(df2)*0.4), 5))
    ax.bar(range(len(df2)), df2["convergence_delta"], color=colors, edgecolor="white", width=0.7)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Triad (ranked by delta)"); ax.set_ylabel("Convergence Delta")
    ax.set_title("All Triads Ranked by Convergence Delta", fontweight="bold")
    sm36 = plt.cm.ScalarMappable(cmap="RdYlGn",
        norm=plt.Normalize(df2["convergence_delta"].min(), df2["convergence_delta"].max()))
    sm36.set_array([])
    fig.colorbar(sm36, ax=ax, label="Convergence Delta")
    plt.tight_layout()
    _save(fig, out / "36_triads_ranked_by_delta.png", "Triads ranked by delta")


def plot_37_comprehensive_correlation_matrix(df: pd.DataFrame, out: Path) -> None:
    """Full correlation matrix across all numeric columns in the merged dataset."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Drop id-like columns
    num_cols = [c for c in num_cols if "id" not in c.lower() and "seq" not in c.lower()]
    if len(num_cols) < 2: return
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(max(12, len(num_cols)), max(10, len(num_cols)*0.8)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.3, annot_kws={"size":7}, ax=ax)
    ax.set_title("Full Correlation Matrix (all numeric features)", fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "37_full_correlation_matrix.png", "Full correlation matrix")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – ADVANCED / INTERESTING (plots 38-50)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_38_pca_embedding_of_triads(df: pd.DataFrame, out: Path) -> None:
    """
    PCA on geometry+convergence features to find natural clusters of triad types.
    """
    feat_cols = ["perimeter","area","angle_variance","mean_historian_distance",
                 "abstract_distance_variance","mean_pairwise_abstract_similarity",
                 "bias_score","convergence_delta"]
    feat_cols = [c for c in feat_cols if c in df.columns]
    sub = df[feat_cols].dropna()
    if len(sub) < 4: return
    scaler = StandardScaler()
    X = scaler.fit_transform(sub)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=sub["convergence_delta"],
                    cmap="RdYlGn", s=80, alpha=0.85, edgecolors="white", lw=0.4)
    plt.colorbar(sc, ax=ax, label="Convergence Delta")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)")
    ax.set_title("PCA of Triad Feature Space\n(colour = convergence delta)", fontweight="bold")
    _save(fig, out / "38_pca_triad_space.png", "PCA triad feature space")


def plot_39_delta_vs_mean_historian_distance(df: pd.DataFrame, out: Path) -> None:
    """Detailed look at the strongest predictor: mean_historian_distance vs delta."""
    if not all(c in df.columns for c in ["mean_historian_distance","convergence_delta"]): return
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = _delta_color(df["convergence_delta"].fillna(0))
    ax.scatter(df["mean_historian_distance"], df["convergence_delta"],
               c=colors, s=75, alpha=0.85, edgecolors="white", lw=0.4, zorder=3)
    mask = df[["mean_historian_distance","convergence_delta"]].dropna()
    if len(mask) > 2:
        m, b, r, p, se = stats.linregress(mask["mean_historian_distance"], mask["convergence_delta"])
        xs = np.linspace(mask["mean_historian_distance"].min(),
                         mask["mean_historian_distance"].max(), 200)
        ys = m*xs + b
        # 95% CI band
        n = len(mask)
        t_crit = stats.t.ppf(0.975, df=n-2)
        se_fit = se * np.sqrt(1/n + (xs - xs.mean())**2 / np.sum((xs - xs.mean())**2))
        ax.plot(xs, ys, "navy", lw=2, label=f"OLS: r={r:.3f}, p={p:.3f}")
        ax.fill_between(xs, ys - t_crit*se_fit, ys + t_crit*se_fit, alpha=0.15, color="navy")
        ax.legend()
    ax.set_xlabel("Mean Historian Distance to Centroid")
    ax.set_ylabel("Convergence Delta")
    ax.set_title("Strongest Predictor: Historian Distance vs Convergence\n(with 95% CI)", fontweight="bold")
    ax.axhline(0, color="gray", ls=":", lw=1)
    _save(fig, out / "39_mean_historian_dist_vs_delta.png", "Mean historian distance vs delta")


def plot_40_diversity_convergence_quadrant(df: pd.DataFrame, out: Path) -> None:
    """
    Quadrant plot: x=area (diversity), y=convergence_delta.
    Highlights which quadrant triads fall in.
    """
    if not all(c in df.columns for c in ["area","convergence_delta"]): return
    med_area  = df["area"].median()
    med_delta = df["convergence_delta"].median()
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = _delta_color(df["convergence_delta"].fillna(0))
    ax.scatter(df["area"], df["convergence_delta"], c=colors, s=75,
               alpha=0.85, edgecolors="white", lw=0.4, zorder=3)
    ax.axvline(med_area,  color="gray", ls="--", lw=1.2)
    ax.axhline(med_delta, color="gray", ls="--", lw=1.2)
    # Quadrant labels
    for x, y, lbl in [
        (0.97, 0.95, "High diversity\nHigh convergence"),
        (0.03, 0.95, "Low diversity\nHigh convergence"),
        (0.97, 0.05, "High diversity\nLow convergence"),
        (0.03, 0.05, "Low diversity\nLow convergence"),
    ]:
        ax.text(x, y, lbl, transform=ax.transAxes, fontsize=8,
                ha="right" if x > 0.5 else "left",
                va="top"   if y > 0.5 else "bottom",
                color="gray", style="italic")
    ax.set_xlabel("Triangle Area (intellectual diversity)")
    ax.set_ylabel("Convergence Delta")
    ax.set_title("Diversity–Convergence Quadrant Plot", fontweight="bold")
    _save(fig, out / "40_diversity_convergence_quadrant.png", "Diversity convergence quadrant")


def plot_41_cumulative_delta_distribution(df: pd.DataFrame, out: Path) -> None:
    """ECDF of convergence_delta — shows full shape of distribution."""
    if "convergence_delta" not in df.columns: return
    data = df["convergence_delta"].dropna().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.sort(data.values), np.linspace(0, 1, len(data)), lw=2,
            label="All triads", color=NEUTRAL_COLOR)
    for label, mask, color in [
        ("Lower half (Δ < median)", data <= data.median(), BIAS_COLOR),
        ("Upper half (Δ ≥ median)", data >  data.median(), ACCENT_COLOR),
    ]:
        sub = data[mask]
        if len(sub):
            ax.plot(np.sort(sub), np.linspace(0, 1, len(sub)), lw=1.5,
                    ls="--", label=label, color=color)
    ax.axvline(0, color="black", ls=":", lw=1)
    ax.set_xlabel("Convergence Delta"); ax.set_ylabel("Cumulative Proportion")
    ax.set_title("Empirical CDF of Convergence Delta", fontweight="bold")
    ax.legend()
    _save(fig, out / "41_ecdf_convergence_delta.png", "ECDF convergence delta")


def plot_42_perimeter_area_3d(df: pd.DataFrame, out: Path) -> None:
    """3D scatter: perimeter × area × convergence_delta."""
    if not all(c in df.columns for c in ["perimeter","area","convergence_delta"]): return
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(df["perimeter"], df["area"], df["convergence_delta"],
                    c=df["convergence_delta"], cmap="RdYlGn",
                    s=60, alpha=0.8, edgecolors="white", lw=0.3)
    plt.colorbar(sc, ax=ax, label="Convergence Delta", pad=0.1)
    ax.set_xlabel("Perimeter"); ax.set_ylabel("Area"); ax.set_zlabel("Convergence Delta")
    ax.set_title("3D Geometry–Convergence Space", fontweight="bold")
    _save(fig, out / "42_3d_geometry_convergence.png", "3D geometry convergence")


def plot_43_bias_vs_historian_distance(df: pd.DataFrame, out: Path) -> None:
    """Does a more spread triad lead to more biased synthesis?"""
    if not all(c in df.columns for c in ["mean_historian_distance","bias_score"]): return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["mean_historian_distance"], df["bias_score"],
               c=df["convergence_delta"], cmap="RdYlGn",
               s=65, alpha=0.85, edgecolors="white", lw=0.3)
    sc = ax.scatter(df["mean_historian_distance"], df["bias_score"],
                    c=df["convergence_delta"], cmap="RdYlGn",
                    s=65, alpha=0.85, edgecolors="white", lw=0.3)
    plt.colorbar(sc, ax=ax, label="Convergence Delta")
    mask = df[["mean_historian_distance","bias_score"]].dropna()
    if len(mask) > 2:
        m, b, r, p, _ = stats.linregress(mask["mean_historian_distance"], mask["bias_score"])
        xs = np.linspace(mask["mean_historian_distance"].min(),
                         mask["mean_historian_distance"].max(), 100)
        ax.plot(xs, m*xs+b, "navy", ls="--", lw=1.5, label=f"r={r:.2f}, p={p:.3f}")
        ax.legend()
    ax.set_xlabel("Mean Historian Distance"); ax.set_ylabel("Bias Score")
    ax.set_title("Does Spread Predict Bias?\nHistorian Distance vs Synthesis Bias", fontweight="bold")
    _save(fig, out / "43_historian_distance_vs_bias.png", "Historian distance vs bias")


def plot_44_dominant_position_vs_delta(df: pd.DataFrame, out: Path) -> None:
    """Box plot: convergence_delta by dominant historian position."""
    if not all(c in df.columns for c in ["dominant_historian_position","convergence_delta"]): return
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [df[df["dominant_historian_position"]==i]["convergence_delta"].dropna().values
              for i in [1, 2, 3]]
    bp = ax.boxplot(groups, patch_artist=True, labels=["Dominant: H1","Dominant: H2","Dominant: H3"])
    for patch, color in zip(bp["boxes"], PALETTE[:3]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.axhline(df["convergence_delta"].mean(), color="red", ls="--",
               label="Overall mean delta")
    ax.set_ylabel("Convergence Delta")
    ax.set_title("Convergence Delta by Dominant Historian", fontweight="bold")
    ax.legend()
    _save(fig, out / "44_dominant_position_vs_delta.png", "Dominant position vs delta")


def plot_45_distance_variance_decomposition(df: pd.DataFrame, out: Path) -> None:
    """Stacked histogram showing where variance in delta comes from."""
    if "convergence_delta" not in df.columns: return
    feats = ["perimeter","mean_historian_distance","abstract_distance_variance","bias_score"]
    feats = [f for f in feats if f in df.columns]
    # Compute partial correlations (absolute r) as pie
    corrs = {}
    for f in feats:
        mask = df[[f,"convergence_delta"]].dropna()
        if len(mask) > 2:
            r = abs(np.corrcoef(mask[f], mask["convergence_delta"])[0,1])
            corrs[f] = r
    if not corrs: return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(list(corrs.values()), labels=list(corrs.keys()),
           autopct="%1.1f%%", startangle=90,
           colors=PALETTE[:len(corrs)],
           wedgeprops={"edgecolor":"white","lw":1.5})
    ax.set_title("|Pearson r| Share of Convergence Delta Variance\n(not causal — descriptive only)",
                 fontweight="bold")
    _save(fig, out / "45_correlation_share_pie.png", "Correlation share pie")


def plot_46_synthesis_question_length(synthesis_df: pd.DataFrame, out: Path) -> None:
    """Length of final_research_question — proxy for specificity of synthesis."""
    if synthesis_df is None or synthesis_df.empty: return
    if "final_research_question" not in synthesis_df.columns: return
    synthesis_df = synthesis_df.copy()
    synthesis_df["q_len"] = synthesis_df["final_research_question"].str.len()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(synthesis_df["q_len"].dropna(), bins=20,
            color=BIAS_COLOR, edgecolor="white", alpha=0.85)
    ax.axvline(synthesis_df["q_len"].mean(), color="red", ls="--",
               label=f"μ={synthesis_df['q_len'].mean():.0f} chars")
    ax.set_xlabel("Final Research Question Length (chars)")
    ax.set_ylabel("Count")
    ax.set_title("Synthesis Research Question Length\n(proxy for specificity)", fontweight="bold")
    ax.legend()
    _save(fig, out / "46_synthesis_question_length.png", "Synthesis question length")


def plot_47_convergence_delta_violin(df: pd.DataFrame, out: Path) -> None:
    """Violin plot of convergence_delta by delta quartile."""
    if "convergence_delta" not in df.columns: return
    fig, ax = plt.subplots(figsize=(9, 6))
    df2 = df[["convergence_delta"]].dropna().copy()
    df2["Quartile"] = pd.qcut(df2["convergence_delta"], q=4,
                               labels=["Q1 (low)","Q2","Q3","Q4 (high)"])
    sns.violinplot(data=df2, x="Quartile", y="convergence_delta", ax=ax,
                   palette=dict(zip(["Q1 (low)","Q2","Q3","Q4 (high)"], PALETTE[:4])),
                   inner="box", cut=0)
    ax.axhline(df2["convergence_delta"].mean(), color="red", ls="--", lw=1.5,
               label=f"Mean={df2['convergence_delta'].mean():.3f}")
    ax.legend()
    ax.set_ylabel("Convergence Delta")
    ax.set_title("Convergence Delta by Quartile", fontweight="bold")
    _save(fig, out / "47_delta_violin.png", "Delta violin")


def plot_48_perimeter_quartile_bias(df: pd.DataFrame, out: Path) -> None:
    """Mean bias score by perimeter quartile."""
    if not all(c in df.columns for c in ["perimeter","bias_score"]): return
    df2 = df.copy()
    df2["perim_q"] = pd.qcut(df2["perimeter"], q=4, labels=["Q1","Q2","Q3","Q4"])
    means = df2.groupby("perim_q", observed=True)["bias_score"].mean()
    sems  = df2.groupby("perim_q", observed=True)["bias_score"].sem()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(means.index, means.values, yerr=sems.values, capsize=5,
           color=PALETTE[:4], edgecolor="white", alpha=0.85)
    ax.axhline(df["bias_score"].mean(), color="red", ls="--", label="Overall mean bias")
    ax.set_xlabel("Perimeter Quartile (Q1=low, Q4=high diversity)")
    ax.set_ylabel("Mean Bias Score ± SE")
    ax.set_title("Synthesis Bias by Triangle Perimeter Quartile", fontweight="bold")
    ax.legend()
    _save(fig, out / "48_perimeter_quartile_bias.png", "Perimeter quartile bias")


def plot_49_all_distances_scatter_matrix(df: pd.DataFrame, out: Path) -> None:
    """Scatter matrix of all centroid-distance columns."""
    dist_cols = ["distance_hist1_to_centroid","distance_hist2_to_centroid",
                 "distance_hist3_to_centroid","mean_abstract_distance",
                 "distance_final_to_centroid"]
    dist_cols = [c for c in dist_cols if c in df.columns]
    if len(dist_cols) < 2: return
    short = {c: c.replace("distance_","").replace("_to_centroid","").replace("mean_","μ_")
             for c in dist_cols}
    sub = df[dist_cols].dropna().rename(columns=short)
    g = sns.pairplot(sub, diag_kind="kde", plot_kws={"alpha":0.5, "s":30, "color": NEUTRAL_COLOR})
    g.figure.suptitle("Pairwise Distance Variables Scatter Matrix", y=1.02, fontweight="bold")
    _save(g.figure, out / "49_distance_scatter_matrix.png", "Distance scatter matrix")


def plot_50_final_summary_heatmap(df: pd.DataFrame, out: Path) -> None:
    """
    Per-triad heatmap: rows=triads, columns=key metrics, normalised.
    Gives a visual fingerprint of every triad in one plot.
    """
    key_cols = ["perimeter","area","angle_variance","mean_historian_distance",
                "abstract_distance_variance","mean_pairwise_abstract_similarity",
                "convergence_delta","bias_score"]
    key_cols = [c for c in key_cols if c in df.columns]
    if len(key_cols) < 2: return
    sub = df[key_cols].dropna().reset_index(drop=True)
    # Z-score normalise
    norm = (sub - sub.mean()) / (sub.std() + 1e-9)
    fig, ax = plt.subplots(figsize=(max(12, len(key_cols)*1.5), max(8, len(sub)*0.35)))
    sns.heatmap(norm, xticklabels=key_cols, yticklabels=[f"Triad {i+1}" for i in sub.index],
                cmap="coolwarm", center=0, linewidths=0.2, annot=len(sub)<=25,
                fmt=".1f", annot_kws={"size":7}, ax=ax)
    ax.set_title("Per-Triad Feature Fingerprint (z-scored)", fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save(fig, out / "50_per_triad_fingerprint_heatmap.png", "Per-triad fingerprint heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – PERPLEXITY (plots 51-56)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_perplexity_df(props_df: pd.DataFrame, synth_df: pd.DataFrame,
                           model_name: str = "gpt2") -> pd.DataFrame:
    """
    Compute per-triad perplexity features using GPT-2.
    Returns DataFrame with triad_id and all ppl columns.
    Caches result to avoid re-computation if called multiple times.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {model_name} for perplexity computation…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    print(f"  GPT-2 loaded on {device}")

    # Smoke-test on one token to catch issues early
    try:
        _test_enc = tokenizer("test", return_tensors="pt")
        _test_enc = {k: v.to(device) for k, v in _test_enc.items()}
        with torch.no_grad():
            _test_loss = model(**_test_enc, labels=_test_enc["input_ids"]).loss
        print(f"  GPT-2 smoke test passed (loss={_test_loss.item():.3f})")
    except Exception as e:
        import traceback
        print(f"  !! GPT-2 smoke test FAILED: {e}\n{traceback.format_exc()}")
        raise

    def _ppl(text: str, max_length: int = 512) -> float:
        if not text or not isinstance(text, str) or not text.strip():
            return np.nan
        try:
            enc = tokenizer(text, return_tensors="pt",
                            max_length=max_length, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                loss = model(**enc, labels=enc["input_ids"]).loss
            return float(torch.exp(loss).item())
        except Exception as e:
            import traceback
            print(f"  !! _ppl failed: {e}\n{traceback.format_exc()}")
            return np.nan

    rows = []
    triad_ids = props_df["triad_id"].unique()
    for i, tid in enumerate(triad_ids):
        if (i + 1) % 20 == 0:
            print(f"  perplexity {i+1}/{len(triad_ids)}…")
        prop_rows = props_df[props_df["triad_id"] == tid]
        syn_rows  = synth_df[synth_df["triad_id"] == tid]
        if prop_rows.empty or syn_rows.empty:
            continue
        ppls = [_ppl(r["abstract"]) for _, r in prop_rows.iterrows()]
        syn_ppl = _ppl(syn_rows.iloc[0]["final_abstract"])
        rows.append({
            "triad_id":         tid,
            "proposal_ppl_mean": float(np.nanmean(ppls)),
            "proposal_ppl_std":  float(np.nanstd(ppls)),
            "proposal_ppl_min":  float(np.nanmin(ppls)),
            "proposal_ppl_max":  float(np.nanmax(ppls)),
            "synthesis_ppl":     syn_ppl,
            "ppl_delta":         syn_ppl - float(np.nanmean(ppls)),
        })
    print(f"  Perplexity computed for {len(rows)}/{len(triad_ids)} triads")
    if rows:
        sample = rows[0]
        print(f"  Sample: { {k: round(v,2) if isinstance(v,float) else v for k,v in sample.items()} }")
    return pd.DataFrame(rows)


def _load_or_compute_ppl(data_dir: Path, props_df: pd.DataFrame,
                         synth_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Load cached perplexity CSV or compute from scratch."""
    cache = data_dir / "perplexity_features.csv"
    if cache.exists():
        print(f"  Loading cached perplexity from {cache.name}")
        return pd.read_csv(cache)
    if props_df is None or props_df.empty or synth_df is None or synth_df.empty:
        print(f"  !! proposals.csv empty/missing: {props_df is None or props_df.empty}")
        print(f"  !! synthesis.csv empty/missing: {synth_df is None or synth_df.empty}")
        return None
    try:
        ppl_df = _compute_perplexity_df(props_df, synth_df)
        ppl_df.to_csv(cache, index=False)
        print(f"  Saved perplexity cache → {cache.name}")
        return ppl_df
    except Exception as e:
        import traceback
        print(f"\n  !! Perplexity computation failed: {e}")
        print(f"  !! Traceback:\n{traceback.format_exc()}")
        return None


def plot_51_perplexity_distributions(ppl_df: pd.DataFrame, out: Path) -> None:
    """Distributions of hypothesis perplexity (mean/std/min/max) and synthesis perplexity."""
    if ppl_df is None or ppl_df.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Perplexity Feature Distributions", fontsize=15, fontweight="bold")

    specs = [
        ("proposal_ppl_mean", "Mean Hypothesis Perplexity",  NEUTRAL_COLOR),
        ("proposal_ppl_std",  "Std Hypothesis Perplexity\n(inter-historian disagreement)", PALETTE[1]),
        ("proposal_ppl_min",  "Min Hypothesis Perplexity",   PALETTE[2]),
        ("proposal_ppl_max",  "Max Hypothesis Perplexity",   PALETTE[3]),
        ("synthesis_ppl",     "Synthesis Perplexity",      BIAS_COLOR),
        ("ppl_delta",         "Perplexity Delta\n(synthesis − mean hypothesis)", ACCENT_COLOR),
    ]
    for ax, (col, lbl, color) in zip(axes.flat, specs):
        if col not in ppl_df.columns:
            ax.set_visible(False); continue
        data = ppl_df[col].dropna()
        ax.hist(data, bins=25, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(),   color="red",   ls="--", lw=1.5, label=f"μ={data.mean():.1f}")
        ax.axvline(data.median(), color="green", ls=":",  lw=1.5, label=f"med={data.median():.1f}")
        if col == "ppl_delta":
            ax.axvline(0, color="black", ls="-", lw=1, alpha=0.5)
        ax.set_title(lbl, fontsize=10); ax.set_xlabel(lbl.split("\n")[0])
        ax.set_ylabel("Count"); ax.legend(fontsize=7)
    plt.tight_layout()
    _save(fig, out / "51_perplexity_distributions.png", "Perplexity distributions")


def plot_52_proposal_vs_synthesis_perplexity(ppl_df: pd.DataFrame, out: Path) -> None:
    """Scatter: mean hypothesis perplexity vs synthesis perplexity with identity line."""
    if ppl_df is None or ppl_df.empty: return
    need = ["proposal_ppl_mean", "synthesis_ppl"]
    if not all(c in ppl_df.columns for c in need): return
    sub = ppl_df[need].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Hypothesis vs Synthesis Perplexity", fontsize=14, fontweight="bold")

    # Scatter with identity line
    axes[0].scatter(sub["proposal_ppl_mean"], sub["synthesis_ppl"],
                    s=60, alpha=0.75, color=NEUTRAL_COLOR, edgecolors="white", lw=0.3)
    lo = min(sub["proposal_ppl_mean"].min(), sub["synthesis_ppl"].min())
    hi = max(sub["proposal_ppl_mean"].max(), sub["synthesis_ppl"].max())
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x (no change)")
    axes[0].set_xlabel("Mean Hypothesis Perplexity"); axes[0].set_ylabel("Synthesis Perplexity")
    axes[0].set_title("Hypothesis vs Synthesis (red = no change)", fontweight="bold")
    axes[0].legend()

    # ppl_delta distribution with zero
    if "ppl_delta" in ppl_df.columns:
        delta = ppl_df["ppl_delta"].dropna()
        axes[1].hist(delta, bins=25, color=BIAS_COLOR, edgecolor="white", alpha=0.85)
        axes[1].axvline(0, color="red",  ls="--", lw=2,   label="No change (Δ=0)")
        axes[1].axvline(delta.mean(), color="navy", ls="--", lw=1.5,
                        label=f"μ={delta.mean():.1f}")
        frac_simpler = (delta < 0).mean()
        axes[1].set_title(f"Perplexity Delta  ({frac_simpler:.0%} syntheses simpler than hypotheses)",
                          fontweight="bold")
        axes[1].set_xlabel("Synthesis PPL − Mean Hypothesis PPL"); axes[1].set_ylabel("Count")
        axes[1].legend()
    plt.tight_layout()
    _save(fig, out / "52_proposal_vs_synthesis_ppl.png", "Hypothesis vs synthesis perplexity")


def plot_53_perplexity_vs_convergence(ppl_df: pd.DataFrame, df_merged: pd.DataFrame, out: Path) -> None:
    """Scatter panels: proposal_ppl_mean, synthesis_ppl, ppl_delta vs convergence_delta."""
    if ppl_df is None or ppl_df.empty or df_merged.empty: return
    merged = df_merged.merge(ppl_df, on="triad_id", how="inner")
    if "convergence_delta" not in merged.columns: return

    ppl_cols = [("proposal_ppl_mean", "Mean Hypothesis PPL", NEUTRAL_COLOR),
                ("synthesis_ppl",     "Synthesis PPL",     BIAS_COLOR),
                ("ppl_delta",         "PPL Delta",         ACCENT_COLOR),
                ("proposal_ppl_std",  "Hypothesis PPL Std\n(historian disagreement)", PALETTE[1])]
    ppl_cols = [(c, l, col) for c, l, col in ppl_cols if c in merged.columns]
    if not ppl_cols: return

    ncols = 2; nrows = (len(ppl_cols) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows))
    fig.suptitle("Perplexity vs Convergence Delta", fontsize=14, fontweight="bold")
    axes = axes.flat
    for ax, (col, lbl, color) in zip(axes, ppl_cols):
        sub = merged[[col, "convergence_delta"]].dropna()
        colors_pts = _delta_color(merged.loc[sub.index, "convergence_delta"].fillna(0))
        ax.scatter(sub[col], sub["convergence_delta"], c=colors_pts,
                   s=55, alpha=0.75, edgecolors="white", lw=0.3)
        if len(sub) > 2:
            m, b, r, p, _ = stats.linregress(sub[col], sub["convergence_delta"])
            xs = np.linspace(sub[col].min(), sub[col].max(), 100)
            ax.plot(xs, m * xs + b, "navy", ls="--", lw=1.5, label=f"r={r:.3f}, p={p:.3f}")
            ax.legend(fontsize=8)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel(lbl.split("\n")[0]); ax.set_ylabel("Convergence Delta")
        ax.set_title(f"{lbl} vs Convergence Delta", fontweight="bold")
    for ax in list(axes)[len(ppl_cols):]:
        ax.set_visible(False)
    plt.tight_layout()
    _save(fig, out / "53_perplexity_vs_convergence.png", "Perplexity vs convergence")


def plot_54_perplexity_vs_bias(ppl_df: pd.DataFrame, df_merged: pd.DataFrame, out: Path) -> None:
    """Is synthesis linguistic complexity linked to how biased the synthesis was?"""
    if ppl_df is None or ppl_df.empty or df_merged.empty: return
    merged = df_merged.merge(ppl_df, on="triad_id", how="inner")
    need = ["synthesis_ppl", "ppl_delta", "bias_score"]
    need = [c for c in need if c in merged.columns]
    if len(need) < 2: return

    fig, axes = plt.subplots(1, len(need) - 1 if "bias_score" in need else 0,
                             figsize=(7 * max(1, len(need) - 1), 5))
    if not hasattr(axes, "__len__"):
        axes = [axes]
    fig.suptitle("Perplexity vs Synthesis Bias", fontsize=14, fontweight="bold")

    ppl_targets = [(c, l) for c, l in [("synthesis_ppl", "Synthesis PPL"),
                                        ("ppl_delta", "PPL Delta")] if c in merged.columns]
    for ax, (pcol, plbl) in zip(axes, ppl_targets):
        if "bias_score" not in merged.columns: continue
        sub = merged[[pcol, "bias_score"]].dropna()
        sc = ax.scatter(sub["bias_score"], sub[pcol],
                        c=sub[pcol], cmap="viridis", s=60, alpha=0.8,
                        edgecolors="white", lw=0.3)
        plt.colorbar(sc, ax=ax, label=plbl)
        if len(sub) > 2:
            m, b, r, p, _ = stats.linregress(sub["bias_score"], sub[pcol])
            xs = np.linspace(sub["bias_score"].min(), sub["bias_score"].max(), 100)
            ax.plot(xs, m * xs + b, "red", ls="--", lw=1.5, label=f"r={r:.3f}, p={p:.3f}")
            ax.legend(fontsize=8)
        ax.set_xlabel("Bias Score"); ax.set_ylabel(plbl)
        ax.set_title(f"Bias Score vs {plbl}", fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "54_perplexity_vs_bias.png", "Perplexity vs bias")


def plot_55_perplexity_vs_source_diversity(ppl_df: pd.DataFrame, df_merged: pd.DataFrame, out: Path) -> None:
    """Does having more diverse sources predict higher linguistic complexity?"""
    if ppl_df is None or ppl_df.empty or df_merged.empty: return
    merged = df_merged.merge(ppl_df, on="triad_id", how="inner")
    src_cols = [c for c in ["mean_source_embedding_distance", "source_embedding_variance"]
                if c in merged.columns]
    ppl_targets = [c for c in ["proposal_ppl_mean", "synthesis_ppl"] if c in merged.columns]
    if not src_cols or not ppl_targets: return

    nrows = len(src_cols); ncols = len(ppl_targets)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    fig.suptitle("Source Diversity vs Linguistic Complexity (Perplexity)",
                 fontsize=14, fontweight="bold")
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for i, scol in enumerate(src_cols):
        for j, pcol in enumerate(ppl_targets):
            ax = axes[i, j]
            sub = merged[[scol, pcol]].dropna()
            ax.scatter(sub[scol], sub[pcol], s=55, alpha=0.75,
                       color=NEUTRAL_COLOR, edgecolors="white", lw=0.3)
            if len(sub) > 2:
                m, b, r, p, _ = stats.linregress(sub[scol], sub[pcol])
                xs = np.linspace(sub[scol].min(), sub[scol].max(), 100)
                ax.plot(xs, m * xs + b, "navy", ls="--", lw=1.5, label=f"r={r:.3f}, p={p:.3f}")
                ax.legend(fontsize=8)
            ax.set_xlabel(scol.replace("_", " ").title())
            ax.set_ylabel(pcol.replace("_", " ").title())
            ax.set_title(f"{scol.split('_')[1].title()} Diversity vs {pcol.split('_')[0].title()} PPL",
                         fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "55_perplexity_vs_source_diversity.png", "Perplexity vs source diversity")


def plot_56_perplexity_full_correlation_heatmap(ppl_df: pd.DataFrame, df_merged: pd.DataFrame,
                                                out: Path) -> None:
    """Correlation heatmap: all perplexity features × all convergence/bias/geometry features."""
    if ppl_df is None or ppl_df.empty or df_merged.empty: return
    merged = df_merged.merge(ppl_df, on="triad_id", how="inner")
    ppl_cols = [c for c in ["proposal_ppl_mean", "proposal_ppl_std",
                             "synthesis_ppl", "ppl_delta"] if c in merged.columns]
    other_cols = [c for c in ["convergence_delta", "bias_score", "mean_historian_distance",
                               "perimeter", "area", "mean_source_embedding_distance",
                               "source_embedding_variance", "abstract_distance_variance",
                               "mean_pairwise_abstract_similarity"] if c in merged.columns]
    all_cols = ppl_cols + other_cols
    if len(all_cols) < 3: return
    corr = merged[all_cols].corr()
    fig, ax = plt.subplots(figsize=(max(12, len(all_cols) * 0.9),
                                    max(10, len(all_cols) * 0.8)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.4, annot_kws={"size": 8}, ax=ax)
    ax.set_title("Perplexity × Convergence/Geometry/Bias Correlation Matrix",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    _save(fig, out / "56_perplexity_full_correlation_heatmap.png",
          "Perplexity full correlation heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – UNVISUALIZED METRICS (plots 57-61)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_57_pairwise_abstract_similarities_individual(df: pd.DataFrame, out: Path) -> None:
    """
    Individual distributions of abstract_similarity_12, _23, _13.
    Previously only the *mean* pairwise similarity was plotted (plot 16).
    This shows each pair separately so asymmetries are visible.
    """
    sim_cols = [("abstract_similarity_12", "Hist 1 ↔ Hist 2"),
                ("abstract_similarity_23", "Hist 2 ↔ Hist 3"),
                ("abstract_similarity_13", "Hist 1 ↔ Hist 3")]
    sim_cols = [(c, l) for c, l in sim_cols if c in df.columns]
    if not sim_cols: return

    fig, axes = plt.subplots(2, len(sim_cols), figsize=(6 * len(sim_cols), 10))
    fig.suptitle("Individual Pairwise Abstract Cosine Similarities", fontsize=14, fontweight="bold")
    if len(sim_cols) == 1:
        axes = axes[:, np.newaxis]

    for j, (col, lbl) in enumerate(sim_cols):
        # Top row: histogram split by delta half
        ax = axes[0, j]
        med57 = df["convergence_delta"].median() if "convergence_delta" in df.columns else 0
        lo57  = df[df["convergence_delta"] <= med57][col].dropna()
        hi57  = df[df["convergence_delta"] >  med57][col].dropna()
        ax.hist(lo57, bins=20, alpha=0.6, color=BIAS_COLOR,    label="Low Δ",  edgecolor="white")
        ax.hist(hi57, bins=20, alpha=0.6, color=NEUTRAL_COLOR, label="High Δ", edgecolor="white")
        ax.set_title(f"{lbl}\n(by delta median split)", fontweight="bold")
        ax.set_xlabel("Cosine Similarity"); ax.set_ylabel("Count")
        ax.legend(fontsize=8)

        # Bottom row: vs convergence_delta scatter
        ax2 = axes[1, j]
        if "convergence_delta" in df.columns:
            sub = df[[col, "convergence_delta"]].dropna()
            colors_pts = _delta_color(df.loc[sub.index, "convergence_delta"].fillna(0))
            ax2.scatter(sub[col], sub["convergence_delta"], c=colors_pts,
                        s=55, alpha=0.75, edgecolors="white", lw=0.3)
            if len(sub) > 2:
                m, b, r, p, _ = stats.linregress(sub[col], sub["convergence_delta"])
                xs = np.linspace(sub[col].min(), sub[col].max(), 100)
                ax2.plot(xs, m * xs + b, "navy", ls="--", lw=1.5, label=f"r={r:.3f}, p={p:.3f}")
                ax2.legend(fontsize=8)
            ax2.axhline(0, color="gray", ls=":", lw=1)
            ax2.set_xlabel("Cosine Similarity"); ax2.set_ylabel("Convergence Delta")
            ax2.set_title(f"{lbl} vs Δ", fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "57_pairwise_abstract_similarities_individual.png",
          "Individual pairwise abstract similarities")


def plot_58_proposal_ppl_std_as_disagreement(ppl_df: pd.DataFrame, df_merged: pd.DataFrame,
                                             out: Path) -> None:
    """
    proposal_ppl_std as a proxy for how much historians disagreed linguistically.
    High std = one historian wrote much more complex/simple text than the others.
    Tests whether linguistic disagreement predicts convergence or bias.
    """
    if ppl_df is None or ppl_df.empty or df_merged.empty: return
    merged = df_merged.merge(ppl_df, on="triad_id", how="inner")
    if "proposal_ppl_std" not in merged.columns: return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Hypothesis PPL Std: Linguistic Disagreement Among Historians",
                 fontsize=14, fontweight="bold")

    # Panel A: distribution
    axes[0].hist(merged["proposal_ppl_std"].dropna(), bins=25,
                 color=PALETTE[1], edgecolor="white", alpha=0.85)
    axes[0].axvline(merged["proposal_ppl_std"].mean(), color="red", ls="--",
                    label=f"μ={merged['proposal_ppl_std'].mean():.1f}")
    axes[0].set_xlabel("Hypothesis PPL Std"); axes[0].set_ylabel("Count")
    axes[0].set_title("A. Distribution of Linguistic Disagreement", fontweight="bold")
    axes[0].legend()

    # Panel B: vs convergence_delta
    if "convergence_delta" in merged.columns:
        sub = merged[["proposal_ppl_std", "convergence_delta"]].dropna()
        colors_pts = _delta_color(merged.loc[sub.index, "convergence_delta"].fillna(0))
        axes[1].scatter(sub["proposal_ppl_std"], sub["convergence_delta"],
                        c=colors_pts, s=55, alpha=0.75, edgecolors="white", lw=0.3)
        if len(sub) > 2:
            m, b, r, p, _ = stats.linregress(sub["proposal_ppl_std"], sub["convergence_delta"])
            xs = np.linspace(sub["proposal_ppl_std"].min(), sub["proposal_ppl_std"].max(), 100)
            axes[1].plot(xs, m * xs + b, "navy", ls="--", lw=1.5, label=f"r={r:.3f}, p={p:.3f}")
            axes[1].legend(fontsize=8)
        axes[1].axhline(0, color="gray", ls=":", lw=1)
        axes[1].set_xlabel("Hypothesis PPL Std"); axes[1].set_ylabel("Convergence Delta")
        axes[1].set_title("B. Linguistic Disagreement vs Convergence", fontweight="bold")

    # Panel C: vs bias_score
    if "bias_score" in merged.columns:
        sub = merged[["proposal_ppl_std", "bias_score"]].dropna()
        axes[2].scatter(sub["proposal_ppl_std"], sub["bias_score"],
                        s=55, alpha=0.75, color=BIAS_COLOR, edgecolors="white", lw=0.3)
        if len(sub) > 2:
            m, b, r, p, _ = stats.linregress(sub["proposal_ppl_std"], sub["bias_score"])
            xs = np.linspace(sub["proposal_ppl_std"].min(), sub["proposal_ppl_std"].max(), 100)
            axes[2].plot(xs, m * xs + b, "navy", ls="--", lw=1.5, label=f"r={r:.3f}, p={p:.3f}")
            axes[2].legend(fontsize=8)
        axes[2].set_xlabel("Hypothesis PPL Std"); axes[2].set_ylabel("Bias Score")
        axes[2].set_title("C. Linguistic Disagreement vs Synthesis Bias", fontweight="bold")

    plt.tight_layout()
    _save(fig, out / "58_hypothesis_ppl_std_disagreement.png",
          "Hypothesis PPL std as linguistic disagreement")


def plot_59_source_embedding_variance_distribution(df: pd.DataFrame, out: Path) -> None:
    """
    source_embedding_variance standalone — previously only used in correlations.
    Shows distribution, vs convergence_delta, and vs mean_source_embedding_distance.
    """
    if "source_embedding_variance" not in df.columns: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Source Embedding Variance (Within-Triad Source Spread)",
                 fontsize=14, fontweight="bold")

    data = df["source_embedding_variance"].dropna()
    axes[0].hist(data, bins=25, color=NEUTRAL_COLOR, edgecolor="white", alpha=0.85)
    axes[0].axvline(data.mean(),   color="red",   ls="--", lw=1.5, label=f"μ={data.mean():.4f}")
    axes[0].axvline(data.median(), color="green", ls=":",  lw=1.5, label=f"med={data.median():.4f}")
    axes[0].set_xlabel("Source Embedding Variance"); axes[0].set_ylabel("Count")
    axes[0].set_title("A. Distribution", fontweight="bold"); axes[0].legend()

    if "convergence_delta" in df.columns:
        sub = df[["source_embedding_variance", "convergence_delta"]].dropna()
        colors_pts = _delta_color(df.loc[sub.index, "convergence_delta"].fillna(0))
        axes[1].scatter(sub["source_embedding_variance"], sub["convergence_delta"],
                        c=colors_pts, s=55, alpha=0.75, edgecolors="white", lw=0.3)
        if len(sub) > 2:
            m, b, r, p, _ = stats.linregress(sub["source_embedding_variance"],
                                              sub["convergence_delta"])
            xs = np.linspace(sub["source_embedding_variance"].min(),
                             sub["source_embedding_variance"].max(), 100)
            axes[1].plot(xs, m * xs + b, "navy", ls="--", lw=1.5, label=f"r={r:.3f}, p={p:.3f}")
            axes[1].legend(fontsize=8)
        axes[1].axhline(0, color="gray", ls=":", lw=1)
        axes[1].set_xlabel("Source Embedding Variance"); axes[1].set_ylabel("Convergence Delta")
        axes[1].set_title("B. Source Variance vs Convergence", fontweight="bold")

    if "mean_source_embedding_distance" in df.columns:
        sub = df[["source_embedding_variance", "mean_source_embedding_distance"]].dropna()
        axes[2].scatter(sub["mean_source_embedding_distance"], sub["source_embedding_variance"],
                        s=55, alpha=0.75, color=ACCENT_COLOR, edgecolors="white", lw=0.3)
        if len(sub) > 2:
            m, b, r, p, _ = stats.linregress(sub["mean_source_embedding_distance"],
                                              sub["source_embedding_variance"])
            xs = np.linspace(sub["mean_source_embedding_distance"].min(),
                             sub["mean_source_embedding_distance"].max(), 100)
            axes[2].plot(xs, m * xs + b, "navy", ls="--", lw=1.5, label=f"r={r:.3f}, p={p:.3f}")
            axes[2].legend(fontsize=8)
        axes[2].set_xlabel("Mean Source Embedding Distance")
        axes[2].set_ylabel("Source Embedding Variance")
        axes[2].set_title("C. Mean Distance vs Variance\n(spread vs average)", fontweight="bold")

    plt.tight_layout()
    _save(fig, out / "59_source_embedding_variance.png", "Source embedding variance")


def plot_60_ppl_delta_vs_bias(ppl_df: pd.DataFrame, df_merged: pd.DataFrame, out: Path) -> None:
    """
    ppl_delta (synthesis simpler/more complex than proposals) vs bias_score.
    Does a biased synthesis also become linguistically simpler?
    Two-panel: scatter + hexbin density.
    """
    if ppl_df is None or ppl_df.empty or df_merged.empty: return
    merged = df_merged.merge(ppl_df, on="triad_id", how="inner")
    need = ["ppl_delta", "bias_score"]
    if not all(c in merged.columns for c in need): return
    sub = merged[need].dropna()
    if len(sub) < 4: return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PPL Delta vs Bias Score: Linguistic Simplification ↔ Historian Dominance",
                 fontsize=13, fontweight="bold")

    colors_pts = _delta_color(merged.loc[sub.index, "convergence_delta"].fillna(0))
    axes[0].scatter(sub["bias_score"], sub["ppl_delta"], c=colors_pts,
                    s=60, alpha=0.8, edgecolors="white", lw=0.3)
    if len(sub) > 2:
        m, b, r, p, _ = stats.linregress(sub["bias_score"], sub["ppl_delta"])
        xs = np.linspace(sub["bias_score"].min(), sub["bias_score"].max(), 100)
        axes[0].plot(xs, m * xs + b, "navy", ls="--", lw=2, label=f"r={r:.3f}, p={p:.3f}")
        axes[0].legend(fontsize=9)
    axes[0].axhline(0, color="black", ls=":", lw=1.2, alpha=0.6)
    axes[0].set_xlabel("Bias Score"); axes[0].set_ylabel("PPL Delta (synthesis − hypotheses)")
    axes[0].set_title("Scatter: Dominance vs Linguistic Change", fontweight="bold")
    axes[0].legend()

    # Hexbin density
    hb = axes[1].hexbin(sub["bias_score"], sub["ppl_delta"],
                        gridsize=20, cmap="YlOrRd", mincnt=1)
    plt.colorbar(hb, ax=axes[1], label="Count")
    axes[1].axhline(0, color="black", ls=":", lw=1.2, alpha=0.6)
    axes[1].set_xlabel("Bias Score"); axes[1].set_ylabel("PPL Delta")
    axes[1].set_title("Density: Dominance vs Linguistic Change", fontweight="bold")

    plt.tight_layout()
    _save(fig, out / "60_ppl_delta_vs_bias.png", "PPL delta vs bias")


def plot_61_ablation_r2_comparison(ablation_path: Optional[Path], out: Path) -> None:
    """
    Compare baseline / extended / full model R² from ablation_study.json.
    Previously computed but never visualized.
    """
    if ablation_path is None or not ablation_path.exists(): return
    with open(ablation_path) as f:
        abl = json.load(f)

    model_keys = ["baseline_model", "extended_model", "full_model"]
    labels      = ["Baseline\n(geometry only)",
                   "Extended\n(+ abstract + bias)",
                   "Full\n(+ source embeddings)"]
    r2_vals     = []
    cv_means    = []
    cv_stds     = []
    valid_labels = []

    for key, lbl in zip(model_keys, labels):
        if key in abl:
            r2_vals.append(abl[key]["r2"])
            cv_means.append(abl[key].get("cv_r2_mean", np.nan))
            cv_stds.append(abl[key].get("cv_r2_std", np.nan))
            valid_labels.append(lbl)

    if not r2_vals: return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Ablation Study: Model R² Comparison", fontsize=14, fontweight="bold")

    x = np.arange(len(valid_labels))
    colors = PALETTE[:len(valid_labels)]

    # In-sample R²
    bars = axes[0].bar(x, r2_vals, color=colors, edgecolor="white", alpha=0.85, width=0.5)
    for bar, val in zip(bars, r2_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005, f"{val:.3f}",
                     ha="center", fontsize=10, fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels(valid_labels, fontsize=10)
    axes[0].set_ylabel("In-sample R²"); axes[0].set_ylim(0, max(r2_vals) * 1.3)
    axes[0].set_title("In-Sample R² (Convergence Delta)", fontweight="bold")
    axes[0].axhline(0, color="gray", ls=":", lw=1)

    # Cross-validated R² with error bars
    cv_means_clean = [v if not np.isnan(v) else 0.0 for v in cv_means]
    cv_stds_clean  = [v if not np.isnan(v) else 0.0 for v in cv_stds]
    axes[1].bar(x, cv_means_clean, yerr=cv_stds_clean, color=colors,
                edgecolor="white", alpha=0.85, capsize=6, width=0.5)
    for xi, val in zip(x, cv_means_clean):
        axes[1].text(xi, val + max(cv_stds_clean) * 1.1 + 0.003,
                     f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(valid_labels, fontsize=10)
    axes[1].set_ylabel("Cross-validated R² ± std")
    axes[1].set_title("CV R² ± 1 SD (5-fold)", fontweight="bold")
    axes[1].axhline(0, color="gray", ls=":", lw=1)

    # Annotate ΔR² improvements if available
    if "extended_improvement" in abl:
        dr2 = abl["extended_improvement"]["delta_r2"]
        axes[0].annotate(f"+{dr2:.3f}", xy=(1, r2_vals[1]), xytext=(1.3, r2_vals[1] * 1.05),
                         fontsize=9, color="darkgreen", arrowprops=dict(arrowstyle="->", color="darkgreen"))
    if "source_improvement" in abl and len(r2_vals) > 2:
        dr2 = abl["source_improvement"]["delta_r2"]
        axes[0].annotate(f"+{dr2:.3f}", xy=(2, r2_vals[2]), xytext=(2.3, r2_vals[2] * 1.05),
                         fontsize=9, color="darkblue", arrowprops=dict(arrowstyle="->", color="darkblue"))

    plt.tight_layout()
    _save(fig, out / "61_ablation_r2_comparison.png", "Ablation R² comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

import logging
logger = logging.getLogger(__name__)


def generate_all(data_dir: Path, out_dir: Path, skip_perplexity: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(" LOADING DATA")
    print(f"{'='*60}")
    tables = _load_csvs(data_dir)
    df_merged = _merged(tables)

    triads_df = tables.get("triads",           pd.DataFrame())
    props_df  = tables.get("proposals",        pd.DataFrame())
    synth_df  = tables.get("synthesis",        pd.DataFrame())
    conv_df   = tables.get("convergence_results", pd.DataFrame())
    fi_df     = tables.get("feature_importance", None)

    inference  = None
    model_json = None
    ablation_path = data_dir / "ablation_study.json"

    inf_path = data_dir / "inference_results.json"
    if inf_path.exists():
        with open(inf_path) as f:
            inference = json.load(f)

    model_path = data_dir / "prediction_model.json"
    if model_path.exists():
        with open(model_path) as f:
            model_json = json.load(f)

    n_loaded = sum(1 for v in tables.values() if not v.empty)
    print(f"  Loaded {n_loaded} CSV tables, {len(df_merged)} merged rows")
    if inference:          print(f"  Loaded inference_results.json")
    if model_json:         print(f"  Loaded prediction_model.json")
    if ablation_path.exists(): print(f"  Loaded ablation_study.json")

    # Perplexity (lazy — loads/computes on demand)
    ppl_df = None
    if not skip_perplexity:
        print("\n  Checking perplexity data…")
        ppl_df = _load_or_compute_ppl(data_dir, props_df, synth_df)
        if ppl_df is not None:
            print(f"  Perplexity data ready: {len(ppl_df)} triads")
        else:
            print("  Perplexity data unavailable — §7 plots will be skipped")
    else:
        print("  --skip-perplexity set — §7/§8 perplexity plots skipped")

    total_plots = 61
    print(f"\n{'='*60}")
    print(f" GENERATING PLOTS ({total_plots} total)")
    print(f"{'='*60}\n")

    # ── Section 1: Geometry ───────────────────────────────────
    print("── SECTION 1: Geometry ─────────────────────────────────")
    plot_01_geometry_distributions(triads_df, out_dir)
    plot_02_perimeter_vs_area(triads_df, out_dir)
    plot_03_side_lengths(triads_df, out_dir)
    plot_04_angle_analysis(triads_df, out_dir)
    plot_05_geometry_correlation_heatmap(triads_df, out_dir)
    plot_06_triangle_regularity(triads_df, out_dir)
    plot_07_side_balance(triads_df, out_dir)
    plot_08_geometry_pairplot(triads_df, out_dir)

    # ── Section 2: Convergence ────────────────────────────────
    print("\n── SECTION 2: Convergence ──────────────────────────────")
    plot_09_convergence_outcome_bar(df_merged, out_dir)
    plot_10_delta_distribution(df_merged, out_dir)
    plot_11_delta_by_geometry(df_merged, out_dir)
    plot_12_convergence_rate_by_area_quartile(df_merged, out_dir)
    plot_13_abstract_vs_final_distance(df_merged, out_dir)
    plot_14_historian_distances_to_centroid(df_merged, out_dir)
    plot_15_abstract_distances_to_centroid(df_merged, out_dir)
    plot_16_pairwise_abstract_similarity(df_merged, out_dir)
    plot_17_delta_vs_abstract_variance(df_merged, out_dir)

    # ── Section 3: Bias ───────────────────────────────────────
    print("\n── SECTION 3: Bias ─────────────────────────────────────")
    plot_18_bias_score_distribution(df_merged, out_dir)
    plot_19_dominant_historian_frequency(df_merged, out_dir)
    plot_20_bias_weights_ternary_proxy(df_merged, out_dir)
    plot_21_bias_score_vs_delta(df_merged, out_dir)
    plot_22_bias_by_geometry(df_merged, out_dir)
    plot_23_bias_weight_distributions(df_merged, out_dir)

    # ── Section 4: Prediction ─────────────────────────────────
    print("\n── SECTION 4: Prediction Model ─────────────────────────")
    plot_24_feature_importance(fi_df, out_dir)
    plot_25_actual_vs_predicted(df_merged, model_json, out_dir)
    plot_26_coefficient_magnitude_ranked(fi_df, out_dir)
    plot_27_inference_correlations(inference, out_dir)
    plot_28_ols_area_vs_delta(inference, df_merged, out_dir)
    plot_29_prediction_summary_dashboard(df_merged, inference, fi_df, out_dir)

    # ── Section 5: Experiment overview ───────────────────────
    print("\n── SECTION 5: Experiment Overview ──────────────────────")
    plot_30_experiment_overview_dashboard(df_merged, out_dir)
    plot_31_historian_participation(props_df, out_dir)
    plot_32_proposal_abstract_length(props_df, out_dir)
    plot_33_synthesis_abstract_length(synth_df, props_df, out_dir)
    plot_34_source_usage(props_df, out_dir)
    plot_35_convergence_by_historian_pair(df_merged, out_dir)
    plot_36_triad_geometry_ranked(df_merged, out_dir)
    plot_37_comprehensive_correlation_matrix(df_merged, out_dir)

    # ── Section 6: Advanced ───────────────────────────────────
    print("\n── SECTION 6: Advanced ─────────────────────────────────")
    plot_38_pca_embedding_of_triads(df_merged, out_dir)
    plot_39_delta_vs_mean_historian_distance(df_merged, out_dir)
    plot_40_diversity_convergence_quadrant(df_merged, out_dir)
    plot_41_cumulative_delta_distribution(df_merged, out_dir)
    plot_42_perimeter_area_3d(df_merged, out_dir)
    plot_43_bias_vs_historian_distance(df_merged, out_dir)
    plot_44_dominant_position_vs_delta(df_merged, out_dir)
    plot_45_distance_variance_decomposition(df_merged, out_dir)
    plot_46_synthesis_question_length(synth_df, out_dir)
    plot_47_convergence_delta_violin(df_merged, out_dir)
    plot_48_perimeter_quartile_bias(df_merged, out_dir)
    plot_49_all_distances_scatter_matrix(df_merged, out_dir)
    plot_50_final_summary_heatmap(df_merged, out_dir)

    # ── Section 7: Perplexity ─────────────────────────────────
    print("\n── SECTION 7: Perplexity ───────────────────────────────")
    if ppl_df is not None:
        plot_51_perplexity_distributions(ppl_df, out_dir)
        plot_52_proposal_vs_synthesis_perplexity(ppl_df, out_dir)
        plot_53_perplexity_vs_convergence(ppl_df, df_merged, out_dir)
        plot_54_perplexity_vs_bias(ppl_df, df_merged, out_dir)
        plot_55_perplexity_vs_source_diversity(ppl_df, df_merged, out_dir)
        plot_56_perplexity_full_correlation_heatmap(ppl_df, df_merged, out_dir)
    else:
        print("  (skipped — no perplexity data)")

    # ── Section 8: Unvisualized metrics ──────────────────────
    print("\n── SECTION 8: Unvisualized Metrics ─────────────────────")
    plot_57_pairwise_abstract_similarities_individual(df_merged, out_dir)
    plot_58_proposal_ppl_std_as_disagreement(ppl_df, df_merged, out_dir)
    plot_59_source_embedding_variance_distribution(df_merged, out_dir)
    plot_60_ppl_delta_vs_bias(ppl_df, df_merged, out_dir)
    plot_61_ablation_r2_comparison(ablation_path if ablation_path.exists() else None, out_dir)

    # ── Section 9: Conceptual Space Figure ───────────────────
    print("\n── SECTION 9: Conceptual Space ─────────────────────────")
    plot_62_conceptual_space_figure(df_merged, out_dir)

    # ── Section 10: Ablation Deep-Dive ───────────────────────
    print("\n── SECTION 10: Ablation Deep-Dive ──────────────────────")
    ab = _load_ablation(ablation_path if ablation_path.exists() else None)
    if ab:
        plot_63_ablation_r2_bars(ab, out_dir)
        plot_64_ablation_error_metrics(ab, out_dir)
        plot_65_ablation_feature_blocks(ab, out_dir)
        plot_66_ablation_cv_violin(ab, df_merged, out_dir)
        plot_67_ablation_marginal_gain(ab, out_dir)
        plot_68_ablation_overfitting_gap(ab, out_dir)
        plot_69_ablation_feature_group_radar(ab, out_dir)
        plot_70_ablation_summary_table(ab, out_dir)
    else:
        print("  (skipped — no ablation_study.json)")

    n_saved = len(list(out_dir.glob("*.png")))
    print(f"\n{'='*60}")
    print(f"  ✓ DONE — {n_saved} plots saved to: {out_dir}")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – CONCEPTUAL SPACE FIGURE (plot 62)
# Inspired by Park et al. / Giorla et al. framing
# ═══════════════════════════════════════════════════════════════════════════════

def plot_62_conceptual_space_figure(df: pd.DataFrame, out: Path) -> None:
    """
    6-panel figure replicating the Park et al. conceptual-space framing
    using actual experiment data.

    Panels:
      a  – Schematic: neural network → embedding space construction
      b  – Schematic: historian positions mapped in conceptual space
      c  – Data: radial movement (distance to centroid) before/after synthesis
      d  – Schematic: perspective diversity vs background diversity definition
      e  – Schematic: five canonical team diversity configurations
      f  – Data: distributions of perspective & background diversity + correlation
    """
    needed = ["distance_hist1_to_centroid", "distance_hist2_to_centroid",
              "distance_hist3_to_centroid", "distance_final_to_centroid",
              "mean_historian_distance", "convergence_delta"]
    if not all(c in df.columns for c in needed):
        print("  (skipped plot_62 — missing convergence columns)")
        return

    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor("white")

    # Grid: 2 rows × 3 cols
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38,
                          left=0.06, right=0.97, top=0.93, bottom=0.07)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    label_kw = dict(fontsize=13, fontweight="bold", transform=fig.transFigure,
                    va="top")
    for ax, letter, x in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f],
                               "abcdef",
                               [0.03, 0.36, 0.69, 0.03, 0.36, 0.69]):
        y = 0.97 if letter in "abc" else 0.50
        fig.text(x, y, letter, **label_kw)

    C1, C2, C3 = "#3498db", "#e74c3c", "#2ecc71"   # historian colors
    CS = "#9b59b6"                                   # synthesis color
    GREY = "#95a5a6"

    # ── Panel a: embedding space schematic ───────────────────────────────────
    ax_a.set_xlim(0, 10); ax_a.set_ylim(0, 10); ax_a.axis("off")
    ax_a.set_title("Conceptual Space Construction", fontsize=10, fontweight="bold", pad=6)

    # Input layer
    for i, (y, lbl) in enumerate([(7.5, "Text"), (5.0, "Citations"), (2.5, "Topics")]):
        ax_a.add_patch(mpatches.Circle((1.2, y), 0.55, color=NEUTRAL_COLOR, zorder=3))
        ax_a.text(1.2, y, lbl, ha="center", va="center", fontsize=6.5,
                  color="white", fontweight="bold")

    # Hidden layer
    for y in [7.0, 5.5, 4.0, 2.5]:
        ax_a.add_patch(mpatches.Circle((4.5, y), 0.4, color=GREY, zorder=3, alpha=0.7))

    # Output: embedding
    ax_a.add_patch(mpatches.FancyBboxPatch((7.2, 4.0), 1.8, 2.0,
                   boxstyle="round,pad=0.15", color=ACCENT_COLOR, zorder=3, alpha=0.85))
    ax_a.text(8.1, 5.0, "768-d\nEmbedding", ha="center", va="center",
              fontsize=6.5, color="white", fontweight="bold")

    # Connections (sparse)
    for iy in [7.5, 5.0, 2.5]:
        for hy in [7.0, 5.5, 4.0, 2.5]:
            ax_a.annotate("", xy=(4.1, hy), xytext=(1.75, iy),
                          arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5, alpha=0.4))
    for hy in [7.0, 5.5, 4.0, 2.5]:
        ax_a.annotate("", xy=(7.2, 5.0), xytext=(4.9, hy),
                      arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5, alpha=0.4))

    ax_a.text(1.2, 0.8, "Inputs", ha="center", fontsize=7, color=GREY)
    ax_a.text(4.5, 0.8, "Encoder", ha="center", fontsize=7, color=GREY)
    ax_a.text(8.1, 0.8, "Output", ha="center", fontsize=7, color=GREY)

    # ── Panel b: historian positions in conceptual space ─────────────────────
    ax_b.set_title("Historian Positions in Conceptual Space", fontsize=10,
                   fontweight="bold", pad=6)
    ax_b.set_xlabel("Dimension 1 (schematic)", fontsize=8)
    ax_b.set_ylabel("Dimension 2 (schematic)", fontsize=8)
    ax_b.set_xlim(-3.5, 3.5); ax_b.set_ylim(-3.5, 3.5)

    # Background: faint density cloud
    rng = np.random.default_rng(42)
    bg_x = rng.normal(0, 1.2, 300)
    bg_y = rng.normal(0, 1.2, 300)
    ax_b.scatter(bg_x, bg_y, s=8, color=GREY, alpha=0.15, zorder=1)

    # Three historian positions (schematic triangle)
    h_pos = np.array([[-1.8, -1.2], [1.8, -1.2], [0.0, 1.8]])
    centroid = h_pos.mean(axis=0)
    synth_pos = centroid * 0.55   # synthesis moves toward centroid

    tri = mpatches.Polygon(h_pos, fill=False, edgecolor=GREY, ls="--", lw=1.2, zorder=2)
    ax_b.add_patch(tri)

    for (x, y), c, lbl in zip(h_pos, [C1, C2, C3], ["H₁", "H₂", "H₃"]):
        ax_b.scatter(x, y, s=120, color=c, zorder=5, edgecolors="white", lw=1.5)
        ax_b.text(x, y + 0.28, lbl, ha="center", fontsize=8, color=c, fontweight="bold")
        ax_b.annotate("", xy=synth_pos, xytext=(x, y),
                      arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2,
                                      mutation_scale=10, alpha=0.6))

    ax_b.scatter(*centroid, s=80, color=GREY, marker="+", zorder=4, lw=2)
    ax_b.scatter(*synth_pos, s=160, color=CS, zorder=6,
                 edgecolors="white", lw=1.5, marker="*")
    ax_b.text(synth_pos[0], synth_pos[1] - 0.35, "Synthesis", ha="center",
              fontsize=7.5, color=CS, fontweight="bold")
    ax_b.text(centroid[0] + 0.15, centroid[1] + 0.15, "Centroid",
              fontsize=6.5, color=GREY)
    ax_b.tick_params(labelsize=7)

    # ── Panel c: radial movement — data-driven ────────────────────────────────
    ax_c.set_title("Idea Movement: Distance to Centroid\nBefore vs After Synthesis",
                   fontsize=10, fontweight="bold", pad=6)

    d_before = df[["distance_hist1_to_centroid",
                   "distance_hist2_to_centroid",
                   "distance_hist3_to_centroid"]].mean(axis=1)
    d_after  = df["distance_final_to_centroid"]
    delta    = df["convergence_delta"]

    # Scatter: mean historian distance → final synthesis distance
    norm_c = plt.Normalize(delta.min(), delta.max())
    sc = ax_c.scatter(d_before, d_after, c=delta, cmap="RdYlGn",
                      norm=norm_c, s=55, alpha=0.75,
                      edgecolors="white", lw=0.3, zorder=3)
    lims = [min(d_before.min(), d_after.min()) * 0.9,
            max(d_before.max(), d_after.max()) * 1.05]
    ax_c.plot(lims, lims, "k--", lw=1.2, alpha=0.4, zorder=2,
              label="No movement")
    ax_c.fill_between(lims, lims, [lims[1]]*2, alpha=0.04, color=C2)  # above = diverged
    ax_c.fill_between(lims, [0]*2, lims,        alpha=0.04, color=C3)  # below = converged
    ax_c.set_xlabel("Mean Historian Distance to Centroid\n(before synthesis)", fontsize=8)
    ax_c.set_ylabel("Synthesis Distance to Centroid\n(after synthesis)", fontsize=8)
    ax_c.legend(fontsize=7)
    ax_c.tick_params(labelsize=7)
    plt.colorbar(sc, ax=ax_c, label="Convergence Δ", shrink=0.75)
    n = len(df)
    n_below = (d_after < d_before).sum()
    ax_c.text(0.03, 0.96,
              f"n={n}  |  {n_below/n:.0%} moved closer to centroid",
              transform=ax_c.transAxes, fontsize=7, va="top", color="black",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # ── Panel d: perspective vs background diversity definition ───────────────
    ax_d.set_xlim(0, 10); ax_d.set_ylim(0, 10); ax_d.axis("off")
    ax_d.set_title("Perspective vs Background Diversity", fontsize=10,
                   fontweight="bold", pad=6)

    # Two clusters in conceptual space
    rng2 = np.random.default_rng(7)
    for cx, cy, col, lbl in [(3.0, 6.5, C1, "a"), (6.5, 3.5, C2, "b"), (3.5, 3.0, C3, "c")]:
        ax_d.scatter(cx, cy, s=180, color=col, zorder=4, edgecolors="white", lw=1.5)
        ax_d.text(cx + 0.4, cy + 0.2, lbl, fontsize=11, fontweight="bold", color=col)

    # Perspective diversity arrow (spread in embedding space)
    ax_d.annotate("", xy=(6.5, 3.5), xytext=(3.0, 6.5),
                  arrowprops=dict(arrowstyle="<->", color="#2c3e50", lw=1.8))
    ax_d.text(4.2, 5.7, "Perspective\nDiversity (σ_embed)",
              fontsize=7.5, color="#2c3e50", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    # Background diversity brace (same field vs different)
    ax_d.annotate("", xy=(3.5, 3.0), xytext=(6.5, 3.5),
                  arrowprops=dict(arrowstyle="<->", color=BIAS_COLOR, lw=1.8))
    ax_d.text(4.5, 2.2, "Background Diversity\n(institutional distance)",
              fontsize=7.5, color=BIAS_COLOR, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    ax_d.text(5.0, 9.2, "Conceptual Space", ha="center", fontsize=8,
              color=GREY, style="italic")
    ax_d.add_patch(mpatches.FancyBboxPatch((0.3, 0.5), 9.4, 9.0,
                   boxstyle="round,pad=0.2", fill=False,
                   edgecolor=GREY, lw=1.0, ls="--"))

    # ── Panel e: diversity configurations ────────────────────────────────────
    ax_e.set_xlim(0, 10); ax_e.set_ylim(0, 10); ax_e.axis("off")
    ax_e.set_title("Team Diversity Configurations", fontsize=10,
                   fontweight="bold", pad=6)

    configs = [
        # (label, positions, description)
        ("Clustered",    [(1.0, 8.5), (1.5, 7.8), (1.2, 8.0)],   "Low P, Low B"),
        ("Dispersed",    [(3.2, 9.0), (4.8, 7.5), (4.0, 8.8)],   "High P, Low B"),
        ("Mixed",        [(6.5, 9.0), (7.8, 8.0), (7.0, 7.5)],   "High P, High B"),
        ("Asymmetric",   [(1.2, 5.5), (1.8, 4.5), (3.0, 5.0)],   "Low P, High B"),
        ("Our triads",   None,                                      "Variable P & B"),
    ]

    for i, (lbl, pts, desc) in enumerate(configs):
        col = PALETTE[i % len(PALETTE)]
        if pts is not None:
            xs, ys = zip(*pts)
            ax_e.scatter(xs, ys, s=90, color=col, zorder=4,
                         edgecolors="white", lw=1.2)
            tri_e = mpatches.Polygon(pts, fill=True, facecolor=col,
                                edgecolor=col, alpha=0.12, zorder=2)
            ax_e.add_patch(tri_e)
        else:
            # "Our triads" — draw a small multi-triangle cloud
            for dx, dy in [(-0.3, 0), (0.3, 0.2), (0, -0.3)]:
                sub = [(2.0+dx+j*0.5, 4.2+dy+k*0.4)
                       for j, k in [(0,0),(1,0),(0.5,0.7)]]
                ax_e.add_patch(mpatches.Polygon(sub, fill=True, facecolor=col,
                                           edgecolor=col, alpha=0.15, zorder=2))
        ax_e.text(2.0 if pts is None else np.mean([p[0] for p in pts]),
                  (3.5 if pts is None else min(p[1] for p in pts)) - 0.4,
                  f"{lbl}\n{desc}", ha="center", fontsize=6.5,
                  color=col, fontweight="bold")

    ax_e.text(5.0, 9.8, "← Perspective Diversity →", ha="center",
              fontsize=7.5, color="#2c3e50")
    ax_e.text(0.2, 5.0, "← Background\nDiversity →", ha="center",
              fontsize=7.5, color=BIAS_COLOR, rotation=90)

    # ── Panel f: data-driven distributions + correlation ─────────────────────
    ax_f.set_title("Perspective & Background Diversity\nDistributions",
                   fontsize=10, fontweight="bold", pad=6)

    # Perspective diversity = mean pairwise historian distance (embedding spread)
    persp = df["mean_historian_distance"] if "mean_historian_distance" in df.columns \
            else (df["side_1"] + df["side_2"] + df["side_3"]) / 3
    # Background diversity = variance in distances to centroid (how asymmetric the triangle)
    bg_div = df[["distance_hist1_to_centroid",
                 "distance_hist2_to_centroid",
                 "distance_hist3_to_centroid"]].std(axis=1)

    persp_n  = (persp  - persp.mean())  / (persp.std()  + 1e-9)
    bg_n     = (bg_div - bg_div.mean()) / (bg_div.std() + 1e-9)

    ax_f.hist(persp_n, bins=25, alpha=0.6, color=NEUTRAL_COLOR,
              edgecolor="white", label="Perspective diversity (norm.)")
    ax_f.hist(bg_n,    bins=25, alpha=0.6, color=BIAS_COLOR,
              edgecolor="white", label="Background diversity (norm.)")

    r, p = stats.pearsonr(persp_n.dropna(), bg_n[persp_n.dropna().index])
    ax_f.set_xlabel("Normalised Diversity Score", fontsize=8)
    ax_f.set_ylabel("Count", fontsize=8)
    ax_f.legend(fontsize=7)
    ax_f.tick_params(labelsize=7)
    ax_f.text(0.97, 0.95, f"r = {r:.3f}\np = {p:.3f}",
              transform=ax_f.transAxes, ha="right", va="top", fontsize=8,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.suptitle(
        "Conceptual Space Construction and Dimensions of Subjectivity",
        fontsize=14, fontweight="bold", y=0.98
    )
    _save(fig, out / "62_conceptual_space_figure.png",
          "Conceptual space figure (panels a–f)")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – ABLATION DEEP-DIVE (plots 63-70)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_ablation(ablation_path) -> Optional[dict]:
    if ablation_path is None or not Path(ablation_path).exists():
        return None
    import json
    with open(ablation_path) as f:
        return json.load(f)


def plot_63_ablation_r2_bars(ab: dict, out: Path) -> None:
    """R² and CV-R² side-by-side for all three models."""
    models = ["Baseline", "Extended", "Full"]
    r2     = [ab["baseline_model"]["r2"],   ab["extended_model"]["r2"],   ab["full_model"]["r2"]]
    cv_r2  = [ab["baseline_model"]["cv_r2_mean"], ab["extended_model"]["cv_r2_mean"], ab["full_model"]["cv_r2_mean"]]
    cv_std = [ab["baseline_model"]["cv_r2_std"],  ab["extended_model"]["cv_r2_std"],  ab["full_model"]["cv_r2_std"]]

    x = np.arange(len(models)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - w/2, r2,   w, label="Train R²",   color=NEUTRAL_COLOR, edgecolor="white", alpha=0.9)
    bars2 = ax.bar(x + w/2, cv_r2, w, label="CV R² (mean±std)", color=ACCENT_COLOR,
                   edgecolor="white", alpha=0.9, yerr=cv_std, capsize=5)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
    for bar, err in zip(bars2, cv_std):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + err + 0.015,
                f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title("Train vs Cross-Validated R² by Model", fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(min(min(cv_r2) - max(cv_std) - 0.1, -0.15), max(r2) * 1.25)
    plt.tight_layout()
    _save(fig, out / "63_ablation_r2_bars.png", "Ablation R² bars")


def plot_64_ablation_error_metrics(ab: dict, out: Path) -> None:
    """MAE and RMSE for all three models."""
    models = ["Baseline", "Extended", "Full"]
    mae  = [ab["baseline_model"]["mae"],  ab["extended_model"]["mae"],  ab["full_model"]["mae"]]
    rmse = [ab["baseline_model"]["rmse"], ab["extended_model"]["rmse"], ab["full_model"]["rmse"]]

    x = np.arange(len(models)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - w/2, mae,  w, label="MAE",  color=BIAS_COLOR,    edgecolor="white", alpha=0.9)
    b2 = ax.bar(x + w/2, rmse, w, label="RMSE", color=NEUTRAL_COLOR, edgecolor="white", alpha=0.9)
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                    f"{bar.get_height():.4f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Error (convergence delta units)", fontsize=10)
    ax.set_title("MAE & RMSE by Model — Lower is Better", fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, out / "64_ablation_error_metrics.png", "Ablation error metrics")


def plot_65_ablation_feature_blocks(ab: dict, out: Path) -> None:
    """Stacked view: which feature block drives the R² gain."""
    fig, ax = plt.subplots(figsize=(10, 6))

    r2_base = ab["baseline_model"]["r2"]
    r2_ext  = ab["extended_model"]["r2"]
    r2_full = ab["full_model"]["r2"]
    gain_semantic = r2_ext  - r2_base
    gain_source   = r2_full - r2_ext

    models = ["Baseline\n(geometry)", "Extended\n(+semantic/bias)", "Full\n(+source embed)"]
    base_vals  = [r2_base,  r2_base,  r2_base]
    sem_gains  = [0,         gain_semantic, gain_semantic]
    src_gains  = [0,         0,             gain_source]

    colors = [NEUTRAL_COLOR, ACCENT_COLOR, BIAS_COLOR]
    labels = ["Geometry features", "Semantic + bias features", "Source embedding features"]
    bottom = np.zeros(3)
    for vals, color, lbl in zip([base_vals, sem_gains, src_gains], colors, labels):
        bars = ax.bar(models, vals, bottom=bottom, color=color, edgecolor="white",
                      alpha=0.88, label=lbl, width=0.5)
        for bar, val in zip(bars, vals):
            if val > 0.005:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_y() + bar.get_height()/2,
                        f"+{val:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottom += np.array(vals)

    for i, total in enumerate([r2_base, r2_ext, r2_full]):
        ax.text(i, total + 0.008, f"R²={total:.3f}", ha="center",
                fontsize=10, fontweight="bold")

    ax.set_ylabel("R² (train)", fontsize=11)
    ax.set_title("R² Decomposition by Feature Block", fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, r2_full * 1.35)
    plt.tight_layout()
    _save(fig, out / "65_ablation_feature_blocks.png", "Ablation feature block decomposition")


def plot_66_ablation_cv_violin(ab: dict, df_merged: pd.DataFrame, out: Path) -> None:
    """
    Simulate CV fold variance visually using cv_r2_mean ± cv_r2_std.
    Draws a normal approximation for each model since raw fold scores aren't stored.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    models = ["Baseline", "Extended", "Full"]
    means  = [ab["baseline_model"]["cv_r2_mean"], ab["extended_model"]["cv_r2_mean"], ab["full_model"]["cv_r2_mean"]]
    stds   = [ab["baseline_model"]["cv_r2_std"],  ab["extended_model"]["cv_r2_std"],  ab["full_model"]["cv_r2_std"]]
    colors = [NEUTRAL_COLOR, ACCENT_COLOR, BIAS_COLOR]

    rng = np.random.default_rng(0)
    for i, (m, s, c, lbl) in enumerate(zip(means, stds, colors, models)):
        samples = rng.normal(m, s, 500)
        parts = ax.violinplot(samples, positions=[i], widths=0.6, showmedians=True,
                              showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(c); pc.set_alpha(0.6)
        parts["cmedians"].set_color("white"); parts["cmedians"].set_linewidth(2)
        ax.scatter(i, m, color=c, s=80, zorder=5, edgecolors="white", lw=1.5)
        ax.text(i, m + s + 0.04, f"μ={m:.3f}\nσ={s:.3f}",
                ha="center", fontsize=8, color=c, fontweight="bold")

    ax.axhline(0, color="black", ls="--", lw=1, alpha=0.5, label="R²=0 baseline")
    ax.set_xticks(range(3)); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Cross-Validated R²", fontsize=11)
    ax.set_title("CV R² Distribution by Model\n(simulated from mean±std)", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, out / "66_ablation_cv_violin.png", "Ablation CV R² violin")


def plot_67_ablation_marginal_gain(ab: dict, out: Path) -> None:
    """Marginal R² gain per additional feature."""
    r2s    = [ab["baseline_model"]["r2"], ab["extended_model"]["r2"], ab["full_model"]["r2"]]
    n_feat = [ab["baseline_model"]["n_features"], ab["extended_model"]["n_features"],
              ab["full_model"]["n_features"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: R² vs n_features
    axes[0].plot(n_feat, r2s, "o-", color=NEUTRAL_COLOR, lw=2, ms=10,
                 markeredgecolor="white", markeredgewidth=1.5)
    for x, y, lbl in zip(n_feat, r2s, ["Baseline", "Extended", "Full"]):
        axes[0].text(x, y + 0.01, f"{lbl}\nR²={y:.3f}", ha="center", fontsize=8)
    axes[0].set_xlabel("Number of Features", fontsize=11)
    axes[0].set_ylabel("Train R²", fontsize=11)
    axes[0].set_title("R² vs Feature Count", fontweight="bold")
    axes[0].set_ylim(0, max(r2s) * 1.3)

    # Right: marginal R² gain per extra feature
    transitions = [
        ("Baseline→Extended", n_feat[1]-n_feat[0], r2s[1]-r2s[0]),
        ("Extended→Full",     n_feat[2]-n_feat[1], r2s[2]-r2s[1]),
    ]
    labels  = [t[0] for t in transitions]
    n_added = [t[1] for t in transitions]
    gains   = [t[2] for t in transitions]
    marginal = [g/n for g, n in zip(gains, n_added)]

    bars = axes[1].bar(labels, marginal, color=[ACCENT_COLOR, BIAS_COLOR],
                       edgecolor="white", width=0.45, alpha=0.88)
    for bar, g, n in zip(bars, gains, n_added):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f"ΔR²={g:.3f}\n÷{n} features",
                     ha="center", fontsize=9, fontweight="bold")
    axes[1].set_ylabel("Marginal R² Gain per Feature Added", fontsize=10)
    axes[1].set_title("Marginal Predictive Value per Feature", fontweight="bold")
    axes[1].set_ylim(0, max(marginal) * 1.45)

    plt.tight_layout()
    _save(fig, out / "67_ablation_marginal_gain.png", "Ablation marginal gain")


def plot_68_ablation_overfitting_gap(ab: dict, out: Path) -> None:
    """Train R² vs CV R² gap — overfitting diagnostic."""
    models  = ["Baseline", "Extended", "Full"]
    train   = [ab["baseline_model"]["r2"],           ab["extended_model"]["r2"],           ab["full_model"]["r2"]]
    cv      = [ab["baseline_model"]["cv_r2_mean"],   ab["extended_model"]["cv_r2_mean"],   ab["full_model"]["cv_r2_mean"]]
    cv_std  = [ab["baseline_model"]["cv_r2_std"],    ab["extended_model"]["cv_r2_std"],    ab["full_model"]["cv_r2_std"]]
    gaps    = [t - c for t, c in zip(train, cv)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: train vs CV lines
    x = np.arange(len(models))
    axes[0].plot(x, train, "o-", color=NEUTRAL_COLOR, lw=2, ms=9,
                 label="Train R²", markeredgecolor="white", markeredgewidth=1.5)
    axes[0].plot(x, cv,    "s--", color=BIAS_COLOR,   lw=2, ms=9,
                 label="CV R²",    markeredgecolor="white", markeredgewidth=1.5)
    axes[0].fill_between(x, cv, train, alpha=0.12, color=BIAS_COLOR, label="Overfitting gap")
    for i, (t, c, s) in enumerate(zip(train, cv, cv_std)):
        axes[0].errorbar(i, c, yerr=s, fmt="none", color=BIAS_COLOR, capsize=4, lw=1.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, fontsize=11)
    axes[0].axhline(0, color="black", ls=":", lw=0.8, alpha=0.4)
    axes[0].set_ylabel("R²", fontsize=11)
    axes[0].set_title("Train vs CV R²: Overfitting Diagnostic", fontweight="bold")
    axes[0].legend(fontsize=9)

    # Right: gap magnitude
    bar_colors = [NEUTRAL_COLOR if g < 0.1 else BIAS_COLOR if g < 0.2 else "#e74c3c"
                  for g in gaps]
    bars = axes[1].bar(models, gaps, color=bar_colors, edgecolor="white", width=0.5, alpha=0.88)
    for bar, g in zip(bars, gaps):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"{g:.3f}", ha="center", fontsize=10, fontweight="bold")
    axes[1].axhline(0.1, color=BIAS_COLOR, ls="--", lw=1.2, alpha=0.6,
                    label="Mild overfit threshold (0.1)")
    axes[1].axhline(0.2, color="#e74c3c",  ls="--", lw=1.2, alpha=0.6,
                    label="Severe overfit threshold (0.2)")
    axes[1].set_ylabel("Train R² − CV R²", fontsize=11)
    axes[1].set_title("Overfitting Gap by Model", fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, max(gaps) * 1.4)

    plt.tight_layout()
    _save(fig, out / "68_ablation_overfitting_gap.png", "Ablation overfitting gap")


def plot_69_ablation_feature_group_radar(ab: dict, out: Path) -> None:
    """Radar chart: each model scored on R², CV-R², low-MAE, low-RMSE, feature efficiency."""
    from matplotlib.patches import FancyArrowPatch

    def _norm(vals):
        lo, hi = min(vals), max(vals)
        return [(v - lo) / (hi - lo + 1e-9) for v in vals]

    models  = ["Baseline", "Extended", "Full"]
    r2s     = [ab["baseline_model"]["r2"],           ab["extended_model"]["r2"],           ab["full_model"]["r2"]]
    cv_r2s  = [ab["baseline_model"]["cv_r2_mean"],   ab["extended_model"]["cv_r2_mean"],   ab["full_model"]["cv_r2_mean"]]
    maes    = [ab["baseline_model"]["mae"],           ab["extended_model"]["mae"],           ab["full_model"]["mae"]]
    rmses   = [ab["baseline_model"]["rmse"],          ab["extended_model"]["rmse"],          ab["full_model"]["rmse"]]
    n_feats = [ab["baseline_model"]["n_features"],    ab["extended_model"]["n_features"],    ab["full_model"]["n_features"]]

    # Efficiency = R² per feature (normalised)
    effic = [r/n for r, n in zip(r2s, n_feats)]

    categories = ["Train R²", "CV R²", "Low MAE", "Low RMSE", "Efficiency"]
    data_raw = [r2s, cv_r2s,
                [1-m for m in _norm(maes)],   # invert: lower is better
                [1-r for r in _norm(rmses)],
                _norm(effic)]
    # Normalise each dimension 0-1
    data = [_norm(d) for d in data_raw]

    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = [NEUTRAL_COLOR, ACCENT_COLOR, BIAS_COLOR]

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [data[j][i] for j in range(N)]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", color=color, lw=2, ms=6, label=model)
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_yticklabels([])
    ax.set_title("Model Comparison Radar\n(all dimensions normalised 0–1)",
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    _save(fig, out / "69_ablation_radar.png", "Ablation radar chart")


def plot_70_ablation_summary_table(ab: dict, out: Path) -> None:
    """Publication-style summary table of all ablation metrics."""
    models = ["Baseline", "Extended", "Full"]
    rows = []
    for key, lbl in [("baseline_model","Baseline"), ("extended_model","Extended"), ("full_model","Full")]:
        m = ab[key]
        rows.append([
            lbl,
            m["n_features"],
            f"{m['r2']:.3f}",
            f"{m['cv_r2_mean']:.3f} ± {m['cv_r2_std']:.3f}",
            f"{m['mae']:.4f}",
            f"{m['rmse']:.4f}",
            f"{m['alpha']}"
        ])

    col_labels = ["Model", "# Features", "Train R²", "CV R² (mean±std)", "MAE", "RMSE", "α (ridge)"]

    fig, ax = plt.subplots(figsize=(13, 3))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)

    # Header style
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Row colors
    row_colors = [NEUTRAL_COLOR, ACCENT_COLOR, BIAS_COLOR]
    for i, color in enumerate(row_colors):
        for j in range(len(col_labels)):
            tbl[(i+1, j)].set_facecolor(color + "22")  # hex alpha
            if j == 2:  # highlight train R²
                tbl[(i+1, j)].set_text_props(fontweight="bold")

    # Highlight best CV R²
    cv_vals = [ab["baseline_model"]["cv_r2_mean"],
               ab["extended_model"]["cv_r2_mean"],
               ab["full_model"]["cv_r2_mean"]]
    best_row = np.argmax(cv_vals) + 1
    tbl[(best_row, 3)].set_facecolor("#2ecc7133")
    tbl[(best_row, 3)].set_text_props(fontweight="bold")

    ax.set_title("Ablation Study Summary", fontweight="bold", fontsize=13, pad=12)
    plt.tight_layout()
    _save(fig, out / "70_ablation_summary_table.png", "Ablation summary table")




def main():
    parser = argparse.ArgumentParser(description="Generate all experiment visualizations")
    parser.add_argument("--data-dir", default="data/agent_experiments",
                        help="Directory containing experiment CSVs and JSON files")
    parser.add_argument("--out-dir",  default="data/agent_experiments/figures",
                        help="Output directory for figures")
    parser.add_argument("--skip-perplexity", action="store_true",
                        help="Skip perplexity computation (sections 7-8 ppl plots)")
    args = parser.parse_args()
    generate_all(Path(args.data_dir), Path(args.out_dir),
                 skip_perplexity=args.skip_perplexity)


if __name__ == "__main__":
    main()