"""
module9_clustering.py — Startup Ecosystem Clustering
======================================================

OBJECTIVE
---------
Group the 15 countries into distinct startup ecosystem clusters
based on their economic and startup characteristics.
Find natural groupings WITHOUT using the target variable —
this is unsupervised learning.

WHY CLUSTERING
--------------
Correlation and regression tell us WHAT predicts growth.
Clustering tells us WHICH countries are structurally similar.
This answers:
  - Are there "ecosystem archetypes" in startup growth?
  - Which countries could learn from each other?
  - Does pandemic response differ by cluster?

ALGORITHM: K-Means
-------------------
1. Initialise K centroids randomly (K-Means++ for stability)
2. Assign each point to nearest centroid:
   cluster(x) = argmin_k ||x - mu_k||²
3. Update centroids: mu_k = mean of all points in cluster k
4. Repeat 2-3 until convergence (centroids stop moving)

K SELECTION
-----------
Elbow method:   plot inertia (within-cluster SS) vs K
                pick K where marginal gain flattens
Silhouette:     s(i) = (b(i) - a(i)) / max(a(i), b(i))
                a(i) = mean distance to same-cluster points
                b(i) = mean distance to nearest other cluster
                s ∈ [-1, +1], higher = better separation

FIGURES (10)
------------
 1. Elbow curve — inertia vs K
 2. Silhouette scores vs K
 3. Silhouette diagram for optimal K
 4. Cluster scatter — PCA 2D projection
 5. Cluster scatter — UMAP (if available) else t-SNE
 6. Cluster profiles — radar chart per cluster
 7. Feature means heatmap per cluster
 8. Country-cluster assignment bar chart
 9. Pandemic growth by cluster (pre vs post)
10. Cluster stability — bootstrapped silhouette distribution

INPUTS
------
- data/master_features.csv

OUTPUTS
-------
- data/outputs/figures/module9/*.png  (10 figures)
- data/outputs/reports/module9_clustering_report.txt
- data/cluster_assignments.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.utils import resample

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utils import get_logger, get_project_root

logger  = get_logger("module9_clustering")
ROOT    = get_project_root()

FIG_DIR = ROOT / "data" / "outputs" / "figures" / "module9"
REP_DIR = ROOT / "data" / "outputs" / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
RANDOM_STATE = 42

# Cluster colour palette — up to 6 clusters
CLUSTER_COLORS = ["#e74c3c", "#3498db", "#2ecc71",
                  "#f39c12", "#9b59b6", "#1abc9c"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    candidates = [
        ROOT / "data" / "master_features.csv",
        ROOT / "data" / "processed" / "master_features.csv",
        ROOT / "data" / "master_clean.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            logger.info(f"Loaded: {path.name}  shape={df.shape}")
            return df
    raise FileNotFoundError("No master dataset found. Run Module 6 first.")


def prepare_cluster_features(df):
    """
    Select features for clustering.
    Exclude target, ID columns, and binary flags.
    Use country-level means across years for a stable profile.
    """
    # Columns to exclude
    exclude_patterns = [
        "country", "year", "period", "country_code",
        "startup_count_yoy", "startup_growth_yoy",
        "startup_count_growth_rate", "yoy_growth",
        "is_pandemic", "is_post_pandemic", "is_partial",
        "strong_recovery", "high_innovation",
        "_scaled",
    ]

    feature_cols = []
    for col in df.select_dtypes(include="number").columns:
        if not any(p in col for p in exclude_patterns):
            feature_cols.append(col)

    # Aggregate to country level (mean across years)
    country_col = next((c for c in ["country", "country_name", "country_code"]
                    if c in df.columns), None)
    if country_col:
        country_df = df.groupby(country_col)[feature_cols].mean()
        country_df = country_df.reset_index().rename(columns={country_col: "country"})
        country_df = country_df.set_index("country")
        country_df["country"] = country_df.index
    else:
        country_df = df[feature_cols].copy()
        country_df["country"] = [f"Country_{i}" for i in range(len(df))]

    countries   = country_df["country"].values
    X_raw       = country_df[feature_cols].fillna(0).values

    # Standardise (K-Means is distance-based — scale matters)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    logger.info(f"Cluster features: {len(feature_cols)}")
    logger.info(f"Countries: {len(countries)}")
    logger.info(f"Feature list: {feature_cols[:10]}...")

    return X_scaled, X_raw, countries, feature_cols, country_df, scaler


# ─────────────────────────────────────────────────────────────────────────────
# K SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_optimal_k(X, k_range=range(2, 8)):
    """
    Compute inertia and silhouette score for each K.
    Returns dict of results and optimal K.
    """
    inertias    = []
    sil_scores  = []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++",
                    n_init=20, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels)
        sil_scores.append(sil)
        logger.info(f"  K={k}  inertia={km.inertia_:.2f}  silhouette={sil:.4f}")

    # Optimal K = highest silhouette
    best_k = list(k_range)[int(np.argmax(sil_scores))]
    logger.info(f"Optimal K = {best_k}  (silhouette={max(sil_scores):.4f})")

    return {
        "k_range":    list(k_range),
        "inertias":   inertias,
        "sil_scores": sil_scores,
        "best_k":     best_k,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def fit_kmeans(X, k):
    km = KMeans(n_clusters=k, init="k-means++",
                n_init=50, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    sil    = silhouette_score(X, labels)
    logger.info(f"Final K-Means: K={k}  silhouette={sil:.4f}")
    return km, labels, sil


def name_clusters(labels, countries, country_df, feature_cols):
    """
    Give each cluster a descriptive name based on its top features.
    """
    cluster_ids = sorted(set(labels))
    names = {}

    for cid in cluster_ids:
        mask     = labels == cid
        members  = countries[mask]
        # Simple naming: most distinctive feature direction
        means    = country_df.loc[mask, feature_cols].mean()
        top_feat = means.abs().idxmax()
        direction = "High" if means[top_feat] > 0 else "Low"
        names[cid] = f"Cluster {cid+1}"
        logger.info(f"  {names[cid]}: {list(members)}  (top: {direction} {top_feat})")

    return names


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def savefig(name):
    path = FIG_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {name}")
    return path


def fig1_elbow(k_sel):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_sel["k_range"], k_sel["inertias"],
            "o-", color="#3498db", linewidth=2.5, markersize=8)
    ax.axvline(k_sel["best_k"], color="red", linestyle="--",
               linewidth=1.5, label=f"Optimal K={k_sel['best_k']}")
    ax.set_title("Elbow Method — Within-Cluster Sum of Squares vs K",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.legend()
    plt.tight_layout()
    savefig("01_elbow_curve.png")


def fig2_silhouette_scores(k_sel):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ["#e74c3c" if k == k_sel["best_k"] else "#3498db"
               for k in k_sel["k_range"]]
    bars = ax.bar(k_sel["k_range"], k_sel["sil_scores"],
                  color=colors, edgecolor="white", alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_title("Silhouette Score vs K\n(higher = better cluster separation)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.axhline(max(k_sel["sil_scores"]), color="red",
               linestyle="--", linewidth=1, alpha=0.5)
    plt.tight_layout()
    savefig("02_silhouette_scores.png")


def fig3_silhouette_diagram(X, labels, k):
    sil_vals = silhouette_samples(X, labels)
    fig, ax  = plt.subplots(figsize=(10, 6))
    y_lower  = 10

    for cid in range(k):
        vals = np.sort(sil_vals[labels == cid])
        size = len(vals)
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         alpha=0.8, color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                         label=f"Cluster {cid+1}")
        y_lower = y_upper + 5

    avg = silhouette_score(X, labels)
    ax.axvline(avg, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean silhouette={avg:.3f}")
    ax.set_title(f"Silhouette Diagram — K={k}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    savefig("03_silhouette_diagram.png")


def fig4_pca_scatter(X, labels, countries, k):
    pca    = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d   = pca.fit_transform(X)
    var_ex = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(11, 7))
    for cid in range(k):
        mask = labels == cid
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                   s=120, alpha=0.85, edgecolors="white", linewidth=1.2,
                   label=f"Cluster {cid+1}", zorder=3)
        for i, c in enumerate(countries[mask]):
            ax.annotate(c, (X_2d[mask][i, 0], X_2d[mask][i, 1]),
                        fontsize=7.5, ha="center", va="bottom",
                        xytext=(0, 6), textcoords="offset points")

    ax.set_title(f"Startup Ecosystem Clusters — PCA 2D Projection\n"
                 f"PC1={var_ex[0]*100:.1f}%  PC2={var_ex[1]*100:.1f}% variance explained",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var_ex[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ex[1]*100:.1f}%)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig("04_pca_scatter.png")
    return X_2d


def fig5_tsne_scatter(X, labels, countries, k):
    """t-SNE 2D projection (alternative view to PCA)."""
    from sklearn.manifold import TSNE
    # n_samples may be small — perplexity must be < n_samples
    perp = min(5, len(countries) - 1)
    tsne = TSNE(n_components=2, perplexity=perp,
            random_state=RANDOM_STATE, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(11, 7))
    for cid in range(k):
        mask = labels == cid
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                   s=120, alpha=0.85, edgecolors="white", linewidth=1.2,
                   label=f"Cluster {cid+1}", zorder=3)
        for i, c in enumerate(countries[mask]):
            ax.annotate(c, (X_2d[mask][i, 0], X_2d[mask][i, 1]),
                        fontsize=7.5, ha="center", va="bottom",
                        xytext=(0, 6), textcoords="offset points")

    ax.set_title("Startup Ecosystem Clusters — t-SNE Projection",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig("05_tsne_scatter.png")


def fig6_radar_chart(country_df, labels, feature_cols, k):
    """Radar chart showing cluster profiles on top features."""
    # Pick top 6 most variable features for readability
    variances  = country_df[feature_cols].var()
    top_feats  = variances.sort_values(ascending=False).head(6).index.tolist()

    # Normalise to [0,1] for radar
    radar_df = country_df[top_feats].copy()
    for col in top_feats:
        lo, hi = radar_df[col].min(), radar_df[col].max()
        radar_df[col] = (radar_df[col] - lo) / (hi - lo + 1e-9)

    N      = len(top_feats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for cid in range(k):
        mask  = labels == cid
        means = radar_df[mask].mean().values.tolist()
        means += means[:1]
        ax.plot(angles, means, "o-", linewidth=2,
                color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                label=f"Cluster {cid+1}")
        ax.fill(angles, means, alpha=0.1,
                color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("_", "\n") for f in top_feats], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Cluster Profiles — Radar Chart\n(normalised feature means)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    savefig("06_radar_chart.png")


def fig7_feature_heatmap(country_df, labels, feature_cols, k):
    """Heatmap: mean feature value per cluster."""
    variances = country_df[feature_cols].var()
    top_feats = variances.sort_values(ascending=False).head(12).index.tolist()

    rows = []
    for cid in range(k):
        mask = labels == cid
        row  = country_df[mask][top_feats].mean()
        row.name = f"Cluster {cid+1}"
        rows.append(row)
    heat = pd.DataFrame(rows)

    # Normalise each column for visual clarity
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(14, max(4, k + 1)))
    sns.heatmap(heat_norm, annot=heat.round(2), fmt="g",
                cmap="RdYlGn", linewidths=0.4, ax=ax,
                cbar_kws={"label": "Normalised mean"},
                annot_kws={"size": 8})
    ax.set_title("Cluster Feature Profiles — Mean Value per Cluster\n"
                 "(colour = normalised, numbers = raw means)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature"); ax.set_ylabel("Cluster")
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.tight_layout()
    savefig("07_feature_heatmap.png")


def fig8_country_assignment(countries, labels, k):
    """Bar chart of cluster assignment per country."""
    assign_df = pd.DataFrame({
        "country": countries,
        "cluster": [f"Cluster {l+1}" for l in labels]
    }).sort_values("cluster")

    fig, ax = plt.subplots(figsize=(12, 5))
    cluster_list = sorted(assign_df["cluster"].unique())
    x = np.arange(len(countries))
    colors = [CLUSTER_COLORS[int(l.split()[-1])-1 % len(CLUSTER_COLORS)]
              for l in assign_df["cluster"]]
    bars = ax.bar(assign_df["country"], assign_df["cluster"].map(
        {c: i for i, c in enumerate(cluster_list)}
    ), color=colors, edgecolor="white", alpha=0.85)
    ax.set_yticks(range(len(cluster_list)))
    ax.set_yticklabels(cluster_list)
    ax.set_title("Country → Cluster Assignment",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Country"); ax.set_ylabel("Cluster")
    plt.xticks(rotation=40, ha="right", fontsize=9)

    # Legend
    patches = [mpatches.Patch(color=CLUSTER_COLORS[i], label=f"Cluster {i+1}")
               for i in range(k)]
    ax.legend(handles=patches, fontsize=9)
    plt.tight_layout()
    savefig("08_country_assignment.png")


def fig9_pandemic_by_cluster(df, countries, labels):
    """Pre vs post pandemic growth rate by cluster."""
    # Detect target column
    target_candidates = [
        "startup_growth_yoy", "startup_count_growth_rate",
        "startup_count_yoy", "yoy_growth"
    ]
    target = None
    for t in target_candidates:
        if t in df.columns:
            target = t
            break
    if target is None:
        logger.warning("No target column found for fig9 — skipping")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Target column not found",
                ha="center", va="center", transform=ax.transAxes)
        savefig("09_pandemic_by_cluster.png")
        return

    # Map each row to its cluster
    country_cluster = {c: labels[i] for i, c in enumerate(countries)}
    df2 = df.copy()
    if "country" in df2.columns:
        df2["cluster"] = df2["country"].map(country_cluster)
        df2 = df2.dropna(subset=["cluster", target])
        df2["cluster"] = df2["cluster"].astype(int)

        # Determine period
        period_col = None
        for p in ["period", "is_pandemic", "is_post_pandemic"]:
            if p in df2.columns:
                period_col = p
                break

        if period_col == "period":
            pre  = df2[df2["period"] == "pre"].groupby("cluster")[target].mean()
            post = df2[df2["period"].isin(["during","post"])].groupby("cluster")[target].mean()
        else:
            yr_col = "year" if "year" in df2.columns else None
            if yr_col:
                pre  = df2[df2[yr_col] < 2020].groupby("cluster")[target].mean()
                post = df2[df2[yr_col] >= 2020].groupby("cluster")[target].mean()
            else:
                pre  = df2.groupby("cluster")[target].mean()
                post = pre.copy()

        k_found = df2["cluster"].nunique()
        x   = np.arange(k_found)
        w   = 0.38
        fig, ax = plt.subplots(figsize=(10, 5))
        clusters = sorted(pre.index)
        pre_vals  = [pre.get(c, 0) for c in clusters]
        post_vals = [post.get(c, 0) for c in clusters]
        ax.bar(x - w/2, pre_vals,  width=w, label="Pre-pandemic",
               color="#3498db", alpha=0.85, edgecolor="white")
        ax.bar(x + w/2, post_vals, width=w, label="Post-pandemic",
               color="#e74c3c", alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Cluster {c+1}" for c in clusters])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Startup Growth Rate by Cluster — Pre vs Post Pandemic",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Startup Growth Rate (%)")
        ax.legend()
        plt.tight_layout()
    else:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Country column not found",
                ha="center", va="center", transform=ax.transAxes)
    savefig("09_pandemic_by_cluster.png")


def fig10_cluster_stability(X, k, n_boot=50):
    """
    Bootstrap silhouette distribution — how stable are the clusters?
    Resample with replacement n_boot times, refit K-Means each time.
    """
    sil_boot = []
    n = X.shape[0]
    for i in range(n_boot):
        idx      = np.random.default_rng(i).choice(n, size=n, replace=True)
        X_boot   = X[idx]
        if len(set(range(k))) > X_boot.shape[0]:
            continue
        try:
            km  = KMeans(n_clusters=k, n_init=10, random_state=i)
            lbl = km.fit_predict(X_boot)
            if len(set(lbl)) == k:
                sil_boot.append(silhouette_score(X_boot, lbl))
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(sil_boot, bins=15, color="#9b59b6", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(sil_boot), color="red", linestyle="--",
               linewidth=1.5, label=f"Mean={np.mean(sil_boot):.3f}")
    ax.set_title(f"Cluster Stability — Bootstrap Silhouette Distribution\n"
                 f"K={k},  n_bootstrap={n_boot}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Silhouette Score"); ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    savefig("10_cluster_stability.png")
    logger.info(f"Bootstrap silhouette: mean={np.mean(sil_boot):.4f}  "
                f"std={np.std(sil_boot):.4f}")
    return np.mean(sil_boot), np.std(sil_boot)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE ASSIGNMENTS
# ─────────────────────────────────────────────────────────────────────────────

def save_assignments(countries, labels, k):
    out = pd.DataFrame({
        "country":      countries,
        "cluster_id":   labels,
        "cluster_label": [f"Cluster {l+1}" for l in labels],
    })
    path = ROOT / "data" / "cluster_assignments.csv"
    out.to_csv(path, index=False)
    logger.info(f"Cluster assignments saved: {path}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_report(k, k_sel, sil_score, boot_mean, boot_std,
                 countries, labels, assign_df):
    lines = [
        "=" * 60,
        "MODULE 9 — CLUSTERING REPORT",
        "=" * 60,
        "",
        f"Algorithm        : K-Means (k-means++ init, n_init=50)",
        f"Optimal K        : {k}",
        f"Silhouette score : {sil_score:.4f}",
        f"Bootstrap mean   : {boot_mean:.4f} +/- {boot_std:.4f}",
        "",
        "K SELECTION SUMMARY",
        "-" * 40,
    ]
    for k_val, sil, inert in zip(k_sel["k_range"],
                                  k_sel["sil_scores"],
                                  k_sel["inertias"]):
        marker = " <-- OPTIMAL" if k_val == k else ""
        lines.append(f"  K={k_val}  silhouette={sil:.4f}  "
                     f"inertia={inert:.1f}{marker}")

    lines += ["", "CLUSTER ASSIGNMENTS", "-" * 40]
    for cid in sorted(set(labels)):
        members = countries[labels == cid]
        lines.append(f"  Cluster {cid+1}: {', '.join(members)}")

    lines += [
        "",
        "INTERPRETATION",
        "-" * 40,
        "  Countries in the same cluster share structural",
        "  startup ecosystem characteristics.",
        "  Policy makers can use cluster peers as benchmarks.",
        "",
        "Figures saved: 10",
        f"Location: {FIG_DIR}",
        "",
        "=" * 60,
    ]

    path = REP_DIR / "module9_clustering_report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved: {path}")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("MODULE 9 — STARTUP ECOSYSTEM CLUSTERING")
    logger.info("=" * 60)

    np.random.seed(RANDOM_STATE)

    df = load_data()
    X_scaled, X_raw, countries, feature_cols, country_df, scaler = \
        prepare_cluster_features(df)

    # Select optimal K
    logger.info("Selecting optimal K...")
    k_sel = select_optimal_k(X_scaled, k_range=range(2, min(8, len(countries))))

    # Fit final model
    k      = k_sel["best_k"]
    km, labels, sil = fit_kmeans(X_scaled, k)

    # Name clusters
    name_clusters(labels, countries, country_df, feature_cols)

    # All 10 figures
    fig1_elbow(k_sel)
    fig2_silhouette_scores(k_sel)
    fig3_silhouette_diagram(X_scaled, labels, k)
    fig4_pca_scatter(X_scaled, labels, countries, k)
    fig5_tsne_scatter(X_scaled, labels, countries, k)
    fig6_radar_chart(country_df, labels, feature_cols, k)
    fig7_feature_heatmap(country_df, labels, feature_cols, k)
    fig8_country_assignment(countries, labels, k)
    fig9_pandemic_by_cluster(df, countries, labels)
    boot_mean, boot_std = fig10_cluster_stability(X_scaled, k)

    # Save assignments
    assign_df = save_assignments(countries, labels, k)

    # Report
    report_lines = write_report(
        k, k_sel, sil, boot_mean, boot_std,
        countries, labels, assign_df
    )

    logger.info("=" * 60)
    logger.info(f"MODULE 9 COMPLETE — 10 figures + 1 report")
    logger.info(f"Optimal K={k}  silhouette={sil:.4f}")
    logger.info(f"Location: {FIG_DIR}")
    for f in sorted(FIG_DIR.glob("*.png")):
        logger.info(f"  {f.name}")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("  MODULE 9 COMPLETE")
    print("=" * 60)
    for line in report_lines:
        print(" ", line)
    print(f"\n  Next: python scripts\\run_module10.py  (Forecasting)")


if __name__ == "__main__":
    main()
