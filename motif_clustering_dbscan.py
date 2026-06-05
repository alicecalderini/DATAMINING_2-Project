# ─────────────────────────────────────────────
# DBSCAN on precomputed DTW distance matrix
# Assumes X_3d and dist_mat are already in memory
# from motif_clustering_dtw.py — run that first.
# ─────────────────────────────────────────────
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1.  K-DISTANCE PLOT  →  choose eps visually
# ─────────────────────────────────────────────
MINPTS_FOR_EPS = 16

# Use the precomputed DTW dist_mat directly
sorted_dists = np.sort(dist_mat, axis=1)   # for each point, sorted distances to all others
kth_dists    = np.sort(sorted_dists[:, MINPTS_FOR_EPS])  # k-th nearest neighbour distance

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(kth_dists)
ax.set_xlabel("Points sorted by distance")
ax.set_ylabel(f"{MINPTS_FOR_EPS}-NN DTW distance")
ax.set_title("k-distance plot — look for the elbow to choose eps")
ax.grid(axis="y", linestyle="--")
plt.tight_layout()
plt.savefig("06_dbscan_optimal_eps.png", dpi=150)
plt.show()
print("Saved → 06_dbscan_optimal_eps.png")

EPS = float(input("\nEnter eps value from the elbow plot: "))

# ─────────────────────────────────────────────
# 2.  SWEEP min_samples
# ─────────────────────────────────────────────
minpts_range = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
sil_list     = []
valid_minpts = []

print(f"\nSweeping min_samples with eps={EPS} ...")
for minpts in minpts_range:
    db = DBSCAN(eps=EPS, min_samples=minpts, metric="precomputed")
    db.fit(dist_mat)

    non_noise  = db.labels_ != -1
    n_clusters = len(set(db.labels_[non_noise]))

    if non_noise.sum() > 1 and n_clusters > 1:
        sil = silhouette_score(dist_mat[np.ix_(non_noise, non_noise)],
                               db.labels_[non_noise],
                               metric="precomputed")
        sil_list.append(sil)
        valid_minpts.append(minpts)
        print(f"  min_samples={minpts:5d}  clusters={n_clusters}  "
              f"noise={( ~non_noise).sum()}  silhouette={sil:.4f}")
    else:
        print(f"  min_samples={minpts:5d}  → no valid clusters, skipped")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(valid_minpts, sil_list, "o-")
ax.set_xlabel("min_samples"); ax.set_ylabel("Silhouette score (DTW)")
ax.set_title(f"Silhouette vs min_samples  (eps={EPS})")
ax.set_xscale("log"); ax.grid(linestyle="--")
plt.tight_layout()
plt.savefig("07_dbscan_silhouette_sweep.png", dpi=150)
plt.show()
print("Saved → 07_dbscan_silhouette_sweep.png")

# ─────────────────────────────────────────────
# 3.  FIT FINAL DBSCAN
# ─────────────────────────────────────────────
best_minpts = valid_minpts[int(np.argmax(sil_list))]
print(f"\nBest min_samples by silhouette: {best_minpts}")

db_final       = DBSCAN(eps=EPS, min_samples=best_minpts, metric="precomputed")
db_final.fit(dist_mat)
cluster_labels = db_final.labels_

non_noise_mask = cluster_labels != -1
n_noise        = (~non_noise_mask).sum()
unique_labels  = sorted(set(cluster_labels))
n_clusters     = len([l for l in unique_labels if l != -1])

sil_final = silhouette_score(
    dist_mat[np.ix_(non_noise_mask, non_noise_mask)],
    cluster_labels[non_noise_mask],
    metric="precomputed"
)
print(f"  Clusters found : {n_clusters}")
print(f"  Noise points   : {n_noise}  ({100*n_noise/len(cluster_labels):.1f}%)")
print(f"  Silhouette     : {sil_final:.4f}")

motifs_df["cluster_dbscan"] = cluster_labels

# ─────────────────────────────────────────────
# 4.  CLUSTER ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("CLUSTER ANALYSIS")
print("="*50)

print("\n--- Cluster sizes (label -1 = noise) ---")
cluster_counts = motifs_df["cluster_dbscan"].value_counts().sort_index()
for c, n in cluster_counts.items():
    tag = " [NOISE]" if c == -1 else ""
    print(f"  Cluster {c:>3d}{tag}: {n:4d} subjects  ({100*n/len(motifs_df):.1f}%)")

print("\n--- Subject indices per cluster (first 10 shown) ---")
for c in unique_labels:
    members = motifs_df.loc[motifs_df["cluster_dbscan"] == c, "sample_index"].tolist()
    tag = " [NOISE]" if c == -1 else ""
    print(f"  Cluster {c}{tag} ({len(members)} subjects): "
          f"{members[:10]}{'...' if len(members) > 10 else ''}")

# Pseudo-centroids: subject closest to the mean DTW distance in each cluster
print("\n--- Pseudo-centroids (most central subject per cluster) ---")
for c in unique_labels:
    if c == -1:
        continue
    mask       = np.where(cluster_labels == c)[0]
    sub_dist   = dist_mat[np.ix_(mask, mask)]
    mean_dists = sub_dist.mean(axis=1)
    central_i  = mask[np.argmin(mean_dists)]
    print(f"  Cluster {c}: most central subject = sample_index "
          f"{motifs_df.iloc[central_i]['sample_index']}  "
          f"(avg DTW dist to cluster = {mean_dists.min():.4f})")

motifs_df[["sample_index", "cluster_dbscan"]].to_csv(
    "subject_clusters_dbscan.csv", index=False)
print("\nCluster assignments saved → subject_clusters_dbscan.csv")

# ─────────────────────────────────────────────
# 5.  PCA + t-SNE  (reuse dist_mat, same as DTW script)
# ─────────────────────────────────────────────
def make_colors(labels, cmap_name="Spectral"):
    unique  = sorted(set(labels))
    n_real  = len([l for l in unique if l != -1])
    cmap    = plt.cm.get_cmap(cmap_name, max(n_real, 1))
    ci      = 0
    color_map = {}
    for l in unique:
        color_map[l] = "lightgrey" if l == -1 else cmap(ci / max(n_real - 1, 1))
        if l != -1:
            ci += 1
    return np.array([color_map[l] for l in labels])

colors = make_colors(cluster_labels)

handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=("lightgrey" if l == -1
                                else plt.cm.Spectral(i / max(n_clusters - 1, 1))),
               markersize=7,
               label=("Noise" if l == -1 else f"Cluster {l}"))
    for i, l in enumerate(unique_labels)
]

# — PCA (flatten X_3d for projection only)
X_flat = X_3d.reshape(len(X_3d), -1)
pca    = PCA(n_components=2, random_state=42)
X_pca  = pca.fit_transform(X_flat)

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=15, alpha=0.7)
for c in unique_labels:
    if c == -1:
        continue
    mask = cluster_labels == c
    cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
    ax.annotate(f"C{c}\n(n={mask.sum()})", (cx, cy), fontsize=8,
                fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
ax.set_title(f"PCA — DBSCAN  (eps={EPS}, min_samples={best_minpts})")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.legend(handles=handles, loc="upper left", title="Clusters", fontsize=7)
plt.tight_layout()
plt.savefig("08_dbscan_pca.png", dpi=150)
plt.show()
print("Saved → 08_dbscan_pca.png")

# — t-SNE (precomputed DTW distances, same as DTW script)
print("Running t-SNE ...")
tsne   = TSNE(n_components=2, perplexity=30, random_state=42,
              metric="precomputed", init="random")
X_tsne = tsne.fit_transform(dist_mat)

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, s=15, alpha=0.7)
for c in unique_labels:
    if c == -1:
        continue
    mask = cluster_labels == c
    cx, cy = X_tsne[mask, 0].mean(), X_tsne[mask, 1].mean()
    ax.annotate(f"C{c}\n(n={mask.sum()})", (cx, cy), fontsize=8,
                fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
ax.set_title(f"t-SNE — DBSCAN  (eps={EPS}, min_samples={best_minpts})")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(handles=handles, loc="upper left", title="Clusters", fontsize=7)
plt.tight_layout()
plt.savefig("09_dbscan_tsne.png", dpi=150)
plt.show()
print("Saved → 09_dbscan_tsne.png")
