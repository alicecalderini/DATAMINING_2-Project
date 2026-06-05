# ─────────────────────────────────────────────
# Hierarchical Clustering on precomputed DTW distance matrix
# Assumes X_3d, dist_mat, motifs_df, CHANNELS, W
# are already in memory from motif_clustering_dtw.py
# ─────────────────────────────────────────────
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1.  BUILD LINKAGE MATRIX
#     squareform converts the (n,n) distance matrix
#     to the condensed (n*(n-1)/2,) vector scipy expects
# ─────────────────────────────────────────────
print("Building linkage matrix ...")
condensed = squareform(dist_mat)
Z         = linkage(condensed, method="ward")
print("  Done.")

# ─────────────────────────────────────────────
# 2.  DENDROGRAM  →  inspect to choose k
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
dendrogram(
    Z,
    truncate_mode="lastp",   # show only last p merges
    p=30,
    ax=ax,
    color_threshold=0.7 * max(Z[:, 2]),
    above_threshold_color="grey"
)
ax.set_title("Dendrogram (truncated to last 30 merges)")
ax.set_xlabel("Subject (or cluster size)")
ax.set_ylabel("DTW Ward distance")
ax.grid(axis="y", linestyle="--")
plt.tight_layout()
plt.savefig("11_dendrogram.png", dpi=150)
plt.show()
print("Saved → 11_dendrogram.png")

# ─────────────────────────────────────────────
# 3.  SWEEP k  →  silhouette on precomputed dist_mat
# ─────────────────────────────────────────────
k_range  = range(2, 17)
sil_list = []

print("\nSweeping k ...")
for k in k_range:
    labels = fcluster(Z, t=k, criterion="maxclust") - 1  # 0-indexed
    sil    = silhouette_score(dist_mat, labels, metric="precomputed")
    sil_list.append(sil)
    print(f"  k={k:2d}  silhouette={sil:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(k_range), sil_list, "s-", color="orange")
auto_best_k = list(k_range)[int(np.argmax(sil_list))]
ax.axvline(auto_best_k, ls="--", color="red", label=f"auto best k={auto_best_k}")
ax.set_xlabel("k"); ax.set_ylabel("Silhouette score (DTW)")
ax.set_title("Hierarchical clustering — silhouette sweep")
ax.legend(); ax.grid(linestyle="--")
plt.tight_layout()
plt.savefig("12_hierarchical_silhouette.png", dpi=150)
plt.show()
print("Saved → 12_hierarchical_silhouette.png")

# ─────────────────────────────────────────────
# 4.  CHOOSE k  (manual override or accept auto)
# ─────────────────────────────────────────────
user_input = input(f"\nAuto best k={auto_best_k}. Enter k to use (or press Enter to accept): ")
best_k     = int(user_input) if user_input.strip() else auto_best_k

cluster_labels = fcluster(Z, t=best_k, criterion="maxclust") - 1  # 0-indexed
motifs_df["cluster_hier"] = cluster_labels

# ─────────────────────────────────────────────
# 5.  CLUSTER ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("CLUSTER ANALYSIS")
print("="*50)

print("\n--- Cluster sizes ---")
cluster_counts = motifs_df["cluster_hier"].value_counts().sort_index()
for c, n in cluster_counts.items():
    print(f"  Cluster {c}: {n:4d} subjects  ({100*n/len(motifs_df):.1f}%)")

print("\n--- Subject indices per cluster (first 10 shown) ---")
for c in range(best_k):
    members = motifs_df.loc[motifs_df["cluster_hier"] == c, "sample_index"].tolist()
    print(f"  Cluster {c} ({len(members)} subjects): "
          f"{members[:10]}{'...' if len(members) > 10 else ''}")

# Centroids: subject closest to the mean DTW distance within each cluster
# (more meaningful than mean window for hierarchical)
print("\n--- Most central subject per cluster ---")
for c in range(best_k):
    mask     = np.where(cluster_labels == c)[0]
    sub_dist = dist_mat[np.ix_(mask, mask)]
    central  = mask[np.argmin(sub_dist.mean(axis=1))]
    print(f"  Cluster {c}: most central subject = sample_index "
          f"{motifs_df.iloc[central]['sample_index']}  "
          f"(avg DTW dist = {sub_dist.mean(axis=1).min():.4f})")

# Motif shape summary per cluster per channel
print("\n--- Centroid motif shape summary (mean ± std across cluster members) ---")
for c in range(best_k):
    mask = cluster_labels == c
    print(f"\n  Cluster {c}  (n={mask.sum()}):")
    for i, ch in enumerate(CHANNELS):
        ch_windows = X_3d[mask, :, i]   # (n_members, W)
        mean_pat   = ch_windows.mean(axis=0)
        print(f"    {ch:>8s}: mean={mean_pat.mean():.4f}  "
              f"std={ch_windows.std():.4f}  "
              f"min={mean_pat.min():.4f}  max={mean_pat.max():.4f}")

motifs_df[["sample_index", "cluster_hier"]].to_csv(
    "subject_clusters_hierarchical.csv", index=False)
print("\nCluster assignments saved → subject_clusters_hierarchical.csv")

# ─────────────────────────────────────────────
# 6.  CENTROID MOTIF SHAPE PLOTS
#     One figure per cluster, one subplot per channel
# ─────────────────────────────────────────────
for c in range(best_k):
    mask     = cluster_labels == c
    n_members = mask.sum()

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    fig.suptitle(f"Cluster {c}  —  {n_members} subjects  "
                 f"({100*n_members/len(cluster_labels):.1f}%)", fontsize=13)
    axes = axes.flatten()

    for i, ch in enumerate(CHANNELS):
        member_seqs = X_3d[mask, :, i]          # (n_members, W)
        centroid_ch = member_seqs.mean(axis=0)  # (W,)

        for seq in member_seqs:
            axes[i].plot(seq, color="grey", alpha=0.15, linewidth=0.8)
        axes[i].plot(centroid_ch, color="black", linewidth=2, label="Mean")
        axes[i].set_title(ch)
        axes[i].set_xlabel("Time step")
        axes[i].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"13_hier_centroid_cluster{c}.png", dpi=150)
    plt.show()
    print(f"Saved → 13_hier_centroid_cluster{c}.png")

# ─────────────────────────────────────────────
# 7.  PCA + t-SNE
# ─────────────────────────────────────────────
CMAP   = "tab10"
X_flat = X_3d.reshape(len(X_3d), -1)

# — PCA
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_flat)

fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                c=cluster_labels, cmap=CMAP, s=15, alpha=0.7)
for c in range(best_k):
    mask = cluster_labels == c
    cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
    ax.annotate(f"C{c}\n(n={mask.sum()})", (cx, cy), fontsize=8,
                fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
ax.set_title(f"PCA — Hierarchical clustering  (k={best_k})")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.colorbar(sc, ax=ax, label="Cluster")
plt.tight_layout()
plt.savefig("14_hier_pca.png", dpi=150)
plt.show()
print("Saved → 14_hier_pca.png")

# — t-SNE (precomputed DTW distances)
print("Running t-SNE ...")
tsne   = TSNE(n_components=2, perplexity=30, random_state=42,
              metric="precomputed", init="random")
X_tsne = tsne.fit_transform(dist_mat)

fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                c=cluster_labels, cmap=CMAP, s=15, alpha=0.7)
for c in range(best_k):
    mask = cluster_labels == c
    cx, cy = X_tsne[mask, 0].mean(), X_tsne[mask, 1].mean()
    ax.annotate(f"C{c}\n(n={mask.sum()})", (cx, cy), fontsize=8,
                fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
ax.set_title(f"t-SNE — Hierarchical clustering  (k={best_k})")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
plt.colorbar(sc, ax=ax, label="Cluster")
plt.tight_layout()
plt.savefig("15_hier_tsne.png", dpi=150)
plt.show()
print("Saved → 15_hier_tsne.png")
