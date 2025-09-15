import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.datasets import make_blobs
import matplotlib.cm as cm

# Set random seed for reproducibility
np.random.seed(42)

# Define color palette
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlbrown': '#8c564b',
    'mlpink': '#e377c2',
    'mlgray': '#7f7f7f'
}

# Generate sample data with varying cluster quality
X, y_true = make_blobs(n_samples=400, centers=4, n_features=2, 
                       cluster_std=[0.5, 1.0, 0.7, 0.9], random_state=42)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Calculate Davies-Bouldin scores for different k values
K_range = range(2, 9)
db_scores = []
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    db_score = davies_bouldin_score(X, labels)
    db_scores.append(db_score)

# LEFT: Davies-Bouldin Index comparison
ax1.set_title('Davies-Bouldin Index\n(Lower is Better)', 
              fontsize=14, fontweight='bold', color=colors['mlblue'])

# Plot DB scores
bars = ax1.bar(K_range, db_scores, color=colors['mlblue'], alpha=0.7, edgecolor='black')

# Highlight the best (lowest) score
best_k = K_range[np.argmin(db_scores)]
best_score = min(db_scores)
ax1.bar(best_k, best_score, color=colors['mlgreen'], alpha=0.9, edgecolor='black', linewidth=2)

# Add value labels on bars
for i, (k, score) in enumerate(zip(K_range, db_scores)):
    ax1.text(k, score + 0.02, f'{score:.2f}', ha='center', va='bottom', fontweight='bold')

# Add best K annotation
ax1.annotate(f'Best K={best_k}\nDB={best_score:.2f}', 
             xy=(best_k, best_score), 
             xytext=(best_k + 1, best_score + 0.3),
             arrowprops=dict(arrowstyle='->', color='green', linewidth=2),
             fontsize=12, color='green', fontweight='bold')

# Add interpretation guide
ax1.text(0.02, 0.95, 
         'DB Index Interpretation:\n' +
         '< 0.5: Excellent separation\n' +
         '0.5-1.0: Good clustering\n' +
         '1.0-1.5: Moderate quality\n' +
         '> 1.5: Poor separation',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))

ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Davies-Bouldin Index', fontsize=12)
ax1.set_ylim([0, max(db_scores) * 1.2])
ax1.grid(True, alpha=0.3, axis='y')

# RIGHT: Visualization of cluster separation for optimal K
ax2.set_title(f'Cluster Visualization (K={best_k})\nShowing Inter & Intra-cluster Distances', 
              fontsize=14, fontweight='bold', color=colors['mlgreen'])

# Perform clustering with optimal K
kmeans_best = KMeans(n_clusters=best_k, random_state=42)
labels_best = kmeans_best.fit_predict(X)
centers = kmeans_best.cluster_centers_

# Plot clusters
for k in range(best_k):
    mask = labels_best == k
    cluster_color = cm.nipy_spectral(float(k) / best_k)
    ax2.scatter(X[mask, 0], X[mask, 1], s=30, alpha=0.6, 
               c=[cluster_color], label=f'Cluster {k}', edgecolors='black', linewidth=0.5)

# Plot cluster centers
ax2.scatter(centers[:, 0], centers[:, 1], marker='*',
            c='red', s=500, alpha=1, edgecolors='black', linewidth=2)

# Draw lines between cluster centers to show inter-cluster distances
for i in range(best_k):
    for j in range(i + 1, best_k):
        ax2.plot([centers[i, 0], centers[j, 0]], 
                [centers[i, 1], centers[j, 1]], 
                'k--', alpha=0.3, linewidth=1)
        
        # Add distance label
        mid_x = (centers[i, 0] + centers[j, 0]) / 2
        mid_y = (centers[i, 1] + centers[j, 1]) / 2
        dist = np.linalg.norm(centers[i] - centers[j])
        ax2.text(mid_x, mid_y, f'{dist:.1f}', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Draw circles to show intra-cluster spread
for k in range(best_k):
    mask = labels_best == k
    cluster_points = X[mask]
    # Calculate average distance from center
    avg_dist = np.mean(np.linalg.norm(cluster_points - centers[k], axis=1))
    circle = plt.Circle(centers[k], avg_dist, fill=False, 
                        edgecolor=cm.nipy_spectral(float(k) / best_k), 
                        linewidth=2, alpha=0.5, linestyle=':')
    ax2.add_patch(circle)

# Add DB score annotation
ax2.text(0.02, 0.98, f'Davies-Bouldin Score: {best_score:.2f}',
         transform=ax2.transAxes, fontsize=12, fontweight='bold',
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Add legend
ax2.text(0.98, 0.02, 
         'Solid lines: Inter-cluster distance\n' +
         'Dotted circles: Intra-cluster spread',
         transform=ax2.transAxes, fontsize=10, 
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.suptitle('Davies-Bouldin Index: Balancing Cluster Separation and Cohesion', 
             fontsize=18, fontweight='bold', y=1.02)

plt.tight_layout()

# Save the figure
plt.savefig('../charts/davies_bouldin.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/davies_bouldin.png', dpi=150, bbox_inches='tight')
print("Davies-Bouldin visualization created successfully!")
plt.close()