import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
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

# Generate sample data with clear clusters
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                       cluster_std=0.5, random_state=42)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Perform clustering
n_clusters = 4
clusterer = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = clusterer.fit_predict(X)

# Calculate silhouette scores
silhouette_avg = silhouette_score(X, cluster_labels)
sample_silhouette_values = silhouette_samples(X, cluster_labels)

# LEFT: Silhouette plot
ax1.set_title(f'Silhouette Analysis\nAverage Score: {silhouette_avg:.3f}', 
              fontsize=14, fontweight='bold', color=colors['mlblue'])

y_lower = 10
for i in range(n_clusters):
    # Aggregate silhouette scores for samples belonging to cluster i
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = cm.nipy_spectral(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    
    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
    # Compute the new y_lower for next plot
    y_lower = y_upper + 10

ax1.set_xlabel("Silhouette Coefficient Values", fontsize=12)
ax1.set_ylabel("Cluster Label", fontsize=12)

# Add vertical line for average silhouette score
ax1.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2)
ax1.text(silhouette_avg + 0.02, ax1.get_ylim()[1] * 0.9, 
         f'Average: {silhouette_avg:.3f}', color='red', fontweight='bold')

# Add interpretation guide
ax1.text(0.8, ax1.get_ylim()[1] * 0.3, 
         'Interpretation:\n> 0.7: Strong\n0.5-0.7: Reasonable\n0.25-0.5: Weak\n< 0.25: Poor',
         fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))

ax1.set_xlim([-0.1, 1])
ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
ax1.grid(True, alpha=0.3)

# RIGHT: Clustered data
ax2.set_title('Clustered Data Points', fontsize=14, fontweight='bold', color=colors['mlgreen'])

# Plot the clusters
scatter_colors = [cm.nipy_spectral(float(i) / n_clusters) for i in cluster_labels]
ax2.scatter(X[:, 0], X[:, 1], marker='o', s=50, alpha=0.7, c=scatter_colors, edgecolors='black')

# Plot cluster centers
centers = clusterer.cluster_centers_
ax2.scatter(centers[:, 0], centers[:, 1], marker='*',
            c='red', s=500, alpha=1, edgecolors='black', linewidth=2)

# Add cluster labels
for i, center in enumerate(centers):
    ax2.text(center[0], center[1] - 0.3, f'C{i}', 
            fontsize=12, fontweight='bold', ha='center')

# Add score annotation
ax2.text(0.02, 0.98, f'Silhouette Score: {silhouette_avg:.3f}',
         transform=ax2.transAxes, fontsize=12, fontweight='bold',
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.suptitle('Silhouette Score: Measuring Cluster Cohesion and Separation', 
             fontsize=18, fontweight='bold', y=1.02)

plt.tight_layout()

# Save the figure
plt.savefig('../charts/silhouette_score.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/silhouette_score.png', dpi=150, bbox_inches='tight')
print("Silhouette score visualization created successfully!")
plt.close()