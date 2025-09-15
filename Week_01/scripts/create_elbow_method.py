import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from matplotlib.patches import Circle

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

# Generate sample data with 4 true clusters
X, y_true = make_blobs(n_samples=500, centers=4, n_features=2, 
                       cluster_std=1.0, random_state=42)

# Create figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Calculate inertia for different k values
K_range = range(1, 10)
inertias = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# SUBPLOT 1: The Elbow Curve
ax1 = axes[0]
ax1.set_title('The Elbow Method\nFinding Optimal K', fontsize=14, fontweight='bold', color=colors['mlblue'])

ax1.plot(K_range, inertias, 'b-', linewidth=2, marker='o', markersize=8)
ax1.plot(K_range, inertias, 'bo', markersize=8)

# Highlight the elbow point
elbow_k = 4
elbow_idx = elbow_k - 1
ax1.plot(elbow_k, inertias[elbow_idx], 'ro', markersize=15)

# Add elbow annotation
ax1.annotate('Elbow Point\n(Optimal K=4)', 
             xy=(elbow_k, inertias[elbow_idx]), 
             xytext=(elbow_k + 1, inertias[elbow_idx] + 5000),
             arrowprops=dict(arrowstyle='->', color='red', linewidth=2),
             fontsize=12, color='red', fontweight='bold')

# Add shaded regions
ax1.axvspan(1, 3.5, alpha=0.2, color='red', label='Under-clustering')
ax1.axvspan(4.5, 9, alpha=0.2, color='orange', label='Over-clustering')
ax1.axvspan(3.5, 4.5, alpha=0.2, color='green', label='Optimal')

ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Within-Cluster Sum of Squares\n(Inertia)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# SUBPLOT 2: Under-clustering (K=2)
ax2 = axes[1]
ax2.set_title('K=2: Under-clustering\n(Too Few Clusters)', fontsize=14, fontweight='bold', color=colors['mlred'])

kmeans_2 = KMeans(n_clusters=2, random_state=42)
labels_2 = kmeans_2.fit_predict(X)

for k in range(2):
    mask = labels_2 == k
    ax2.scatter(X[mask, 0], X[mask, 1], s=30, alpha=0.6, 
               label=f'Cluster {k}')

ax2.scatter(kmeans_2.cluster_centers_[:, 0], 
           kmeans_2.cluster_centers_[:, 1], 
           marker='*', s=300, c='red', edgecolors='black', linewidth=2)

ax2.text(0.5, 0.02, f'Inertia: {kmeans_2.inertia_:.0f}',
         transform=ax2.transAxes, fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Show true cluster boundaries (faded)
for center in [[1, 2], [1, -2], [5, 3], [8, 3]]:
    circle = Circle(center, 2, fill=False, edgecolor='gray', 
                   linewidth=1, alpha=0.3, linestyle='--')
    ax2.add_patch(circle)

ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# SUBPLOT 3: Optimal clustering (K=4)
ax3 = axes[2]
ax3.set_title('K=4: Optimal Clustering\n(Just Right)', fontsize=14, fontweight='bold', color=colors['mlgreen'])

kmeans_4 = KMeans(n_clusters=4, random_state=42)
labels_4 = kmeans_4.fit_predict(X)

for k in range(4):
    mask = labels_4 == k
    ax3.scatter(X[mask, 0], X[mask, 1], s=30, alpha=0.6,
               label=f'Cluster {k}')

ax3.scatter(kmeans_4.cluster_centers_[:, 0], 
           kmeans_4.cluster_centers_[:, 1], 
           marker='*', s=300, c='red', edgecolors='black', linewidth=2)

ax3.text(0.5, 0.02, f'Inertia: {kmeans_4.inertia_:.0f}',
         transform=ax3.transAxes, fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax3.set_xlabel('Feature 1', fontsize=12)
ax3.set_ylabel('Feature 2', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

plt.suptitle('The Elbow Method: Finding the Right Number of Clusters', 
             fontsize=18, fontweight='bold', y=1.02)

plt.tight_layout()

# Save the figure
plt.savefig('../charts/elbow_method.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/elbow_method.png', dpi=150, bbox_inches='tight')
print("Elbow method visualization created successfully!")
plt.close()