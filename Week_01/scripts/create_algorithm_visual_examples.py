#!/usr/bin/env python3
"""
Create Visual Examples for Each Clustering Algorithm
Shows how K-means, DBSCAN, Hierarchical, and GMM work on the same dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure with subplots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Generate sample dataset with challenging characteristics
# Mix of blobs and non-convex shapes
X1, y1 = make_blobs(n_samples=150, centers=3, n_features=2, 
                    cluster_std=0.5, random_state=42)
X2, y2 = make_moons(n_samples=100, noise=0.1, random_state=42)
X2 = StandardScaler().fit_transform(X2)
X2 *= 2
X2[:, 0] += 5
X2[:, 1] += 3

X = np.vstack([X1, X2])

# Define clustering algorithms
algorithms = [
    ('K-Means\n(K=5)', KMeans(n_clusters=5, random_state=42, n_init=10)),
    ('DBSCAN\n(eps=0.5)', DBSCAN(eps=0.5, min_samples=5)),
    ('Hierarchical\n(n=5)', AgglomerativeClustering(n_clusters=5)),
    ('GMM\n(n=5)', GaussianMixture(n_components=5, random_state=42))
]

# Color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Row 1: Algorithm results
for idx, (name, algorithm) in enumerate(algorithms):
    ax = axes[0, idx]
    
    # Fit and predict
    if name.startswith('GMM'):
        labels = algorithm.fit_predict(X)
    else:
        labels = algorithm.fit_predict(X)
    
    # Handle outliers for DBSCAN (label -1)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # Plot clusters
    for label in unique_labels:
        if label == -1:
            # Outliers in DBSCAN
            mask = labels == label
            ax.scatter(X[mask, 0], X[mask, 1], c='gray', s=20, 
                      alpha=0.3, marker='x', label='Outliers')
        else:
            mask = labels == label
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[label % len(colors)], 
                      s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Add cluster centers for K-means and GMM
    if name.startswith('K-Means'):
        centers = algorithm.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, 
                  marker='*', edgecolors='white', linewidth=2, zorder=5)
        ax.text(0.5, 0.95, f'Centers shown as stars', 
               transform=ax.transAxes, fontsize=8, ha='center')
    elif name.startswith('GMM'):
        centers = algorithm.means_
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, 
                  marker='*', edgecolors='white', linewidth=2, zorder=5)
        # Draw covariance ellipses
        for i in range(algorithm.n_components):
            covar = algorithm.covariances_[i]
            if algorithm.covariance_type == 'full':
                v, w = np.linalg.eigh(covar)
                angle = np.arctan2(w[1, 0], w[0, 0])
                angle = 180 * angle / np.pi
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                from matplotlib.patches import Ellipse
                ell = Ellipse(centers[i], v[0], v[1], angle=angle,
                            color=colors[i % len(colors)], alpha=0.2)
                ax.add_patch(ell)
    
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1' if idx == 0 else '')
    ax.set_ylabel('Feature 2' if idx == 0 else '')
    ax.text(0.5, 0.02, f'{n_clusters} clusters found', 
           transform=ax.transAxes, fontsize=9, ha='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.grid(True, alpha=0.3)

# Row 2: Algorithm characteristics
characteristics = [
    # K-means
    "✓ Fast and scalable\n✓ Spherical clusters\n✗ Fixed K required\n✗ Sensitive to outliers",
    # DBSCAN
    "✓ Finds arbitrary shapes\n✓ Identifies outliers\n✓ No K needed\n✗ Sensitive to parameters",
    # Hierarchical
    "✓ Dendrogram output\n✓ No K needed initially\n✓ Interpretable\n✗ Computationally expensive",
    # GMM
    "✓ Soft assignments\n✓ Elliptical clusters\n✓ Probabilistic\n✗ Assumes Gaussian distribution"
]

for idx, (name, chars) in enumerate(zip([n for n, _ in algorithms], characteristics)):
    ax = axes[1, idx]
    ax.axis('off')
    
    # Add algorithm name
    ax.text(0.5, 0.9, name.replace('\n', ' '), fontsize=13, 
           fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Add characteristics
    ax.text(0.1, 0.7, chars, fontsize=10, transform=ax.transAxes,
           verticalalignment='top')
    
    # Add use case
    use_cases = [
        "Best for: Quick segmentation\nwith known cluster count",
        "Best for: Anomaly detection\nand irregular patterns",
        "Best for: Taxonomies and\nexploring relationships",
        "Best for: Overlapping groups\nand uncertainty modeling"
    ]
    
    ax.text(0.5, 0.25, use_cases[idx], fontsize=9, ha='center',
           transform=ax.transAxes, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Add complexity
    complexities = ["O(nkt)", "O(n log n)", "O(n²)", "O(nkt)"]
    ax.text(0.5, 0.05, f"Complexity: {complexities[idx]}", 
           fontsize=9, ha='center', transform=ax.transAxes,
           color='gray')

# Overall title
fig.suptitle('Clustering Algorithms Visual Comparison\nSame Data, Different Approaches', 
            fontsize=16, fontweight='bold', y=1.02)

# Add dataset description
fig.text(0.5, -0.02, 
        'Dataset: Mix of 3 Gaussian blobs and 2 moon-shaped clusters (250 points total)',
        ha='center', fontsize=10, style='italic')

plt.tight_layout()

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/algorithm_visual_examples.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/algorithm_visual_examples.png', 
           dpi=150, bbox_inches='tight')

print("Algorithm visual examples created successfully!")
print("Files saved:")
print("  - charts/algorithm_visual_examples.pdf")
print("  - charts/algorithm_visual_examples.png")