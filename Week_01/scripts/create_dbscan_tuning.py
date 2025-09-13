#!/usr/bin/env python3
"""
Create DBSCAN Parameter Tuning Visualization for Week 1
Shows how eps and min_samples parameters affect clustering results
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# Generate sample innovation dataset with varying densities
# Mix of different innovation patterns
X1, _ = make_blobs(n_samples=150, centers=2, n_features=2, 
                   cluster_std=0.4, random_state=42)
X2, _ = make_moons(n_samples=100, noise=0.1, random_state=42)
X2 = StandardScaler().fit_transform(X2)
X2[:, 0] += 4
X2[:, 1] += 2

# Add some outliers (radical innovations)
X3 = np.random.uniform(-2, 6, (20, 2))

X = np.vstack([X1, X2, X3])

# Create grid for parameter variations
eps_values = [0.3, 0.5, 0.8]
min_samples_values = [3, 5, 10]

# Color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Create subplots grid
for i, eps in enumerate(eps_values):
    for j, min_samples in enumerate(min_samples_values):
        ax = plt.subplot(3, 3, i * 3 + j + 1)
        
        # Apply DBSCAN with current parameters
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Count clusters and outliers
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_outliers = list(labels).count(-1)
        
        # Plot results
        for label in unique_labels:
            if label == -1:
                # Outliers
                mask = labels == label
                ax.scatter(X[mask, 0], X[mask, 1], c='gray', s=20, 
                          alpha=0.3, marker='x', label='Outliers')
            else:
                mask = labels == label
                ax.scatter(X[mask, 0], X[mask, 1], c=colors[label % len(colors)], 
                          s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
        
        # Add title with parameters and results
        ax.set_title(f'eps={eps}, min_samples={min_samples}', fontsize=10, fontweight='bold')
        ax.text(0.5, 0.95, f'Clusters: {n_clusters}, Outliers: {n_outliers}', 
               transform=ax.transAxes, fontsize=8, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add parameter effect description
        if eps == 0.3 and min_samples == 3:
            ax.text(0.5, 0.02, 'Many small clusters', transform=ax.transAxes,
                   fontsize=7, ha='center', style='italic', color='red')
        elif eps == 0.3 and min_samples == 10:
            ax.text(0.5, 0.02, 'Too restrictive', transform=ax.transAxes,
                   fontsize=7, ha='center', style='italic', color='red')
        elif eps == 0.8 and min_samples == 3:
            ax.text(0.5, 0.02, 'May merge clusters', transform=ax.transAxes,
                   fontsize=7, ha='center', style='italic', color='orange')
        elif eps == 0.5 and min_samples == 5:
            ax.text(0.5, 0.02, 'Balanced', transform=ax.transAxes,
                   fontsize=7, ha='center', style='italic', color='green')
        
        ax.set_xlim(-2, 6)
        ax.set_ylim(-2, 4)
        ax.set_aspect('equal')
        
        # Remove individual axes labels for cleaner look
        if j == 0:
            ax.set_ylabel(f'eps = {eps}', fontsize=11, fontweight='bold')
        if i == 0:
            ax.set_xlabel(f'min_samples = {min_samples}', fontsize=11, fontweight='bold')
            ax.xaxis.set_label_position('top')

# Overall title
fig.suptitle('DBSCAN Parameter Tuning: Impact on Innovation Clustering', 
            fontsize=16, fontweight='bold', y=0.98)

# Add parameter explanation panel
explanation_text = (
    "Parameter Guidelines:\n\n"
    "eps (epsilon): Maximum distance between points in same cluster\n"
    "  * Small eps → Many small, tight clusters\n"
    "  * Large eps → Fewer, larger clusters\n"
    "  * Too large → All points in one cluster\n\n"
    "min_samples: Minimum points to form dense region\n"
    "  * Small min_samples → More clusters, sensitive to noise\n"
    "  * Large min_samples → Fewer clusters, robust to noise\n"
    "  * Too large → Many outliers\n\n"
    "For Innovation Data:\n"
    "  * Use small eps for distinct innovation types\n"
    "  * Use larger eps for broader innovation categories\n"
    "  * Adjust min_samples based on data density"
)

# Add text box with explanation
fig.text(0.02, 0.5, explanation_text, fontsize=9, va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

# Add tuning strategy box
strategy_text = (
    "Tuning Strategy:\n"
    "1. Start with k-distance plot\n"
    "2. Look for 'elbow' in plot\n"
    "3. Set eps at elbow point\n"
    "4. min_samples = 2*dimensions\n"
    "5. Validate with domain knowledge"
)

fig.text(0.85, 0.5, strategy_text, fontsize=9, va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))

# Add innovation context
innovation_text = (
    "Innovation Clustering Context:\n"
    "* Dense areas = Mainstream innovations\n"
    "* Sparse areas = Radical innovations\n"
    "* Outliers = Breakthrough ideas"
)

fig.text(0.5, 0.02, innovation_text, fontsize=9, ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

plt.tight_layout(rect=[0.15, 0.05, 0.82, 0.95])

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/dbscan_tuning.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/dbscan_tuning.png', 
           dpi=150, bbox_inches='tight')

print("DBSCAN parameter tuning visualization created successfully!")
print("Files saved:")
print("  - charts/dbscan_tuning.pdf")
print("  - charts/dbscan_tuning.png")