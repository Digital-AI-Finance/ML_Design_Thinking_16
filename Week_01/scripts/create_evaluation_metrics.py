#!/usr/bin/env python3
"""
Create Clustering Evaluation Metrics Comparison for Week 1
Shows different metrics for evaluating clustering quality
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.datasets import make_blobs
import matplotlib.patches as mpatches

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Generate sample datasets with different characteristics
datasets = []

# Dataset 1: Well-separated clusters
X1, y1 = make_blobs(n_samples=300, centers=3, n_features=2, 
                    cluster_std=0.5, random_state=42)
datasets.append(('Well-Separated Clusters', X1))

# Dataset 2: Overlapping clusters
X2, y2 = make_blobs(n_samples=300, centers=3, n_features=2, 
                    cluster_std=1.5, random_state=42)
datasets.append(('Overlapping Clusters', X2))

# Dataset 3: Different sized clusters
X3_1, _ = make_blobs(n_samples=100, centers=[[0, 0]], n_features=2, 
                     cluster_std=0.3, random_state=42)
X3_2, _ = make_blobs(n_samples=150, centers=[[3, 3]], n_features=2, 
                     cluster_std=0.8, random_state=42)
X3_3, _ = make_blobs(n_samples=50, centers=[[6, 0]], n_features=2, 
                     cluster_std=0.4, random_state=42)
X3 = np.vstack([X3_1, X3_2, X3_3])
datasets.append(('Different Sizes', X3))

# Calculate metrics for different K values
k_range = range(2, 8)
metrics_results = {name: {'silhouette': [], 'davies_bouldin': [], 'calinski': []} 
                  for name, _ in datasets}

for name, X in datasets:
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Silhouette Score (higher is better, range: -1 to 1)
        sil_score = silhouette_score(X, labels)
        metrics_results[name]['silhouette'].append(sil_score)
        
        # Davies-Bouldin Index (lower is better)
        db_score = davies_bouldin_score(X, labels)
        metrics_results[name]['davies_bouldin'].append(db_score)
        
        # Calinski-Harabasz Index (higher is better)
        ch_score = calinski_harabasz_score(X, labels)
        metrics_results[name]['calinski'].append(ch_score)

# Plot 1-3: Dataset visualizations with K=3
for i, (name, X) in enumerate(datasets):
    ax = axes[i]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                        s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
              c='red', s=200, marker='*', edgecolors='black', linewidth=2)
    
    # Add metrics as text
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    
    metrics_text = f'Silhouette: {sil:.3f}\nDavies-Bouldin: {db:.3f}\nCalinski-Harabasz: {ch:.0f}'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)

# Plot 4: Silhouette Score Comparison
ax = axes[3]
for name in metrics_results:
    ax.plot(k_range, metrics_results[name]['silhouette'], 
           marker='o', linewidth=2, label=name)
ax.set_xlabel('Number of Clusters (K)', fontsize=11)
ax.set_ylabel('Silhouette Score', fontsize=11)
ax.set_title('Silhouette Score\n(Higher is Better)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Plot 5: Davies-Bouldin Index Comparison
ax = axes[4]
for name in metrics_results:
    ax.plot(k_range, metrics_results[name]['davies_bouldin'], 
           marker='s', linewidth=2, label=name)
ax.set_xlabel('Number of Clusters (K)', fontsize=11)
ax.set_ylabel('Davies-Bouldin Index', fontsize=11)
ax.set_title('Davies-Bouldin Index\n(Lower is Better)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 6: Calinski-Harabasz Index Comparison
ax = axes[5]
for name in metrics_results:
    # Normalize for better visualization
    ch_scores = np.array(metrics_results[name]['calinski'])
    ch_scores_norm = ch_scores / ch_scores.max()
    ax.plot(k_range, ch_scores_norm, 
           marker='^', linewidth=2, label=name)
ax.set_xlabel('Number of Clusters (K)', fontsize=11)
ax.set_ylabel('Normalized Calinski-Harabasz', fontsize=11)
ax.set_title('Calinski-Harabasz Index\n(Higher is Better)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Overall title
fig.suptitle('Clustering Evaluation Metrics Comparison\nHow Different Metrics Behave on Various Data Patterns', 
            fontsize=16, fontweight='bold', y=1.02)

# Add metrics description box
metrics_info = """
Metrics Guide:
• Silhouette: Measures separation between clusters (-1 to 1, higher better)
• Davies-Bouldin: Ratio of within-cluster to between-cluster distance (lower better)
• Calinski-Harabasz: Ratio of between-cluster to within-cluster variance (higher better)
"""

fig.text(0.5, -0.05, metrics_info, ha='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/evaluation_metrics_comparison.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/evaluation_metrics_comparison.png', 
           dpi=150, bbox_inches='tight')

print("Evaluation metrics comparison created successfully!")
print("Files saved:")
print("  - charts/evaluation_metrics_comparison.pdf")
print("  - charts/evaluation_metrics_comparison.png")