#!/usr/bin/env python3
"""
Create t-SNE vs PCA Comparison Visualization for Week 1
Shows different dimensionality reduction techniques for innovation data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(14, 10))

# Generate high-dimensional innovation dataset
# Simulate different innovation characteristics in high dimensions
n_samples = 500
n_features = 50  # High dimensional data

# Create different innovation clusters with distinct patterns
# Cluster 1: Traditional innovations (gaussian blob)
X1 = np.random.randn(150, n_features) * 0.5 + np.array([2] * n_features)

# Cluster 2: Disruptive innovations (shifted and stretched)
X2 = np.random.randn(100, n_features)
X2[:, :10] *= 3  # Strong signal in first 10 dimensions
X2[:, 10:] *= 0.3  # Weak signal in others
X2 += np.array([-2] * n_features)

# Cluster 3: Incremental innovations (tight cluster)
X3 = np.random.randn(100, n_features) * 0.3

# Cluster 4: Radical innovations (sparse, outliers)
X4 = np.random.randn(75, n_features) * 2
X4[:, 20:30] *= 4  # Very strong in specific dimensions

# Cluster 5: Hybrid innovations (non-linear pattern)
theta = np.linspace(0, 4 * np.pi, 75)
X5 = np.zeros((75, n_features))
X5[:, 0] = np.cos(theta) * 3
X5[:, 1] = np.sin(theta) * 3
X5[:, 2:10] = np.random.randn(75, 8) * 0.5
X5[:, 10:] = np.random.randn(75, n_features - 10) * 0.1

# Combine all data
X = np.vstack([X1, X2, X3, X4, X5])
y = np.array([0]*150 + [1]*100 + [2]*100 + [3]*75 + [4]*75)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Color scheme for innovation types
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
labels = ['Traditional', 'Disruptive', 'Incremental', 'Radical', 'Hybrid']

# Plot 1: Original High-Dimensional Data (first 2 dimensions only)
ax = axes[0, 0]
for i in range(5):
    mask = y == i
    ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1], c=colors[i], 
              label=labels[i], alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
ax.set_title('Original Data\n(First 2 of 50 dimensions)', fontsize=11, fontweight='bold')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.text(0.5, 0.02, 'Much information hidden', transform=ax.transAxes,
       fontsize=8, ha='center', style='italic', color='red')

# Plot 2: PCA Reduction
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

ax = axes[0, 1]
for i in range(5):
    mask = y == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], 
              label=labels[i], alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
ax.set_title('PCA Reduction\n(Linear projection)', fontsize=11, fontweight='bold')
ax.set_xlabel('PC 1 ({:.1f}% var)'.format(pca.explained_variance_ratio_[0] * 100))
ax.set_ylabel('PC 2 ({:.1f}% var)'.format(pca.explained_variance_ratio_[1] * 100))
ax.text(0.5, 0.02, 'Preserves global structure', transform=ax.transAxes,
       fontsize=8, ha='center', style='italic', color='green')

# Plot 3: t-SNE Reduction (perplexity=30)
tsne30 = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne30 = tsne30.fit_transform(X_scaled)

ax = axes[0, 2]
for i in range(5):
    mask = y == i
    ax.scatter(X_tsne30[mask, 0], X_tsne30[mask, 1], c=colors[i], 
              label=labels[i], alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
ax.set_title('t-SNE (perplexity=30)\n(Non-linear embedding)', fontsize=11, fontweight='bold')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.text(0.5, 0.02, 'Reveals local clusters', transform=ax.transAxes,
       fontsize=8, ha='center', style='italic', color='green')

# Plot 4: t-SNE with different perplexity (perplexity=5)
tsne5 = TSNE(n_components=2, perplexity=5, random_state=42)
X_tsne5 = tsne5.fit_transform(X_scaled)

ax = axes[1, 0]
for i in range(5):
    mask = y == i
    ax.scatter(X_tsne5[mask, 0], X_tsne5[mask, 1], c=colors[i], 
              label=labels[i], alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
ax.set_title('t-SNE (perplexity=5)\n(Focus on local)', fontsize=11, fontweight='bold')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.text(0.5, 0.02, 'Many small clusters', transform=ax.transAxes,
       fontsize=8, ha='center', style='italic', color='orange')

# Plot 5: t-SNE with different perplexity (perplexity=50)
tsne50 = TSNE(n_components=2, perplexity=50, random_state=42)
X_tsne50 = tsne50.fit_transform(X_scaled)

ax = axes[1, 1]
for i in range(5):
    mask = y == i
    ax.scatter(X_tsne50[mask, 0], X_tsne50[mask, 1], c=colors[i], 
              label=labels[i], alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
ax.set_title('t-SNE (perplexity=50)\n(Focus on global)', fontsize=11, fontweight='bold')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.text(0.5, 0.02, 'Broader structure', transform=ax.transAxes,
       fontsize=8, ha='center', style='italic', color='blue')

# Plot 6: Comparison Summary
ax = axes[1, 2]
ax.axis('off')

# Comparison table
comparison_data = [
    ['', 'PCA', 't-SNE'],
    ['Speed', 'Fast', 'Slow'],
    ['Scalability', 'Excellent', 'Limited'],
    ['Interpretation', 'Clear axes', 'No axes meaning'],
    ['Global structure', 'Preserved', 'May distort'],
    ['Local structure', 'May lose', 'Well preserved'],
    ['Parameter tuning', 'None', 'Perplexity critical'],
    ['Use for innovation', 'Overview', 'Find patterns']
]

# Create table
cell_colors = []
for i, row in enumerate(comparison_data):
    y_pos = 0.9 - i * 0.12
    for j, cell in enumerate(row):
        x_pos = 0.1 + j * 0.3
        
        # Header row
        if i == 0:
            ax.text(x_pos, y_pos, cell, fontsize=10, fontweight='bold', ha='left')
            color = 'lightgray'
        # First column
        elif j == 0:
            ax.text(x_pos, y_pos, cell, fontsize=9, fontweight='bold', ha='left')
            color = 'lightgray'
        else:
            ax.text(x_pos, y_pos, cell, fontsize=9, ha='left')
            # Color code advantages
            if 'Fast' in cell or 'Excellent' in cell or 'Clear' in cell or 'Preserved' in cell or 'None' in cell:
                color = 'lightgreen'
            elif 'Slow' in cell or 'Limited' in cell or 'distort' in cell or 'lose' in cell:
                color = 'lightcoral'
            else:
                color = 'lightyellow'
        
        # Draw cell background
        if i > 0 and j > 0:
            rect = plt.Rectangle((x_pos - 0.02, y_pos - 0.05), 0.28, 0.1, 
                                facecolor=color, alpha=0.3)
            ax.add_patch(rect)

ax.set_title('Method Comparison', fontsize=11, fontweight='bold', y=1.0)

# Add recommendation
recommendation = (
    "Recommendation for Innovation Analysis:\n"
    "1. Start with PCA for quick overview\n"
    "2. Use t-SNE to find hidden patterns\n"
    "3. Try multiple perplexity values\n"
    "4. Validate findings with domain knowledge"
)
ax.text(0.1, 0.15, recommendation, fontsize=8,
       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))

# Overall title and subtitle
fig.suptitle('Dimensionality Reduction: PCA vs t-SNE for Innovation Data', 
            fontsize=16, fontweight='bold', y=1.02)
fig.text(0.5, 0.98, 'Revealing Hidden Patterns in High-Dimensional Innovation Space', 
        fontsize=11, ha='center', style='italic', color='gray')

# Add legend
handles = [mpatches.Patch(color=colors[i], label=labels[i], alpha=0.6) for i in range(5)]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
          bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/tsne_pca_comparison.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/tsne_pca_comparison.png', 
           dpi=150, bbox_inches='tight')

print("t-SNE vs PCA comparison created successfully!")
print("Files saved:")
print("  - charts/tsne_pca_comparison.pdf")
print("  - charts/tsne_pca_comparison.png")