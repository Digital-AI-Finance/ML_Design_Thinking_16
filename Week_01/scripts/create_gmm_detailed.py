#!/usr/bin/env python3
"""
Create Gaussian Mixture Models Detailed Explanation for Week 1
Shows GMM as soft clustering alternative to K-means
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy import linalg

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# Generate sample data with overlapping clusters
n_samples = 300
# Create three Gaussian clusters with some overlap
mean1 = [2, 2]
cov1 = [[1.5, 0.5], [0.5, 1]]
mean2 = [6, 3]
cov2 = [[1, -0.7], [-0.7, 1.5]]
mean3 = [4, 7]
cov3 = [[1.2, 0.3], [0.3, 0.8]]

X1 = np.random.multivariate_normal(mean1, cov1, n_samples // 3)
X2 = np.random.multivariate_normal(mean2, cov2, n_samples // 3)
X3 = np.random.multivariate_normal(mean3, cov3, n_samples // 3)
X = np.vstack([X1, X2, X3])

# Color scheme
colors = ['#3498db', '#e74c3c', '#2ecc71']

# Subplot 1: K-means Hard Clustering
ax1 = plt.subplot(2, 3, 1)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

for i in range(3):
    mask = kmeans_labels == i
    ax1.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=30, alpha=0.6, 
               edgecolors='white', linewidth=0.5)
    
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='black', s=200, marker='X', edgecolors='white', linewidth=2)
ax1.set_title('K-Means: Hard Assignment', fontsize=11, fontweight='bold')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.text(0.5, 0.02, 'Each point belongs to ONE cluster', 
        transform=ax1.transAxes, fontsize=8, ha='center', style='italic')

# Subplot 2: GMM Soft Clustering
ax2 = plt.subplot(2, 3, 2)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_probs = gmm.fit_predict(X)
gmm_probabilities = gmm.predict_proba(X)

# Plot points colored by their most likely cluster
for i in range(3):
    mask = gmm_probs == i
    ax2.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=30, alpha=0.6,
               edgecolors='white', linewidth=0.5)

# Add Gaussian ellipses
for i in range(3):
    mean = gmm.means_[i]
    covar = gmm.covariances_[i]
    v, w = linalg.eigh(covar)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    angle = np.arctan2(w[1, 0], w[0, 0])
    angle = 180. * angle / np.pi
    ell = Ellipse(mean, v[0], v[1], angle=angle, 
                 color=colors[i], alpha=0.2, linewidth=2)
    ax2.add_patch(ell)
    ax2.scatter(mean[0], mean[1], c='black', s=200, marker='X',
               edgecolors='white', linewidth=2)

ax2.set_title('GMM: Soft Assignment', fontsize=11, fontweight='bold')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.text(0.5, 0.02, 'Points have probabilities for ALL clusters',
        transform=ax2.transAxes, fontsize=8, ha='center', style='italic')

# Subplot 3: Probability Distribution
ax3 = plt.subplot(2, 3, 3)
# Show uncertainty for points near boundaries
uncertainty = 1 - np.max(gmm_probabilities, axis=1)
scatter = ax3.scatter(X[:, 0], X[:, 1], c=uncertainty, cmap='YlOrRd', 
                     s=30, alpha=0.7, edgecolors='gray', linewidth=0.5)
plt.colorbar(scatter, ax=ax3, label='Uncertainty')
ax3.set_title('Uncertainty Map', fontsize=11, fontweight='bold')
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')
ax3.text(0.5, 0.02, 'Red = High uncertainty (overlap regions)',
        transform=ax3.transAxes, fontsize=8, ha='center', style='italic')

# Subplot 4: Mixture Components
ax4 = plt.subplot(2, 3, 4)
# Create grid for probability density
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
XX, YY = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
XY = np.array([XX.ravel(), YY.ravel()]).T
Z = -gmm.score_samples(XY)
Z = Z.reshape(XX.shape)

contour = ax4.contour(XX, YY, Z, levels=10, linewidths=0.5, colors='gray')
ax4.scatter(X[:, 0], X[:, 1], c='gray', s=10, alpha=0.3)
for i in range(3):
    mean = gmm.means_[i]
    ax4.scatter(mean[0], mean[1], c=colors[i], s=200, marker='o',
               edgecolors='black', linewidth=2)
    ax4.text(mean[0], mean[1] - 0.5, f'μ{i+1}', fontsize=10, ha='center')

ax4.set_title('Gaussian Components', fontsize=11, fontweight='bold')
ax4.set_xlabel('Feature 1')
ax4.set_ylabel('Feature 2')
ax4.text(0.5, 0.02, 'Each component is a full Gaussian',
        transform=ax4.transAxes, fontsize=8, ha='center', style='italic')

# Subplot 5: GMM vs K-means Comparison
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')

comparison_text = """
GMM Advantages:
• Soft assignments (probabilities)
• Captures cluster shape (elliptical)
• Handles overlapping clusters
• Provides uncertainty estimates
• Models data generation process

K-means Advantages:
• Faster computation
• Simpler interpretation
• Less parameters
• More stable results
• Works well for spherical clusters

When to use GMM:
• Overlapping innovation categories
• Need probability scores
• Non-spherical clusters
• Uncertainty quantification needed
"""

ax5.text(0.1, 0.9, 'GMM vs K-means', fontsize=12, fontweight='bold',
        transform=ax5.transAxes)
ax5.text(0.1, 0.85, comparison_text, fontsize=9, transform=ax5.transAxes,
        verticalalignment='top')

# Subplot 6: Innovation Application
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Create example innovation probability table
innovation_examples = [
    ['Innovation', 'Tech', 'Service', 'Social'],
    ['AI Assistant', '0.85', '0.10', '0.05'],
    ['Sharing Platform', '0.30', '0.45', '0.25'],
    ['Green Energy', '0.60', '0.15', '0.25'],
    ['Digital Health', '0.40', '0.50', '0.10']
]

# Create table
cell_colors = []
for i, row in enumerate(innovation_examples):
    y_pos = 0.7 - i * 0.12
    for j, cell in enumerate(row):
        x_pos = 0.1 + j * 0.22
        
        if i == 0:  # Header
            ax6.text(x_pos, y_pos, cell, fontsize=9, fontweight='bold')
            color = 'lightgray'
        elif j == 0:  # Row labels
            ax6.text(x_pos, y_pos, cell, fontsize=8)
            color = 'lightgray'
        else:  # Probability values
            value = float(cell)
            ax6.text(x_pos, y_pos, cell, fontsize=8)
            # Color code by probability
            if value >= 0.5:
                color = 'lightgreen'
            elif value >= 0.3:
                color = 'lightyellow'
            else:
                color = 'lightcoral'
            
            # Draw background
            rect = plt.Rectangle((x_pos - 0.02, y_pos - 0.04), 0.2, 0.08,
                                facecolor=color, alpha=0.3)
            ax6.add_patch(rect)

ax6.text(0.1, 0.85, 'Innovation Category Probabilities', 
        fontsize=11, fontweight='bold', transform=ax6.transAxes)
ax6.text(0.1, 0.05, 'GMM provides probability of belonging to each category',
        fontsize=8, style='italic', transform=ax6.transAxes)

# Overall title
fig.suptitle('Gaussian Mixture Models (GMM): Soft Clustering for Innovation', 
            fontsize=16, fontweight='bold', y=0.98)
fig.text(0.5, 0.94, 'Beyond Hard Boundaries: Probabilistic Innovation Classification',
        fontsize=11, ha='center', style='italic', color='gray')

# Add mathematical formula
formula = r'$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$'
fig.text(0.5, 0.02, formula, fontsize=10, ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.3))

plt.tight_layout(rect=[0, 0.04, 1, 0.93])

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/gmm_detailed.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/gmm_detailed.png', 
           dpi=150, bbox_inches='tight')

print("GMM detailed explanation created successfully!")
print("Files saved:")
print("  - charts/gmm_detailed.pdf")
print("  - charts/gmm_detailed.png")