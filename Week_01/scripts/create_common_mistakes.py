#!/usr/bin/env python3
"""
Create Common Mistakes and Troubleshooting Guide for Week 1
Shows typical clustering pitfalls and how to avoid them
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig = plt.figure(figsize=(16, 10))

# Define common mistakes with visual examples
# Mistake 1: Not scaling features
ax1 = plt.subplot(3, 4, 1)
X1_unscaled = np.random.randn(100, 2)
X1_unscaled[:, 0] *= 100  # Very different scale
X1_unscaled[:, 1] *= 1
kmeans1 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels1 = kmeans1.fit_predict(X1_unscaled)

for i in range(3):
    mask = labels1 == i
    ax1.scatter(X1_unscaled[mask, 0], X1_unscaled[mask, 1], 
               alpha=0.6, s=20)
ax1.set_title('MISTAKE: Unscaled Features', fontsize=9, color='red', fontweight='bold')
ax1.set_xlabel('Feature 1 (0-100)', fontsize=8)
ax1.set_ylabel('Feature 2 (0-1)', fontsize=8)

ax2 = plt.subplot(3, 4, 2)
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1_unscaled)
labels2 = kmeans1.fit_predict(X1_scaled)
for i in range(3):
    mask = labels2 == i
    ax2.scatter(X1_scaled[mask, 0], X1_scaled[mask, 1], 
               alpha=0.6, s=20)
ax2.set_title('CORRECT: Scaled Features', fontsize=9, color='green', fontweight='bold')
ax2.set_xlabel('Feature 1 (scaled)', fontsize=8)
ax2.set_ylabel('Feature 2 (scaled)', fontsize=8)

# Mistake 2: Wrong K selection
ax3 = plt.subplot(3, 4, 3)
X2 = np.vstack([np.random.randn(50, 2) + [0, 0],
                np.random.randn(50, 2) + [3, 3],
                np.random.randn(50, 2) + [6, 0]])
kmeans_wrong = KMeans(n_clusters=2, random_state=42, n_init=10)
labels3 = kmeans_wrong.fit_predict(X2)
for i in range(2):
    mask = labels3 == i
    ax3.scatter(X2[mask, 0], X2[mask, 1], alpha=0.6, s=20)
ax3.set_title('MISTAKE: K=2 (too few)', fontsize=9, color='red', fontweight='bold')
ax3.text(0.5, 0.02, '3 natural clusters forced into 2', 
        transform=ax3.transAxes, fontsize=7, ha='center')

ax4 = plt.subplot(3, 4, 4)
kmeans_right = KMeans(n_clusters=3, random_state=42, n_init=10)
labels4 = kmeans_right.fit_predict(X2)
for i in range(3):
    mask = labels4 == i
    ax4.scatter(X2[mask, 0], X2[mask, 1], alpha=0.6, s=20)
ax4.set_title('CORRECT: K=3 (optimal)', fontsize=9, color='green', fontweight='bold')
ax4.text(0.5, 0.02, 'Natural clusters identified', 
        transform=ax4.transAxes, fontsize=7, ha='center')

# Mistake 3: Ignoring outliers
ax5 = plt.subplot(3, 4, 5)
X3 = np.random.randn(100, 2)
X3 = np.vstack([X3, [[10, 10], [-10, -10], [10, -10]]])  # Add outliers
kmeans_outlier = KMeans(n_clusters=3, random_state=42, n_init=10)
labels5 = kmeans_outlier.fit_predict(X3)
for i in range(3):
    mask = labels5 == i
    ax5.scatter(X3[mask, 0], X3[mask, 1], alpha=0.6, s=20)
ax5.set_title('MISTAKE: Outliers distort', fontsize=9, color='red', fontweight='bold')
ax5.text(0.5, 0.02, 'Outliers create false clusters', 
        transform=ax5.transAxes, fontsize=7, ha='center')

ax6 = plt.subplot(3, 4, 6)
# Remove outliers (simple threshold)
mask_inliers = (np.abs(X3[:, 0]) < 5) & (np.abs(X3[:, 1]) < 5)
X3_clean = X3[mask_inliers]
labels6 = kmeans_outlier.fit_predict(X3_clean)
for i in range(3):
    mask = labels6 == i
    ax6.scatter(X3_clean[mask, 0], X3_clean[mask, 1], alpha=0.6, s=20)
ax6.set_title('CORRECT: Outliers removed', fontsize=9, color='green', fontweight='bold')
ax6.text(0.5, 0.02, 'Clean data clusters properly', 
        transform=ax6.transAxes, fontsize=7, ha='center')

# Mistake 4: Assuming spherical clusters
ax7 = plt.subplot(3, 4, 7)
# Create moon-shaped data
from sklearn.datasets import make_moons
X4, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
kmeans_moon = KMeans(n_clusters=2, random_state=42, n_init=10)
labels7 = kmeans_moon.fit_predict(X4)
for i in range(2):
    mask = labels7 == i
    ax7.scatter(X4[mask, 0], X4[mask, 1], alpha=0.6, s=20)
ax7.set_title('MISTAKE: K-means on moons', fontsize=9, color='red', fontweight='bold')
ax7.text(0.5, 0.02, 'K-means fails on non-spherical', 
        transform=ax7.transAxes, fontsize=7, ha='center')

ax8 = plt.subplot(3, 4, 8)
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels8 = dbscan.fit_predict(X4)
for i in set(labels8):
    if i == -1:
        continue
    mask = labels8 == i
    ax8.scatter(X4[mask, 0], X4[mask, 1], alpha=0.6, s=20)
ax8.set_title('CORRECT: DBSCAN for shapes', fontsize=9, color='green', fontweight='bold')
ax8.text(0.5, 0.02, 'DBSCAN handles any shape', 
        transform=ax8.transAxes, fontsize=7, ha='center')

# Troubleshooting Guide (bottom section)
ax9 = plt.subplot(3, 1, 3)
ax9.axis('off')

# Create troubleshooting table
problems = [
    ['Problem', 'Symptoms', 'Solution', 'Prevention'],
    ['Poor separation', 'Low silhouette score', 'Try different K or algorithm', 'Use elbow method'],
    ['Unstable results', 'Different runs = different clusters', 'Set random_state, increase n_init', 'Use deterministic init'],
    ['Slow performance', 'Takes too long to converge', 'Reduce features, subsample data', 'Use MiniBatch K-means'],
    ['Unbalanced clusters', 'One huge, others tiny', 'Check for outliers, try GMM', 'Inspect data distribution'],
    ['No convergence', 'Algorithm doesnt stop', 'Increase max_iter, check data', 'Normalize features first']
]

# Draw table
for i, row in enumerate(problems):
    for j, cell in enumerate(row):
        x_pos = 0.05 + j * 0.23
        y_pos = 0.85 - i * 0.15
        
        if i == 0:  # Header
            ax9.text(x_pos, y_pos, cell, fontsize=10, fontweight='bold',
                    transform=ax9.transAxes)
            # Draw header line
            ax9.plot([0.02, 0.98], [y_pos - 0.05, y_pos - 0.05], 
                    'k-', linewidth=1, transform=ax9.transAxes)
        else:
            # Wrap text for readability
            if j == 0:  # Problem column
                ax9.text(x_pos, y_pos, cell, fontsize=9, fontweight='bold',
                        transform=ax9.transAxes, color='darkred')
            elif j == 1:  # Symptoms
                ax9.text(x_pos, y_pos, cell, fontsize=8,
                        transform=ax9.transAxes, style='italic')
            elif j == 2:  # Solution
                ax9.text(x_pos, y_pos, cell, fontsize=8,
                        transform=ax9.transAxes, color='darkgreen')
            else:  # Prevention
                ax9.text(x_pos, y_pos, cell, fontsize=8,
                        transform=ax9.transAxes, color='darkblue')

# Add quick tips box
tips_text = (
    "GOLDEN RULES:\n"
    "1. Always scale your features\n"
    "2. Visualize before clustering\n"
    "3. Try multiple algorithms\n"
    "4. Validate with domain knowledge\n"
    "5. Check cluster stability"
)
ax9.text(0.02, 0.35, tips_text, fontsize=9, transform=ax9.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

# Add warning signs box
warnings_text = (
    "WARNING SIGNS:\n"
    "* Silhouette < 0.3\n"
    "* Clusters change each run\n"
    "* Single point clusters\n"
    "* All points in one cluster"
)
ax9.text(0.35, 0.35, warnings_text, fontsize=9, transform=ax9.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))

# Add success indicators
success_text = (
    "SUCCESS INDICATORS:\n"
    "* Silhouette > 0.5\n"
    "* Stable across runs\n"
    "* Balanced cluster sizes\n"
    "* Makes business sense"
)
ax9.text(0.68, 0.35, success_text, fontsize=9, transform=ax9.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

# Overall title
fig.suptitle('Common Clustering Mistakes & Troubleshooting Guide', 
            fontsize=16, fontweight='bold', y=0.98)
fig.text(0.5, 0.95, 'Learn from These Mistakes to Master Clustering',
        fontsize=11, ha='center', style='italic', color='gray')

# Add section labels
fig.text(0.02, 0.88, 'Visual Examples of Common Mistakes:', fontsize=11, fontweight='bold')
fig.text(0.02, 0.32, 'Troubleshooting Guide:', fontsize=11, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.94])

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/common_mistakes.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/common_mistakes.png', 
           dpi=150, bbox_inches='tight')

print("Common mistakes and troubleshooting guide created successfully!")
print("Files saved:")
print("  - charts/common_mistakes.pdf")
print("  - charts/common_mistakes.png")