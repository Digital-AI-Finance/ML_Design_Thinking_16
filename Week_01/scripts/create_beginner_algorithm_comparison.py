#!/usr/bin/env python3
"""
Create Beginner-Friendly Algorithm Comparison
Visual explanations without technical jargon
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Standard color palette
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e', 
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'yellow': '#f39c12',
    'dark': '#3c3c3c',
    'light': '#f0f0f0'
}

# Create figure
fig = plt.figure(figsize=(16, 12))

# Generate simple dataset
np.random.seed(42)
# Create 3 clear groups
group1 = np.random.multivariate_normal([2, 2], [[0.5, 0.1], [0.1, 0.5]], 50)
group2 = np.random.multivariate_normal([6, 2], [[0.5, -0.2], [-0.2, 0.5]], 50)
group3 = np.random.multivariate_normal([4, 6], [[0.6, 0.1], [0.1, 0.6]], 50)
X = np.vstack([group1, group2, group3])

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means explanation
ax1 = plt.subplot(3, 4, 1)
ax1.set_title('K-Means: How it Works', fontsize=11, fontweight='bold')
ax1.axis('off')

kmeans_steps = [
    '1. Pick 3 center points',
    '2. Assign each point to nearest center',
    '3. Move centers to middle of groups',
    '4. Repeat until stable'
]

y_pos = 0.9
for step in kmeans_steps:
    ax1.text(0.05, y_pos, step, fontsize=9, transform=ax1.transAxes)
    y_pos -= 0.2

ax1.text(0.5, 0.1, 'Like organizing by neighborhoods', 
         fontsize=9, ha='center', transform=ax1.transAxes,
         bbox=dict(boxstyle='round', facecolor=colors['mlblue'], alpha=0.2))

# K-Means result
ax2 = plt.subplot(3, 4, 2)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)
scatter = ax2.scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis', s=30, alpha=0.7)
ax2.scatter(scaler.inverse_transform(kmeans.cluster_centers_)[:, 0],
           scaler.inverse_transform(kmeans.cluster_centers_)[:, 1],
           marker='*', s=200, c='red', edgecolor='black', linewidth=1)
ax2.set_title('K-Means Result', fontsize=11, fontweight='bold')
ax2.set_xlabel('Feature 1', fontsize=9)
ax2.set_ylabel('Feature 2', fontsize=9)
ax2.text(0.5, -0.15, 'Stars = Centers', fontsize=8, ha='center', transform=ax2.transAxes)

# K-Means good/bad
ax3 = plt.subplot(3, 4, 3)
ax3.set_title('When to Use K-Means', fontsize=11, fontweight='bold')
ax3.axis('off')

good_bad = [
    ('Good for:', colors['mlgreen']),
    ('• Round groups', colors['mlgreen']),
    ('• Similar sizes', colors['mlgreen']),
    ('• Fast results', colors['mlgreen']),
    ('', 'black'),
    ('Not good for:', colors['mlred']),
    ('• Weird shapes', colors['mlred']),
    ('• Different sizes', colors['mlred']),
]

y_pos = 0.9
for text, color in good_bad:
    ax3.text(0.05, y_pos, text, fontsize=9, color=color, transform=ax3.transAxes,
            fontweight='bold' if 'for:' in text else 'normal')
    y_pos -= 0.11

# K-Means speed indicator
ax4 = plt.subplot(3, 4, 4)
ax4.set_title('Speed & Difficulty', fontsize=11, fontweight='bold')
ax4.axis('off')

# Speed bar
ax4.text(0.1, 0.7, 'Speed:', fontsize=10, transform=ax4.transAxes)
speed_bar = patches.Rectangle((0.3, 0.68), 0.6, 0.08, 
                              facecolor=colors['mlgreen'], edgecolor='black')
ax4.add_patch(speed_bar)
ax4.text(0.6, 0.72, 'FAST', fontsize=9, ha='center', transform=ax4.transAxes, color='white')

# Difficulty bar
ax4.text(0.1, 0.5, 'Difficulty:', fontsize=10, transform=ax4.transAxes)
diff_bar = patches.Rectangle((0.3, 0.48), 0.3, 0.08,
                             facecolor=colors['mlgreen'], edgecolor='black')
ax4.add_patch(diff_bar)
ax4.text(0.45, 0.52, 'EASY', fontsize=9, ha='center', transform=ax4.transAxes, color='white')

# Settings needed
ax4.text(0.1, 0.3, 'You need to know:', fontsize=10, transform=ax4.transAxes, fontweight='bold')
ax4.text(0.1, 0.2, '• How many groups (K)', fontsize=9, transform=ax4.transAxes)

# DBSCAN explanation
ax5 = plt.subplot(3, 4, 5)
ax5.set_title('DBSCAN: How it Works', fontsize=11, fontweight='bold')
ax5.axis('off')

dbscan_steps = [
    '1. Look at each point',
    '2. Count neighbors nearby',
    '3. If enough neighbors → core point',
    '4. Connect core points → groups'
]

y_pos = 0.9
for step in dbscan_steps:
    ax5.text(0.05, y_pos, step, fontsize=9, transform=ax5.transAxes)
    y_pos -= 0.2

ax5.text(0.5, 0.1, 'Like finding crowds at a party', 
         fontsize=9, ha='center', transform=ax5.transAxes,
         bbox=dict(boxstyle='round', facecolor=colors['mlorange'], alpha=0.2))

# DBSCAN result
ax6 = plt.subplot(3, 4, 6)
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)
scatter = ax6.scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='viridis', s=30, alpha=0.7)
ax6.set_title('DBSCAN Result', fontsize=11, fontweight='bold')
ax6.set_xlabel('Feature 1', fontsize=9)
ax6.set_ylabel('Feature 2', fontsize=9)
# Mark outliers
outliers = labels_dbscan == -1
if outliers.any():
    ax6.scatter(X[outliers, 0], X[outliers, 1], c='red', marker='x', s=50)
    ax6.text(0.5, -0.15, 'X = Outliers', fontsize=8, ha='center', transform=ax6.transAxes)

# DBSCAN good/bad
ax7 = plt.subplot(3, 4, 7)
ax7.set_title('When to Use DBSCAN', fontsize=11, fontweight='bold')
ax7.axis('off')

good_bad = [
    ('Good for:', colors['mlgreen']),
    ('• Any shape groups', colors['mlgreen']),
    ('• Finding outliers', colors['mlgreen']),
    ('• Unknown group count', colors['mlgreen']),
    ('', 'black'),
    ('Not good for:', colors['mlred']),
    ('• Different densities', colors['mlred']),
    ('• Need exact K groups', colors['mlred']),
]

y_pos = 0.9
for text, color in good_bad:
    ax7.text(0.05, y_pos, text, fontsize=9, color=color, transform=ax7.transAxes,
            fontweight='bold' if 'for:' in text else 'normal')
    y_pos -= 0.11

# DBSCAN speed indicator
ax8 = plt.subplot(3, 4, 8)
ax8.set_title('Speed & Difficulty', fontsize=11, fontweight='bold')
ax8.axis('off')

# Speed bar
ax8.text(0.1, 0.7, 'Speed:', fontsize=10, transform=ax8.transAxes)
speed_bar = patches.Rectangle((0.3, 0.68), 0.4, 0.08,
                              facecolor=colors['mlorange'], edgecolor='black')
ax8.add_patch(speed_bar)
ax8.text(0.5, 0.72, 'MEDIUM', fontsize=9, ha='center', transform=ax8.transAxes, color='white')

# Difficulty bar
ax8.text(0.1, 0.5, 'Difficulty:', fontsize=10, transform=ax8.transAxes)
diff_bar = patches.Rectangle((0.3, 0.48), 0.5, 0.08,
                             facecolor=colors['mlorange'], edgecolor='black')
ax8.add_patch(diff_bar)
ax8.text(0.55, 0.52, 'MEDIUM', fontsize=9, ha='center', transform=ax8.transAxes, color='white')

# Settings needed
ax8.text(0.1, 0.3, 'You need to know:', fontsize=10, transform=ax8.transAxes, fontweight='bold')
ax8.text(0.1, 0.2, '• Distance (eps)', fontsize=9, transform=ax8.transAxes)
ax8.text(0.1, 0.1, '• Min neighbors', fontsize=9, transform=ax8.transAxes)

# Hierarchical explanation
ax9 = plt.subplot(3, 4, 9)
ax9.set_title('Hierarchical: How it Works', fontsize=11, fontweight='bold')
ax9.axis('off')

hier_steps = [
    '1. Start: each point alone',
    '2. Find closest pair',
    '3. Merge into group',
    '4. Repeat until all connected'
]

y_pos = 0.9
for step in hier_steps:
    ax9.text(0.05, y_pos, step, fontsize=9, transform=ax9.transAxes)
    y_pos -= 0.2

ax9.text(0.5, 0.1, 'Like a family tree', 
         fontsize=9, ha='center', transform=ax9.transAxes,
         bbox=dict(boxstyle='round', facecolor=colors['mlgreen'], alpha=0.2))

# Hierarchical result
ax10 = plt.subplot(3, 4, 10)
hierarchical = AgglomerativeClustering(n_clusters=3)
labels_hier = hierarchical.fit_predict(X_scaled)
scatter = ax10.scatter(X[:, 0], X[:, 1], c=labels_hier, cmap='viridis', s=30, alpha=0.7)
ax10.set_title('Hierarchical Result', fontsize=11, fontweight='bold')
ax10.set_xlabel('Feature 1', fontsize=9)
ax10.set_ylabel('Feature 2', fontsize=9)
ax10.text(0.5, -0.15, 'Tree structure', fontsize=8, ha='center', transform=ax10.transAxes)

# Hierarchical good/bad
ax11 = plt.subplot(3, 4, 11)
ax11.set_title('When to Use Hierarchical', fontsize=11, fontweight='bold')
ax11.axis('off')

good_bad = [
    ('Good for:', colors['mlgreen']),
    ('• See all groupings', colors['mlgreen']),
    ('• Small datasets', colors['mlgreen']),
    ('• Need tree view', colors['mlgreen']),
    ('', 'black'),
    ('Not good for:', colors['mlred']),
    ('• Large datasets', colors['mlred']),
    ('• Need speed', colors['mlred']),
]

y_pos = 0.9
for text, color in good_bad:
    ax11.text(0.05, y_pos, text, fontsize=9, color=color, transform=ax11.transAxes,
            fontweight='bold' if 'for:' in text else 'normal')
    y_pos -= 0.11

# Hierarchical speed indicator
ax12 = plt.subplot(3, 4, 12)
ax12.set_title('Speed & Difficulty', fontsize=11, fontweight='bold')
ax12.axis('off')

# Speed bar
ax12.text(0.1, 0.7, 'Speed:', fontsize=10, transform=ax12.transAxes)
speed_bar = patches.Rectangle((0.3, 0.68), 0.2, 0.08,
                              facecolor=colors['mlred'], edgecolor='black')
ax12.add_patch(speed_bar)
ax12.text(0.4, 0.72, 'SLOW', fontsize=9, ha='center', transform=ax12.transAxes, color='white')

# Difficulty bar
ax12.text(0.1, 0.5, 'Difficulty:', fontsize=10, transform=ax12.transAxes)
diff_bar = patches.Rectangle((0.3, 0.48), 0.3, 0.08,
                             facecolor=colors['mlgreen'], edgecolor='black')
ax12.add_patch(diff_bar)
ax12.text(0.45, 0.52, 'EASY', fontsize=9, ha='center', transform=ax12.transAxes, color='white')

# Settings needed
ax12.text(0.1, 0.3, 'You need to know:', fontsize=10, transform=ax12.transAxes, fontweight='bold')
ax12.text(0.1, 0.2, '• How many groups', fontsize=9, transform=ax12.transAxes)
ax12.text(0.1, 0.1, '  OR cut height', fontsize=9, transform=ax12.transAxes)

# Main title
fig.suptitle('Clustering Algorithms: Simple Comparison', 
            fontsize=14, fontweight='bold', y=0.98)
fig.text(0.5, 0.96, 'Three Different Ways to Find Groups', 
        fontsize=11, ha='center', style='italic', color='gray')

# Decision helper at bottom
decision_text = (
    'Quick Decision Guide: '
    'Need speed? → K-Means | '
    'Weird shapes? → DBSCAN | '
    'Want to see structure? → Hierarchical'
)
fig.text(0.5, 0.01, decision_text, fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout(rect=[0, 0.02, 1, 0.95])

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/beginner_algorithm_comparison.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/beginner_algorithm_comparison.png', 
           dpi=150, bbox_inches='tight')

print("Beginner algorithm comparison created successfully!")
print("Files saved:")
print("  - charts/beginner_algorithm_comparison.pdf")
print("  - charts/beginner_algorithm_comparison.png")