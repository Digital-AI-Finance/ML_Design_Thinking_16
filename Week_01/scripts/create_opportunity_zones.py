#!/usr/bin/env python3
"""
Create Opportunity Zones Visualization for Week 1 Part 3
Shows white spaces and innovation opportunities between existing market segments
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon, Circle
import matplotlib.patches as mpatches

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Generate market data with clear gaps (opportunity zones)
n_points = 150

# Existing market clusters (where competition is high)
cluster1 = np.random.multivariate_normal([2, 2], [[0.5, 0.1], [0.1, 0.5]], n_points)
cluster2 = np.random.multivariate_normal([8, 2], [[0.6, -0.1], [-0.1, 0.5]], n_points)
cluster3 = np.random.multivariate_normal([2, 8], [[0.5, 0.15], [0.15, 0.6]], n_points)
cluster4 = np.random.multivariate_normal([8, 8], [[0.7, 0.1], [0.1, 0.5]], n_points)

# Combine clusters
market_data = np.vstack([cluster1, cluster2, cluster3, cluster4])

# Apply K-means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(market_data)

# Colors for existing markets
market_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# LEFT PLOT: Current Market Landscape
ax1.set_title('Current Market Landscape\n(Competition Density)', fontsize=14, fontweight='bold', pad=15)

# Plot existing markets
for i in range(4):
    mask = labels == i
    ax1.scatter(market_data[mask, 0], market_data[mask, 1], 
               c=market_colors[i], s=40, alpha=0.6, edgecolors='white', linewidth=0.5)

# Add density contours to show competition intensity
from scipy.stats import gaussian_kde

# Create density map
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
positions = np.vstack([X.ravel(), Y.ravel()])

# Calculate kernel density
kernel = gaussian_kde(market_data.T)
Z = np.reshape(kernel(positions).T, X.shape)

# Plot contours
contours = ax1.contour(X, Y, Z, levels=5, colors='gray', alpha=0.3, linewidths=1)
ax1.clabel(contours, inline=True, fontsize=8, fmt='%.0f')

# Mark cluster centers
centers = kmeans.cluster_centers_
ax1.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='x', linewidth=3)

# Add market labels
market_names = ['Traditional\nSolutions', 'Premium\nSegment', 'Budget\nOptions', 'Enterprise\nTools']
for i, (center, name) in enumerate(zip(centers, market_names)):
    ax1.annotate(name, xy=center, xytext=(center[0], center[1]-0.7),
                fontsize=9, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax1.set_xlabel('Feature Complexity', fontsize=12, fontweight='bold')
ax1.set_ylabel('Price Point', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.grid(True, alpha=0.2)

# RIGHT PLOT: Innovation Opportunity Zones
ax2.set_title('Innovation Opportunity Zones\n(White Spaces for Disruption)', fontsize=14, fontweight='bold', pad=15)

# Plot existing markets (lighter)
for i in range(4):
    mask = labels == i
    ax2.scatter(market_data[mask, 0], market_data[mask, 1], 
               c=market_colors[i], s=20, alpha=0.2, edgecolors='none')

# Highlight opportunity zones (areas with low density)
# Zone 1: Center area (5, 5)
zone1 = Circle((5, 5), 1.8, color='gold', alpha=0.4, linewidth=3, 
              linestyle='--', fill=True, edgecolor='darkorange')
ax2.add_patch(zone1)
ax2.annotate('OPPORTUNITY #1\nIntegrated Platform\n(Underserved)', 
            xy=(5, 5), xytext=(5, 6.5),
            fontsize=10, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='gold', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=2))

# Zone 2: Mid-lower area (5, 2.5)
zone2 = Circle((5, 2.5), 1.3, color='lightgreen', alpha=0.4, linewidth=3,
              linestyle='--', fill=True, edgecolor='green')
ax2.add_patch(zone2)
ax2.annotate('OPPORTUNITY #2\nFreemium Model\n(Gap in Market)', 
            xy=(5, 2.5), xytext=(5, 0.5),
            fontsize=10, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

# Zone 3: Upper-mid area (5, 7.5)
zone3 = Circle((5, 7.5), 1.3, color='lightcoral', alpha=0.4, linewidth=3,
              linestyle='--', fill=True, edgecolor='red')
ax2.add_patch(zone3)
ax2.annotate('OPPORTUNITY #3\nLuxury Hybrid\n(Blue Ocean)', 
            xy=(5, 7.5), xytext=(5, 9.5),
            fontsize=10, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

# Zone 4: Side areas
zone4 = Circle((0.5, 5), 1.0, color='lightblue', alpha=0.3, linewidth=2,
              linestyle='--', fill=True, edgecolor='blue')
ax2.add_patch(zone4)

zone5 = Circle((9.5, 5), 1.0, color='lightblue', alpha=0.3, linewidth=2,
              linestyle='--', fill=True, edgecolor='blue')
ax2.add_patch(zone5)

# Add market pressure arrows showing movement potential
arrow_props = dict(arrowstyle='->', lw=2, alpha=0.5, color='purple')
ax2.annotate('', xy=(5, 5), xytext=(2, 2), arrowprops=arrow_props)
ax2.annotate('', xy=(5, 5), xytext=(8, 2), arrowprops=arrow_props)
ax2.annotate('', xy=(5, 5), xytext=(2, 8), arrowprops=arrow_props)
ax2.annotate('', xy=(5, 5), xytext=(8, 8), arrowprops=arrow_props)

# Add strategic insight text
ax2.text(1, 9.5, 'Market Forces →', fontsize=9, color='purple', alpha=0.7)

ax2.set_xlabel('Feature Complexity', fontsize=12, fontweight='bold')
ax2.set_ylabel('Price Point', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.grid(True, alpha=0.2)

# Add legend for opportunity zones
legend_elements = [
    mpatches.Patch(color='gold', alpha=0.4, label='High Potential Zone'),
    mpatches.Patch(color='lightgreen', alpha=0.4, label='Growth Opportunity'),
    mpatches.Patch(color='lightcoral', alpha=0.4, label='Blue Ocean Space'),
    mpatches.Patch(color='lightblue', alpha=0.3, label='Niche Markets')
]
ax2.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)

# Add overall insights
fig.text(0.5, 0.02, 
        'Innovation Strategy: Target white spaces between existing market clusters where competition is low but demand potential is high',
        ha='center', fontsize=11, fontweight='bold', style='italic')

plt.tight_layout()

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/opportunity_zones.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/opportunity_zones.png', 
           dpi=150, bbox_inches='tight')

print("Opportunity zones visualization created successfully!")
print("Files saved:")
print("  - charts/opportunity_zones.pdf")
print("  - charts/opportunity_zones.png")