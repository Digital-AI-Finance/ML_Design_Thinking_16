#!/usr/bin/env python3
"""
Create Innovation Pattern Visualization for Week 1 Part 3
Shows how clustering reveals innovation opportunities across different market segments
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.lines as mlines

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Define innovation segments with realistic patterns
n_points_per_segment = 200

# Segment 1: Early Adopters (high innovation, high risk tolerance)
early_adopters = np.random.multivariate_normal(
    [7, 8], [[0.8, 0.3], [0.3, 0.6]], n_points_per_segment
)

# Segment 2: Pragmatists (moderate innovation, moderate risk)
pragmatists = np.random.multivariate_normal(
    [5, 5], [[1.2, -0.2], [-0.2, 0.9]], n_points_per_segment
)

# Segment 3: Conservatives (low innovation, low risk)
conservatives = np.random.multivariate_normal(
    [3, 3], [[0.7, 0.1], [0.1, 0.8]], n_points_per_segment
)

# Segment 4: Skeptics (resistant to change)
skeptics = np.random.multivariate_normal(
    [2, 7], [[0.6, -0.3], [-0.3, 1.0]], n_points_per_segment
)

# Segment 5: Visionaries (high innovation, varied risk)
visionaries = np.random.multivariate_normal(
    [8, 3], [[0.9, 0.4], [0.4, 1.5]], n_points_per_segment
)

# Combine all segments
all_data = np.vstack([early_adopters, pragmatists, conservatives, skeptics, visionaries])

# Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(all_data)

# Color palette for segments
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
segment_names = ['Early Adopters', 'Pragmatists', 'Conservatives', 'Skeptics', 'Visionaries']

# Plot clustered data points
for i in range(5):
    mask = labels == i
    ax.scatter(all_data[mask, 0], all_data[mask, 1], 
              c=colors[i], s=30, alpha=0.6, edgecolors='none')

# Add cluster centers
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], 
          c='black', s=300, alpha=1.0, marker='*', 
          edgecolors='white', linewidth=2, zorder=5)

# Add innovation opportunity zones (highlighted areas between clusters)
# Zone 1: Between Early Adopters and Visionaries
zone1 = Circle(((centers[0, 0] + centers[4, 0])/2, 
               (centers[0, 1] + centers[4, 1])/2), 
               1.5, color='gold', alpha=0.2, zorder=1)
ax.add_patch(zone1)
ax.annotate('Innovation\nOpportunity #1', 
           xy=((centers[0, 0] + centers[4, 0])/2, 
               (centers[0, 1] + centers[4, 1])/2),
           xytext=(9, 6), fontsize=10, fontweight='bold',
           arrowprops=dict(arrowstyle='->', color='gold', lw=2))

# Zone 2: Between Pragmatists and Conservatives
zone2 = Circle(((centers[1, 0] + centers[2, 0])/2, 
               (centers[1, 1] + centers[2, 1])/2), 
               1.3, color='lightgreen', alpha=0.2, zorder=1)
ax.add_patch(zone2)
ax.annotate('Innovation\nOpportunity #2', 
           xy=((centers[1, 0] + centers[2, 0])/2, 
               (centers[1, 1] + centers[2, 1])/2),
           xytext=(3.5, 5.5), fontsize=10, fontweight='bold',
           arrowprops=dict(arrowstyle='->', color='green', lw=2))

# Zone 3: Underserved area
zone3 = Circle((6, 7), 1.2, color='lightcoral', alpha=0.2, zorder=1)
ax.add_patch(zone3)
ax.annotate('Underserved\nMarket Gap', 
           xy=(6, 7), xytext=(6.5, 9), fontsize=10, fontweight='bold',
           arrowprops=dict(arrowstyle='->', color='red', lw=2))

# Add segment labels near clusters
label_offsets = [(0.5, 0.5), (-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)]
for i, (center, name) in enumerate(zip(centers, segment_names)):
    ax.text(center[0] + label_offsets[i][0], 
           center[1] + label_offsets[i][1], 
           name, fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='white', alpha=0.8))

# Add grid lines for better readability
ax.grid(True, alpha=0.3, linestyle='--')

# Labels and title
ax.set_xlabel('Innovation Readiness Score', fontsize=14, fontweight='bold')
ax.set_ylabel('Risk Tolerance Index', fontsize=14, fontweight='bold')
ax.set_title('Innovation Pattern Discovery Through Clustering\nRevealing Hidden Market Opportunities', 
            fontsize=16, fontweight='bold', pad=20)

# Set axis limits
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Add legend
legend_elements = [mlines.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=colors[i], markersize=10, 
                                label=segment_names[i], alpha=0.7)
                  for i in range(5)]
legend_elements.append(mlines.Line2D([0], [0], marker='*', color='w', 
                                    markerfacecolor='black', markersize=15, 
                                    label='Cluster Center'))
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95)

# Add insights box
insights_text = (
    "Key Innovation Insights:\n"
    "• 5 distinct market segments identified\n"
    "• 3 innovation opportunities discovered\n"
    "• Underserved gap at high readiness/risk\n"
    "• Natural bridges between segments"
)
ax.text(0.5, 1.5, insights_text, fontsize=10,
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/innovation_patterns_visual.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/innovation_patterns_visual.png', 
           dpi=150, bbox_inches='tight')

print("Innovation patterns visualization created successfully!")
print("Files saved:")
print("  - charts/innovation_patterns_visual.pdf")
print("  - charts/innovation_patterns_visual.png")