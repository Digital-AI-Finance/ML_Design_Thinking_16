#!/usr/bin/env python3
"""
Create Innovation Ecosystem Mapping Visualization for Week 1
Shows how clustering reveals relationships in innovation networks
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyArrowPatch
import networkx as nx

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Left panel: Raw ecosystem data
ax1.set_title('Before Clustering: Complex Innovation Ecosystem', fontsize=12, fontweight='bold')

# Generate ecosystem entities
n_entities = 150
entity_types = {
    'Startups': {'count': 50, 'color': '#3498db', 'marker': 'o', 'size': 30},
    'Corporations': {'count': 20, 'color': '#e74c3c', 'marker': 's', 'size': 100},
    'Universities': {'count': 15, 'color': '#2ecc71', 'marker': '^', 'size': 80},
    'VCs': {'count': 25, 'color': '#f39c12', 'marker': 'D', 'size': 60},
    'Accelerators': {'count': 10, 'color': '#9b59b6', 'marker': 'p', 'size': 70},
    'Government': {'count': 8, 'color': '#1abc9c', 'marker': 'H', 'size': 90},
    'Research Labs': {'count': 12, 'color': '#34495e', 'marker': '*', 'size': 75},
    'Individual Innovators': {'count': 10, 'color': '#95a5a6', 'marker': '.', 'size': 20}
}

# Generate positions for raw data (somewhat random but with some structure)
all_positions = []
all_types = []
all_colors = []
all_markers = []
all_sizes = []

for entity_type, info in entity_types.items():
    # Create clusters of entities with some overlap
    if entity_type == 'Startups':
        centers = [(3, 3), (7, 7), (5, 5)]
    elif entity_type == 'Corporations':
        centers = [(8, 8)]
    elif entity_type == 'Universities':
        centers = [(2, 8)]
    elif entity_type == 'VCs':
        centers = [(7, 3)]
    else:
        centers = [(np.random.uniform(1, 9), np.random.uniform(1, 9))]
    
    for i in range(info['count']):
        center = centers[i % len(centers)]
        x = center[0] + np.random.normal(0, 1.5)
        y = center[1] + np.random.normal(0, 1.5)
        all_positions.append([x, y])
        all_types.append(entity_type)
        all_colors.append(info['color'])
        all_markers.append(info['marker'])
        all_sizes.append(info['size'])

positions = np.array(all_positions)

# Plot raw ecosystem
for i, (pos, color, marker, size) in enumerate(zip(positions, all_colors, all_markers, all_sizes)):
    ax1.scatter(pos[0], pos[1], c=color, marker=marker, s=size, alpha=0.6, edgecolors='white', linewidth=0.5)

# Add some random connections to show complexity
n_connections = 50
for _ in range(n_connections):
    i, j = np.random.choice(len(positions), 2, replace=False)
    ax1.plot([positions[i, 0], positions[j, 0]], 
            [positions[i, 1], positions[j, 1]], 
            'gray', alpha=0.1, linewidth=0.5)

ax1.text(5, 0.5, 'Complex relationships, unclear patterns', 
        ha='center', fontsize=9, style='italic')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

# Right panel: After clustering
ax2.set_title('After Clustering: Innovation Clusters Revealed', fontsize=12, fontweight='bold')

# Apply K-means clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(positions)

# Define cluster meanings
cluster_meanings = {
    0: {'name': 'Tech Innovation Hub', 'color': '#3498db'},
    1: {'name': 'Corporate R&D', 'color': '#e74c3c'},
    2: {'name': 'Academic Research', 'color': '#2ecc71'},
    3: {'name': 'Startup Ecosystem', 'color': '#f39c12'},
    4: {'name': 'Funding Network', 'color': '#9b59b6'}
}

# Plot clustered ecosystem
for cluster_id in range(n_clusters):
    mask = cluster_labels == cluster_id
    cluster_positions = positions[mask]
    
    # Draw cluster boundary
    if len(cluster_positions) > 2:
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(cluster_positions)
            for simplex in hull.simplices:
                ax2.plot(cluster_positions[simplex, 0], 
                        cluster_positions[simplex, 1], 
                        color=cluster_meanings[cluster_id]['color'], 
                        alpha=0.3, linewidth=2)
        except:
            pass
    
    # Plot entities in cluster
    for i, is_in_cluster in enumerate(mask):
        if is_in_cluster:
            ax2.scatter(positions[i, 0], positions[i, 1], 
                       c=cluster_meanings[cluster_id]['color'],
                       marker=all_markers[i], s=all_sizes[i], 
                       alpha=0.7, edgecolors='white', linewidth=0.5)

# Add cluster centers and labels
centers = kmeans.cluster_centers_
for i, center in enumerate(centers):
    ax2.scatter(center[0], center[1], c='black', marker='X', s=200, 
               edgecolors='white', linewidth=2, zorder=5)
    ax2.text(center[0], center[1] - 0.5, cluster_meanings[i]['name'], 
            fontsize=8, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', 
                     facecolor=cluster_meanings[i]['color'], 
                     alpha=0.3))

# Draw connections between cluster centers
for i in range(n_clusters):
    for j in range(i+1, n_clusters):
        distance = np.linalg.norm(centers[i] - centers[j])
        if distance < 5:  # Only show close connections
            ax2.plot([centers[i, 0], centers[j, 0]], 
                    [centers[i, 1], centers[j, 1]], 
                    'gray', alpha=0.3, linewidth=2, linestyle='--')

ax2.text(5, 0.5, 'Clear innovation clusters and relationships identified', 
        ha='center', fontsize=9, style='italic')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Add legend for entity types
legend_items = []
for entity_type, info in list(entity_types.items())[:5]:  # Show first 5 types
    legend_items.append(plt.Line2D([0], [0], marker=info['marker'], color='w', 
                                  markerfacecolor=info['color'], markersize=8, 
                                  label=entity_type, alpha=0.7))
ax1.legend(handles=legend_items, loc='upper left', fontsize=8)

# Add insights panel
insights_text = (
    "Clustering Benefits:\n"
    "* Identify key players\n"
    "* Find collaboration opportunities\n"
    "* Detect innovation gaps\n"
    "* Map knowledge flows\n"
    "* Predict emerging trends"
)
fig.text(0.5, 0.02, insights_text, ha='center', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

# Overall title
fig.suptitle('Innovation Ecosystem Mapping with Clustering', 
            fontsize=16, fontweight='bold', y=0.98)

# Add metrics comparison
metrics_before = "Metrics (Before):\nConnections: Random\nDensity: Unknown\nCentrality: Unclear"
metrics_after = "Metrics (After):\nClusters: 5 distinct\nDensity: Measured\nCentrality: Identified"

ax1.text(0.5, 9.5, metrics_before, fontsize=8, 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))
ax2.text(0.5, 9.5, metrics_after, fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/innovation_ecosystem.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/innovation_ecosystem.png', 
           dpi=150, bbox_inches='tight')

print("Innovation ecosystem mapping created successfully!")
print("Files saved:")
print("  - charts/innovation_ecosystem.pdf")
print("  - charts/innovation_ecosystem.png")