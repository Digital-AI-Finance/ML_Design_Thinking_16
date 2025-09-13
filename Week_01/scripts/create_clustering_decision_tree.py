#!/usr/bin/env python3
"""
Create Clustering Algorithm Decision Tree for Week 1
Helps students choose the right clustering algorithm based on their data and requirements
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Define decision tree nodes
nodes = {
    'start': {
        'pos': (7, 9),
        'text': 'START\nChoose Clustering\nAlgorithm',
        'color': '#2c3e50',
        'type': 'decision'
    },
    'know_k': {
        'pos': (7, 7.5),
        'text': 'Do you know\nnumber of clusters?',
        'color': '#3498db',
        'type': 'decision'
    },
    'shape_matters': {
        'pos': (4, 6),
        'text': 'Are clusters\nspherical?',
        'color': '#3498db',
        'type': 'decision'
    },
    'outliers': {
        'pos': (10, 6),
        'text': 'Need to detect\noutliers?',
        'color': '#3498db',
        'type': 'decision'
    },
    'hierarchy': {
        'pos': (7, 4.5),
        'text': 'Need multiple\ngranularities?',
        'color': '#3498db',
        'type': 'decision'
    },
    'kmeans': {
        'pos': (2, 4),
        'text': 'K-MEANS\n✓ Fast\n✓ Simple\n✓ Scalable',
        'color': '#27ae60',
        'type': 'algorithm'
    },
    'gmm': {
        'pos': (5.5, 4),
        'text': 'GMM\n✓ Soft clusters\n✓ Probability\n✓ Overlap',
        'color': '#9b59b6',
        'type': 'algorithm'
    },
    'dbscan': {
        'pos': (10, 4),
        'text': 'DBSCAN\n✓ Any shape\n✓ Outliers\n✓ No K needed',
        'color': '#e74c3c',
        'type': 'algorithm'
    },
    'hierarchical': {
        'pos': (7, 2.5),
        'text': 'HIERARCHICAL\n✓ Dendrogram\n✓ Multi-level\n✓ Interpretable',
        'color': '#f39c12',
        'type': 'algorithm'
    },
    'meanshift': {
        'pos': (12, 4),
        'text': 'MEAN SHIFT\n✓ No params\n✓ Mode seeking\n✓ Robust',
        'color': '#16a085',
        'type': 'algorithm'
    }
}

# Draw nodes
for node_id, node in nodes.items():
    if node['type'] == 'decision':
        # Diamond shape for decision nodes
        diamond = FancyBboxPatch(
            (node['pos'][0] - 1.2, node['pos'][1] - 0.4),
            2.4, 0.8,
            boxstyle="round,pad=0.1",
            facecolor='white',
            edgecolor=node['color'],
            linewidth=2
        )
        ax.add_patch(diamond)
        ax.text(node['pos'][0], node['pos'][1], node['text'],
               fontsize=10, ha='center', va='center',
               fontweight='bold', color=node['color'])
    else:
        # Rectangle for algorithm nodes
        rect = FancyBboxPatch(
            (node['pos'][0] - 1.0, node['pos'][1] - 0.5),
            2.0, 1.0,
            boxstyle="round,pad=0.05",
            facecolor=node['color'],
            edgecolor='white',
            alpha=0.9,
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(node['pos'][0], node['pos'][1], node['text'],
               fontsize=9, ha='center', va='center',
               fontweight='bold', color='white')

# Define decision paths with labels
paths = [
    ('start', 'know_k', ''),
    ('know_k', 'shape_matters', 'YES'),
    ('know_k', 'outliers', 'NO'),
    ('shape_matters', 'kmeans', 'YES'),
    ('shape_matters', 'gmm', 'NO'),
    ('outliers', 'dbscan', 'YES'),
    ('outliers', 'hierarchy', 'NO'),
    ('hierarchy', 'hierarchical', 'YES'),
    ('outliers', 'meanshift', 'MAYBE')
]

# Draw paths
for start, end, label in paths:
    start_pos = nodes[start]['pos']
    end_pos = nodes[end]['pos']
    
    # Determine arrow color based on answer
    if label == 'YES':
        arrow_color = '#27ae60'
    elif label == 'NO':
        arrow_color = '#e74c3c'
    else:
        arrow_color = '#7f8c8d'
    
    # Draw arrow
    arrow = FancyArrowPatch(
        start_pos, end_pos,
        arrowstyle='->,head_width=0.3,head_length=0.3',
        color=arrow_color,
        linewidth=2,
        alpha=0.7,
        connectionstyle="arc3,rad=0.1"
    )
    ax.add_patch(arrow)
    
    # Add label
    if label:
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2
        ax.text(mid_x, mid_y, label,
               fontsize=9, fontweight='bold',
               color=arrow_color,
               bbox=dict(boxstyle='round,pad=0.2',
                        facecolor='white', alpha=0.8))

# Add criteria boxes
criteria_data = [
    (1, 8.5, 'DATA SIZE', ['< 1K: Any algorithm', '1K-100K: K-means/DBSCAN', '> 100K: Mini-batch K-means']),
    (13, 8.5, 'SHAPE TYPE', ['Spherical: K-means', 'Elliptical: GMM', 'Arbitrary: DBSCAN/Mean Shift']),
    (1, 1, 'PARAMETERS', ['K-means: K only', 'DBSCAN: eps, MinPts', 'Hierarchical: distance', 'Mean Shift: bandwidth']),
    (13, 1, 'OUTPUT TYPE', ['K-means: Hard labels', 'GMM: Probabilities', 'DBSCAN: Core/border/noise', 'Hierarchical: Tree'])
]

for x, y, title, items in criteria_data:
    # Draw box
    box = FancyBboxPatch(
        (x - 1.2, y - 0.8),
        2.4, 1.6,
        boxstyle="round,pad=0.05",
        facecolor='#ecf0f1',
        edgecolor='#95a5a6',
        linewidth=1
    )
    ax.add_patch(box)
    
    # Add title
    ax.text(x, y + 0.5, title,
           fontsize=10, fontweight='bold',
           ha='center', color='#2c3e50')
    
    # Add items
    for i, item in enumerate(items):
        ax.text(x, y + 0.1 - i*0.3, item,
               fontsize=8, ha='center',
               color='#34495e')

# Add title and subtitle
ax.text(7, 9.8, 'Clustering Algorithm Selection Guide',
       fontsize=16, fontweight='bold', ha='center', color='#2c3e50')
ax.text(7, 9.4, 'Follow the decision tree to find the best algorithm for your innovation analysis',
       fontsize=11, ha='center', style='italic', color='#7f8c8d')

# Add legend
legend_items = [
    ('YES path', '#27ae60'),
    ('NO path', '#e74c3c'),
    ('Alternative', '#7f8c8d')
]

for i, (label, color) in enumerate(legend_items):
    ax.plot([0.5, 1], [0.3 - i*0.2, 0.3 - i*0.2],
           color=color, linewidth=2)
    ax.text(1.2, 0.3 - i*0.2, label,
           fontsize=9, va='center', color=color)

# Remove axes
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

plt.tight_layout()

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/clustering_decision_tree.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/clustering_decision_tree.png', 
           dpi=150, bbox_inches='tight')

print("Clustering decision tree created successfully!")
print("Files saved:")
print("  - charts/clustering_decision_tree.pdf")
print("  - charts/clustering_decision_tree.png")