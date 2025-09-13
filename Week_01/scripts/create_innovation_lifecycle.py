#!/usr/bin/env python3
"""
Create Innovation Lifecycle Stages Diagram for Week 1
Shows how clustering helps track innovation through different stages
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Wedge
import matplotlib.patches as mpatches

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Define lifecycle stages with positions and characteristics
stages = {
    'Emergence': {
        'pos': (2, 5),
        'color': '#3498db',
        'size': 0.8,
        'ideas': 100,
        'success_rate': '2%',
        'characteristics': ['High uncertainty', 'Many variations', 'Weak signals'],
        'clustering': 'DBSCAN for outliers'
    },
    'Growth': {
        'pos': (5, 5),
        'color': '#2ecc71',
        'size': 1.0,
        'ideas': 40,
        'success_rate': '10%',
        'characteristics': ['Pattern formation', 'Early adoption', 'Rapid iteration'],
        'clustering': 'K-means for segments'
    },
    'Maturity': {
        'pos': (8, 5),
        'color': '#f39c12',
        'size': 1.2,
        'ideas': 15,
        'success_rate': '40%',
        'characteristics': ['Clear categories', 'Best practices', 'Optimization'],
        'clustering': 'Hierarchical for taxonomy'
    },
    'Decline/Renewal': {
        'pos': (11, 5),
        'color': '#e74c3c',
        'size': 0.9,
        'ideas': 5,
        'success_rate': '20%',
        'characteristics': ['Disruption signals', 'New combinations', 'Transformation'],
        'clustering': 'GMM for transitions'
    }
}

# Draw lifecycle curve
x_curve = np.linspace(1, 12, 100)
y_curve = 2 + 3 * np.exp(-(x_curve - 5)**2 / 4) + 0.5 * np.sin(x_curve)
ax.plot(x_curve, y_curve, 'gray', linewidth=2, alpha=0.3, zorder=1)

# Draw stages
for stage_name, info in stages.items():
    x, y = info['pos']
    
    # Stage circle
    circle = Circle((x, y), info['size'], color=info['color'], alpha=0.3, zorder=2)
    ax.add_patch(circle)
    
    # Stage name
    ax.text(x, y + info['size'] + 0.3, stage_name, fontsize=12, fontweight='bold',
           ha='center', va='bottom')
    
    # Success rate
    ax.text(x, y, f"{info['success_rate']}", fontsize=14, fontweight='bold',
           ha='center', va='center')
    ax.text(x, y - 0.3, f"success", fontsize=8,
           ha='center', va='center', style='italic')
    
    # Ideas count
    ax.text(x, y - info['size'] - 0.3, f"{info['ideas']} ideas", fontsize=9,
           ha='center', va='top', color='gray')
    
    # Characteristics box
    char_text = '\n'.join([f"* {c}" for c in info['characteristics']])
    ax.text(x, y - info['size'] - 1.2, char_text, fontsize=7,
           ha='center', va='top',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Clustering method
    ax.text(x, y - info['size'] - 2.5, info['clustering'], fontsize=8,
           ha='center', va='top', style='italic', color=info['color'])

# Add flow arrows between stages
flow_positions = [
    ((3, 5), (4, 5)),
    ((6, 5), (7, 5)),
    ((9, 5), (10, 5))
]

for start, end in flow_positions:
    arrow = FancyArrowPatch(start, end,
                          arrowstyle='->', mutation_scale=20,
                          color='gray', alpha=0.5, linewidth=2)
    ax.add_patch(arrow)

# Add feedback loop from Decline to Emergence
feedback_arrow = FancyArrowPatch((11, 4), (2, 4),
                               arrowstyle='->', mutation_scale=15,
                               color='purple', alpha=0.4, linewidth=1.5,
                               connectionstyle="arc3,rad=-0.3",
                               linestyle='dashed')
ax.add_patch(feedback_arrow)
ax.text(6.5, 3.2, 'Renewal Cycle', fontsize=8, ha='center',
       style='italic', color='purple')

# Add ML insights panel
insights_text = (
    "ML Clustering Insights Across Lifecycle:\n\n"
    "1. Emergence: Detect weak signals and outliers\n"
    "2. Growth: Identify emerging patterns and segments\n"
    "3. Maturity: Build comprehensive taxonomies\n"
    "4. Decline: Spot disruption and transformation"
)
ax.text(1.5, 8, insights_text, fontsize=9,
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

# Add data characteristics over lifecycle
data_chars = [
    (2, 7.5, 'Sparse\nNoisy', 8),
    (5, 7.5, 'Growing\nClustered', 8),
    (8, 7.5, 'Dense\nStructured', 8),
    (11, 7.5, 'Fragmented\nMixed', 8)
]

for x, y, label, size in data_chars:
    ax.text(x, y, label, fontsize=size, ha='center', va='center',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.5))

# Add innovation funnel visualization
funnel_x = 12.5
funnel_stages = [
    (100, 'Ideas', '#3498db'),
    (40, 'Concepts', '#2ecc71'),
    (15, 'Prototypes', '#f39c12'),
    (5, 'Products', '#e74c3c'),
    (1, 'Success', '#9b59b6')
]

for i, (count, label, color) in enumerate(funnel_stages):
    width = count / 20
    y_pos = 8 - i * 0.8
    rect = FancyBboxPatch((funnel_x - width/2, y_pos - 0.3), width, 0.6,
                         boxstyle="round,pad=0.02",
                         facecolor=color, alpha=0.3)
    ax.add_patch(rect)
    ax.text(funnel_x, y_pos, f"{count} {label}", fontsize=8,
           ha='center', va='center')

ax.text(funnel_x, 8.5, 'Innovation\nFunnel', fontsize=10, fontweight='bold',
       ha='center', va='center')

# Title and subtitle
ax.text(7, 9.5, 'Innovation Lifecycle Stages', fontsize=16, fontweight='bold', ha='center')
ax.text(7, 9, 'How Clustering Reveals Innovation Patterns at Each Stage', 
       fontsize=11, ha='center', style='italic', color='gray')

# Add stage progression indicator
stage_flow = "Weak Signals → Pattern Formation → Category Definition → Transformation"
ax.text(7, 1.5, stage_flow, fontsize=9, ha='center',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))

# Add key metrics legend
legend_elements = [
    mpatches.Circle((0, 0), 0.1, color='#3498db', alpha=0.3, label='Emergence (2% success)'),
    mpatches.Circle((0, 0), 0.1, color='#2ecc71', alpha=0.3, label='Growth (10% success)'),
    mpatches.Circle((0, 0), 0.1, color='#f39c12', alpha=0.3, label='Maturity (40% success)'),
    mpatches.Circle((0, 0), 0.1, color='#e74c3c', alpha=0.3, label='Decline/Renewal (20% success)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

# Remove axes
ax.set_xlim(0, 14)
ax.set_ylim(1, 10)
ax.axis('off')

plt.tight_layout()

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/innovation_lifecycle.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/innovation_lifecycle.png', 
           dpi=150, bbox_inches='tight')

print("Innovation lifecycle diagram created successfully!")
print("Files saved:")
print("  - charts/innovation_lifecycle.pdf")
print("  - charts/innovation_lifecycle.png")