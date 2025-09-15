import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
import matplotlib.lines as mlines

# Set random seed for reproducibility
np.random.seed(42)

# Create figure with subplots
fig = plt.figure(figsize=(14, 10))

# Define color palette
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlbrown': '#8c564b',
    'mlpink': '#e377c2',
    'mlgray': '#7f7f7f'
}

# Create two subplots: left (problem) and right (solution)
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

# LEFT SIDE: One-Size-Fits-All Problem
ax1.set_title('Current Reality: Generic Categories', fontsize=18, fontweight='bold', pad=20)

# Draw generic boxes (one-size-fits-all)
box_positions = [(0.5, 0.7), (0.5, 0.5), (0.5, 0.3)]
box_labels = ['Category A', 'Category B', 'Category C']

for pos, label in zip(box_positions, box_labels):
    box = FancyBboxPatch((pos[0] - 0.15, pos[1] - 0.05), 0.3, 0.1,
                         boxstyle="round,pad=0.01",
                         linewidth=2, edgecolor='gray',
                         facecolor='lightgray', alpha=0.5)
    ax1.add_patch(box)
    ax1.text(pos[0], pos[1], label, ha='center', va='center', 
            fontsize=12, fontweight='bold')

# Add diverse innovation types trying to fit into boxes
innovation_types = [
    {'pos': (0.15, 0.7), 'label': 'Disruptive\nTech', 'color': colors['mlred'], 'shape': 'star'},
    {'pos': (0.85, 0.7), 'label': 'Incremental\nImprovement', 'color': colors['mlblue'], 'shape': 'circle'},
    {'pos': (0.1, 0.5), 'label': 'Service\nInnovation', 'color': colors['mlgreen'], 'shape': 'triangle'},
    {'pos': (0.9, 0.5), 'label': 'Business\nModel', 'color': colors['mlorange'], 'shape': 'square'},
    {'pos': (0.2, 0.3), 'label': 'Process\nOptimization', 'color': colors['mlpurple'], 'shape': 'pentagon'},
    {'pos': (0.8, 0.3), 'label': 'Platform\nEcosystem', 'color': colors['mlpink'], 'shape': 'hexagon'},
    {'pos': (0.5, 0.1), 'label': 'Hybrid\nApproach', 'color': colors['mlbrown'], 'shape': 'diamond'},
]

# Draw innovations as mismatched shapes
for inn in innovation_types:
    if inn['shape'] == 'circle':
        circle = Circle(inn['pos'], 0.05, color=inn['color'], alpha=0.7)
        ax1.add_patch(circle)
    elif inn['shape'] == 'star':
        # Simple star representation
        ax1.scatter(inn['pos'][0], inn['pos'][1], marker='*', s=500, 
                   color=inn['color'], alpha=0.7)
    elif inn['shape'] == 'triangle':
        ax1.scatter(inn['pos'][0], inn['pos'][1], marker='^', s=300, 
                   color=inn['color'], alpha=0.7)
    elif inn['shape'] == 'square':
        ax1.scatter(inn['pos'][0], inn['pos'][1], marker='s', s=300, 
                   color=inn['color'], alpha=0.7)
    else:
        ax1.scatter(inn['pos'][0], inn['pos'][1], marker='o', s=300, 
                   color=inn['color'], alpha=0.7)
    
    # Add labels
    ax1.text(inn['pos'][0], inn['pos'][1] - 0.08, inn['label'], 
            ha='center', va='top', fontsize=8, color=inn['color'])

# Add frustrated arrows showing misfit
arrow_props = dict(arrowstyle='->', lw=1.5, color='red', alpha=0.5)
ax1.annotate('', xy=(0.35, 0.7), xytext=(0.15, 0.7), arrowprops=arrow_props)
ax1.annotate('', xy=(0.65, 0.7), xytext=(0.85, 0.7), arrowprops=arrow_props)
ax1.annotate('', xy=(0.35, 0.5), xytext=(0.1, 0.5), arrowprops=arrow_props)
ax1.annotate('', xy=(0.65, 0.5), xytext=(0.9, 0.5), arrowprops=arrow_props)

# Add X marks showing poor fit
ax1.text(0.3, 0.7, '✗', fontsize=20, color='red', ha='center', va='center')
ax1.text(0.7, 0.7, '✗', fontsize=20, color='red', ha='center', va='center')
ax1.text(0.3, 0.5, '✗', fontsize=20, color='red', ha='center', va='center')
ax1.text(0.7, 0.5, '✗', fontsize=20, color='red', ha='center', va='center')

# Add problem indicators
ax1.text(0.5, 0.15, 'Missed Opportunities', fontsize=12, ha='center', 
        color=colors['mlred'], fontweight='bold')
ax1.text(0.5, 0.05, 'Edge Cases Ignored', fontsize=12, ha='center', 
        color=colors['mlred'], fontweight='bold')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 0.9)
ax1.axis('off')

# RIGHT SIDE: Data-Driven Solution
ax2.set_title('ML Solution: Natural Clusters', fontsize=18, fontweight='bold', pad=20, color=colors['mlgreen'])

# Generate clustered data
np.random.seed(42)
n_points = 150

# Create natural clusters
clusters = {
    'Disruptive': {'center': (0.2, 0.7), 'std': 0.05, 'color': colors['mlred']},
    'Incremental': {'center': (0.8, 0.7), 'std': 0.04, 'color': colors['mlblue']},
    'Service': {'center': (0.3, 0.4), 'std': 0.06, 'color': colors['mlgreen']},
    'Platform': {'center': (0.7, 0.4), 'std': 0.05, 'color': colors['mlorange']},
    'Hybrid': {'center': (0.5, 0.2), 'std': 0.07, 'color': colors['mlpurple']}
}

for name, cluster in clusters.items():
    x = np.random.normal(cluster['center'][0], cluster['std'], 30)
    y = np.random.normal(cluster['center'][1], cluster['std'], 30)
    ax2.scatter(x, y, s=30, color=cluster['color'], alpha=0.6, label=name)
    
    # Draw cluster boundary
    circle = Circle(cluster['center'], cluster['std'] * 2.5, 
                   fill=False, edgecolor=cluster['color'], 
                   linewidth=2, alpha=0.5, linestyle='--')
    ax2.add_patch(circle)
    
    # Add cluster label
    ax2.text(cluster['center'][0], cluster['center'][1], name,
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=cluster['color'])

# Add checkmarks showing good fit
checkmark_positions = [(0.2, 0.85), (0.8, 0.85), (0.3, 0.55), (0.7, 0.55), (0.5, 0.35)]
for pos in checkmark_positions:
    ax2.text(pos[0], pos[1], '✓', fontsize=16, color=colors['mlgreen'], 
            ha='center', va='center', fontweight='bold')

# Add benefit indicators
ax2.text(0.5, 0.1, 'Perfect Fit', fontsize=12, ha='center', 
        color=colors['mlgreen'], fontweight='bold')
ax2.text(0.5, 0.03, 'All Patterns Captured', fontsize=12, ha='center', 
        color=colors['mlgreen'], fontweight='bold')

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 0.9)
ax2.axis('off')

# Add main title
fig.suptitle('The Problem with One-Size-Fits-All Innovation Categories', 
            fontsize=20, fontweight='bold', y=0.98)

# Add comparison arrow
fig.text(0.5, 0.5, '→', fontsize=60, ha='center', va='center', 
        color=colors['mlpurple'], fontweight='bold')

plt.tight_layout()

# Save the figure
plt.savefig('../charts/current_reality_visual.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/current_reality_visual.png', dpi=150, bbox_inches='tight')
print("Current reality visualization created successfully!")
plt.close()