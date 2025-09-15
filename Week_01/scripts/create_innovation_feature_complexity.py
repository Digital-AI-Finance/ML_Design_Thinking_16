import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import networkx as nx

# Set random seed for reproducibility
np.random.seed(42)

# Create figure with custom layout
fig = plt.figure(figsize=(16, 10))

# Create custom grid layout
gs = fig.add_gridspec(1, 3, width_ratios=[1, 2, 1], wspace=0.15)
ax1 = fig.add_subplot(gs[0])  # Left: Single innovation
ax2 = fig.add_subplot(gs[1])  # Center: Network of features
ax3 = fig.add_subplot(gs[2])  # Right: Feature categories

# Define color palette
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'mlbrown': '#8c564b',
    'mlpink': '#e377c2',
    'mlgray': '#7f7f7f',
    'mlyellow': '#bcbd22',
    'mlcyan': '#17becf'
}

# Feature categories with colors
feature_categories = {
    'Technical': colors['mlblue'],
    'Market': colors['mlorange'],
    'User': colors['mlgreen'],
    'Financial': colors['mlred'],
    'Legal': colors['mlpurple'],
    'Environmental': colors['mlbrown'],
    'Social': colors['mlpink'],
    'Competitive': colors['mlgray'],
    'Supply Chain': colors['mlyellow'],
    'Strategic': colors['mlcyan']
}

# LEFT PANEL: Single Innovation Idea
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)

# Draw central innovation node
innovation_circle = Circle((5, 5), 1.5, color=colors['mlblue'], alpha=0.8, zorder=10)
ax1.add_patch(innovation_circle)
ax1.text(5, 5, 'One\nInnovation\nIdea', fontsize=14, fontweight='bold', 
         ha='center', va='center', color='white', zorder=11)

# Add title
ax1.text(5, 8.5, 'What looks simple...', fontsize=16, fontweight='bold', 
         ha='center', color=colors['mlblue'])

# Add question
ax1.text(5, 1.5, 'Just one idea?', fontsize=12, ha='center', 
         style='italic', color=colors['mlgray'])

ax1.axis('off')

# CENTER PANEL: Complex Feature Network
ax2.set_xlim(-2, 12)
ax2.set_ylim(-1, 11)

# Create network graph
G = nx.Graph()

# Add central node
G.add_node('idea', pos=(5, 5))

# Generate feature nodes in clusters around the center
np.random.seed(42)
feature_positions = []
feature_nodes = []
feature_colors_list = []

n_features_per_category = 27  # ~270 total features
radius_range = (2, 4.5)

for idx, (category, color) in enumerate(feature_categories.items()):
    # Calculate angle range for this category
    angle_start = (idx * 2 * np.pi) / len(feature_categories)
    angle_end = ((idx + 1) * 2 * np.pi) / len(feature_categories)
    
    for j in range(n_features_per_category):
        # Random angle within category range
        angle = np.random.uniform(angle_start, angle_end)
        # Random radius
        radius = np.random.uniform(*radius_range)
        
        x = 5 + radius * np.cos(angle)
        y = 5 + radius * np.sin(angle)
        
        node_name = f'{category}_{j}'
        G.add_node(node_name, pos=(x, y))
        G.add_edge('idea', node_name)
        feature_positions.append((x, y))
        feature_nodes.append(node_name)
        feature_colors_list.append(color)

# Get positions
pos = nx.get_node_attributes(G, 'pos')

# Draw edges with low alpha to show complexity without overwhelming
for edge in G.edges():
    if edge[0] == 'idea' or edge[1] == 'idea':
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        ax2.plot([x1, x2], [y1, y2], color='gray', alpha=0.1, linewidth=0.5, zorder=1)

# Draw feature nodes
for node, (x, y) in pos.items():
    if node != 'idea':
        idx = feature_nodes.index(node) if node in feature_nodes else 0
        ax2.scatter(x, y, s=20, c=[feature_colors_list[idx]], alpha=0.6, zorder=2)

# Draw central innovation node
ax2.scatter(5, 5, s=500, c=colors['mlblue'], alpha=0.9, edgecolors='white', 
            linewidth=2, zorder=10)
ax2.text(5, 5, 'IDEA', fontsize=12, fontweight='bold', ha='center', 
         va='center', color='white', zorder=11)

# Add annotations
ax2.text(5, 9.5, '...is actually THIS complex!', fontsize=16, fontweight='bold', 
         ha='center', color=colors['mlred'])

# Add feature count
total_features = len(feature_nodes)
ax2.text(5, 0.5, f'{total_features} Features', fontsize=20, fontweight='bold', 
         ha='center', color=colors['mlblue'])
ax2.text(5, -0.2, f'{total_features * (total_features-1) // 2:,} Possible Interactions', 
         fontsize=14, ha='center', color=colors['mlorange'])

ax2.axis('off')

# RIGHT PANEL: Feature Categories
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)

# Title
ax3.text(5, 9.5, 'Feature Dimensions', fontsize=16, fontweight='bold', 
         ha='center', color=colors['mlblue'])

# Draw category boxes
y_start = 8.5
box_height = 0.6
box_spacing = 0.1

for idx, (category, color) in enumerate(feature_categories.items()):
    y_pos = y_start - idx * (box_height + box_spacing)
    
    # Draw colored box
    rect = FancyBboxPatch((1, y_pos - box_height/2), 8, box_height,
                          boxstyle="round,pad=0.05", 
                          facecolor=color, alpha=0.7,
                          edgecolor='white', linewidth=1)
    ax3.add_patch(rect)
    
    # Add category name
    ax3.text(5, y_pos, category, fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    
    # Add feature count
    ax3.text(8.5, y_pos, f'~{n_features_per_category}', fontsize=10,
            ha='center', va='center', color='white')

# Add bottom text
ax3.text(5, 1.8, 'Each category contains', fontsize=11, ha='center', 
         color=colors['mlgray'])
ax3.text(5, 1.3, 'dozens of measurable features', fontsize=11, ha='center', 
         color=colors['mlgray'])

# Add conclusion
ax3.text(5, 0.5, 'Too complex for humans!', fontsize=13, fontweight='bold',
         ha='center', color=colors['mlred'], style='italic')

ax3.axis('off')

# Add main title
fig.suptitle('The Hidden Complexity: Each Innovation Depends on Hundreds of Features', 
             fontsize=22, fontweight='bold', y=0.98)

# Add subtitle
fig.text(0.5, 0.02, 'This is why we need Machine Learning to find patterns in innovation data', 
         fontsize=16, ha='center', color=colors['mlpurple'], style='italic')

plt.tight_layout()

# Save the figure
plt.savefig('../charts/innovation_feature_complexity.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/innovation_feature_complexity.png', dpi=150, bbox_inches='tight')
print(f"Innovation feature complexity chart created successfully!")
print(f"Total features shown: {total_features}")
print(f"Total possible interactions: {total_features * (total_features-1) // 2:,}")
plt.close()