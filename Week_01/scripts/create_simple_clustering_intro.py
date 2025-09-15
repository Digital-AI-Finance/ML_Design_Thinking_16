#!/usr/bin/env python3
"""
Create Simple Clustering Introduction for BSc Students
No prerequisites assumed - visual first approach
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch

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
fig = plt.figure(figsize=(16, 10))

# Panel 1: What is Clustering? (Visual Analogy)
ax1 = plt.subplot(2, 3, 1)
ax1.set_title('What is Clustering?', fontsize=12, fontweight='bold')

# Before clustering - scattered points
np.random.seed(42)
scatter_x = np.random.uniform(-2, 2, 30)
scatter_y = np.random.uniform(-2, 2, 30)
ax1.scatter(scatter_x, scatter_y, c='gray', alpha=0.5, s=50)
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_xlabel('Before: Mixed Data', fontsize=10)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.text(0, -2.5, 'Random points - no groups', fontsize=9, ha='center')

# Panel 2: After Clustering
ax2 = plt.subplot(2, 3, 2)
ax2.set_title('After Clustering', fontsize=12, fontweight='bold')

# Create 3 clear clusters
cluster1 = np.random.multivariate_normal([-1, 1], [[0.1, 0], [0, 0.1]], 10)
cluster2 = np.random.multivariate_normal([1, 1], [[0.1, 0], [0, 0.1]], 10)
cluster3 = np.random.multivariate_normal([0, -1], [[0.1, 0], [0, 0.1]], 10)

ax2.scatter(cluster1[:, 0], cluster1[:, 1], c=colors['mlblue'], s=50, label='Group 1')
ax2.scatter(cluster2[:, 0], cluster2[:, 1], c=colors['mlorange'], s=50, label='Group 2')
ax2.scatter(cluster3[:, 0], cluster3[:, 1], c=colors['mlgreen'], s=50, label='Group 3')
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_xlabel('After: Organized Groups', fontsize=10)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.text(0, -2.5, 'Points grouped by similarity', fontsize=9, ha='center')

# Panel 3: Real World Example (Simplified)
ax3 = plt.subplot(2, 3, 3)
ax3.set_title('Example: Student Projects', fontsize=12, fontweight='bold')
ax3.axis('off')

# Create simple icons for project types
y_positions = [0.8, 0.5, 0.2]
project_types = ['Tech Projects', 'Art Projects', 'Science Projects']
project_colors = [colors['mlblue'], colors['mlorange'], colors['mlgreen']]

for i, (proj_type, color, y_pos) in enumerate(zip(project_types, project_colors, y_positions)):
    # Draw circle for category
    circle = Circle((0.2, y_pos), 0.08, color=color, alpha=0.7)
    ax3.add_patch(circle)
    # Add label
    ax3.text(0.35, y_pos, proj_type, fontsize=10, va='center')
    # Add example items
    examples = ['Apps, Websites', 'Paintings, Music', 'Lab work, Research'][i]
    ax3.text(0.35, y_pos - 0.05, f'({examples})', fontsize=8, va='center', style='italic', color='gray')

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

# Panel 4: How Computers See Data
ax4 = plt.subplot(2, 3, 4)
ax4.set_title('How Computers See Data', fontsize=12, fontweight='bold')
ax4.axis('off')

# Show data as numbers
data_table = [
    ['Item', 'Feature 1', 'Feature 2'],
    ['A', '3.2', '1.5'],
    ['B', '3.1', '1.6'],
    ['C', '8.5', '7.2'],
    ['D', '8.7', '7.1']
]

for i, row in enumerate(data_table):
    for j, cell in enumerate(row):
        x_pos = 0.2 + j * 0.25
        y_pos = 0.8 - i * 0.15
        
        if i == 0:  # Header
            ax4.text(x_pos, y_pos, cell, fontsize=10, fontweight='bold', ha='center')
        else:
            color = 'black'
            if i in [1, 2] and j > 0:  # Similar values
                color = colors['mlblue']
            elif i in [3, 4] and j > 0:  # Similar values
                color = colors['mlorange']
            ax4.text(x_pos, y_pos, cell, fontsize=9, ha='center', color=color)

ax4.text(0.5, 0.15, 'Similar numbers = Same group', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Panel 5: Distance = Similarity
ax5 = plt.subplot(2, 3, 5)
ax5.set_title('Distance = Similarity', fontsize=12, fontweight='bold')

# Show two points and distance
point1 = [1, 2]
point2 = [3, 4]
ax5.scatter(*point1, s=100, c=colors['mlblue'], zorder=3)
ax5.scatter(*point2, s=100, c=colors['mlorange'], zorder=3)
ax5.plot([point1[0], point2[0]], [point1[1], point2[1]], 'k--', alpha=0.5)

# Add labels
ax5.text(point1[0], point1[1] - 0.5, 'Point A', fontsize=9, ha='center')
ax5.text(point2[0], point2[1] + 0.5, 'Point B', fontsize=9, ha='center')
ax5.text(2, 2.8, 'Distance', fontsize=9, ha='center', color=colors['mlred'])

# Show nearby points
nearby1 = [0.8, 2.2]
nearby2 = [1.2, 1.8]
ax5.scatter(*nearby1, s=60, c=colors['mlblue'], alpha=0.5)
ax5.scatter(*nearby2, s=60, c=colors['mlblue'], alpha=0.5)
ax5.text(1, 1.3, 'Close = Similar', fontsize=8, ha='center', color=colors['mlgreen'])

ax5.set_xlim(0, 4)
ax5.set_ylim(0, 5)
ax5.set_xlabel('Feature 1', fontsize=9)
ax5.set_ylabel('Feature 2', fontsize=9)
ax5.grid(True, alpha=0.3)

# Panel 6: Step by Step Process
ax6 = plt.subplot(2, 3, 6)
ax6.set_title('Clustering Process', fontsize=12, fontweight='bold')
ax6.axis('off')

steps = [
    '1. Start with data points',
    '2. Measure distances',
    '3. Find close neighbors',
    '4. Form groups',
    '5. Check if good groups',
    '6. Done!'
]

y_start = 0.9
for i, step in enumerate(steps):
    # Draw step box
    if i < 5:
        box_color = colors['light']
    else:
        box_color = colors['mlgreen']
    
    rect = FancyBboxPatch((0.1, y_start - i*0.14), 0.8, 0.1,
                          boxstyle="round,pad=0.02",
                          facecolor=box_color, edgecolor=colors['dark'],
                          linewidth=1)
    ax6.add_patch(rect)
    
    # Add text
    ax6.text(0.5, y_start - i*0.14 + 0.05, step, fontsize=9,
            ha='center', va='center')
    
    # Add arrow (except for last step)
    if i < 5:
        ax6.arrow(0.5, y_start - i*0.14 - 0.02, 0, -0.08, 
                 head_width=0.03, head_length=0.02, fc='gray', ec='gray')

ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)

# Main title
fig.suptitle('Clustering: Finding Groups in Data (No Math Required!)', 
            fontsize=14, fontweight='bold', y=0.98)
fig.text(0.5, 0.95, 'A Visual Introduction for Beginners', 
        fontsize=11, ha='center', style='italic', color='gray')

# Footer message
fig.text(0.5, 0.02, 'Key Idea: Clustering finds natural groups by looking at which items are similar (close together)',
        fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout(rect=[0, 0.04, 1, 0.94])

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/simple_clustering_intro.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/simple_clustering_intro.png', 
           dpi=150, bbox_inches='tight')

print("Simple clustering introduction created successfully!")
print("Files saved:")
print("  - charts/simple_clustering_intro.pdf")
print("  - charts/simple_clustering_intro.png")