import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set random seed for reproducibility
np.random.seed(42)

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

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

# Generate scattered innovation ideas (5000+ points)
n_ideas = 5000
np.random.seed(42)

# Create multiple overlapping distributions to represent different innovation types
# But they're all mixed together without clear boundaries
centers = [(2, 3), (5, 7), (8, 4), (3, 8), (7, 2), (1, 6), (9, 8), (5, 1)]
ideas_x = []
ideas_y = []

for center in centers:
    cluster_size = np.random.randint(400, 800)
    x = np.random.normal(center[0], 1.5, cluster_size)
    y = np.random.normal(center[1], 1.5, cluster_size)
    ideas_x.extend(x)
    ideas_y.extend(y)

# Add some random noise points
noise_x = np.random.uniform(0, 10, 500)
noise_y = np.random.uniform(0, 10, 500)
ideas_x.extend(noise_x)
ideas_y.extend(noise_y)

# Trim to exactly 5000 points
ideas_x = np.array(ideas_x[:n_ideas])
ideas_y = np.array(ideas_y[:n_ideas])

# Plot all points in gray to show confusion
ax.scatter(ideas_x, ideas_y, c='#cccccc', alpha=0.3, s=20, edgecolors='none')

# Highlight a few random points in different colors to show potential categories
actual_points = len(ideas_x)
highlight_indices = np.random.choice(actual_points, min(50, actual_points), replace=False)
n_highlights = len(highlight_indices)
highlight_colors = [list(colors.values())[i % len(colors)] for i in range(n_highlights)]
ax.scatter(ideas_x[highlight_indices],
           ideas_y[highlight_indices],
           c=highlight_colors, s=100, alpha=0.8, edgecolors='white', linewidth=1.5)

# Add question marks at strategic positions
question_positions = [(2, 9), (8, 8), (1, 2), (9, 1), (5, 5)]
for pos in question_positions:
    ax.text(pos[0], pos[1], '?', fontsize=40, color=colors['mlred'], 
            fontweight='bold', ha='center', va='center', alpha=0.7)

# Add title and labels
ax.set_title('5000+ Innovation Ideas: Where Do We Start?', fontsize=24, fontweight='bold', pad=20)
ax.text(5, -0.8, 'How do we identify patterns?', fontsize=16, ha='center', color=colors['mlblue'])
ax.text(5, -1.3, 'Which ideas belong together?', fontsize=16, ha='center', color=colors['mlorange'])
ax.text(5, -1.8, 'What are the innovation categories?', fontsize=16, ha='center', color=colors['mlgreen'])

# Style the plot
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-2.5, 10.5)
ax.set_aspect('equal')
ax.axis('off')

# Add a subtle grid
for i in range(11):
    ax.axhline(y=i, color='gray', linestyle=':', alpha=0.1)
    ax.axvline(x=i, color='gray', linestyle=':', alpha=0.1)

# Add annotation arrows pointing to clusters that might exist
arrow_props = dict(arrowstyle='->', lw=2, color=colors['mlpurple'], alpha=0.5)
ax.annotate('Hidden Pattern?', xy=(2, 3), xytext=(0.5, 5),
            arrowprops=arrow_props, fontsize=12, color=colors['mlpurple'])
ax.annotate('Related Ideas?', xy=(8, 4), xytext=(9.5, 6),
            arrowprops=arrow_props, fontsize=12, color=colors['mlpurple'])
ax.annotate('Innovation Type?', xy=(5, 7), xytext=(3, 9.5),
            arrowprops=arrow_props, fontsize=12, color=colors['mlpurple'])

# Add a call-to-action box
rect = Rectangle((0.5, 9.2), 4, 1, linewidth=2, 
                edgecolor=colors['mlred'], facecolor='white', alpha=0.9)
ax.add_patch(rect)
ax.text(2.5, 9.7, 'Challenge: Find the Structure!', 
        fontsize=14, fontweight='bold', ha='center', va='center', color=colors['mlred'])

plt.tight_layout()

# Save the figure
plt.savefig('../charts/innovation_discovery.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/innovation_discovery.png', dpi=150, bbox_inches='tight')
print("Innovation discovery chart created successfully!")
plt.close()