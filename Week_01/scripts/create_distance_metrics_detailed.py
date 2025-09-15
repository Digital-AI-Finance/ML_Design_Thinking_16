import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow, FancyArrowPatch
import matplotlib.lines as mlines

# Set random seed for reproducibility
np.random.seed(42)

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

# Create figure with three subplots for each metric
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Points for demonstration
point1 = np.array([2, 3])
point2 = np.array([5, 7])

# EUCLIDEAN DISTANCE
ax1 = axes[0]
ax1.set_title('Euclidean Distance\n"As the crow flies"', fontsize=14, fontweight='bold', color=colors['mlblue'])

# Plot points
ax1.scatter(*point1, s=200, c=colors['mlred'], zorder=5)
ax1.scatter(*point2, s=200, c=colors['mlgreen'], zorder=5)
ax1.text(point1[0]-0.3, point1[1], 'A', fontsize=14, fontweight='bold')
ax1.text(point2[0]+0.3, point2[1], 'B', fontsize=14, fontweight='bold')

# Draw direct line
ax1.plot([point1[0], point2[0]], [point1[1], point2[1]], 
         'b-', linewidth=3, alpha=0.7, label='Direct path')

# Add dotted lines for calculation visualization
ax1.plot([point1[0], point2[0]], [point1[1], point1[1]], 
         'k--', linewidth=1, alpha=0.5)
ax1.plot([point2[0], point2[0]], [point1[1], point2[1]], 
         'k--', linewidth=1, alpha=0.5)

# Add measurements
ax1.text((point1[0] + point2[0])/2, point1[1] - 0.3, 
         f'Δx = {point2[0] - point1[0]:.0f}', ha='center', fontsize=10)
ax1.text(point2[0] + 0.3, (point1[1] + point2[1])/2, 
         f'Δy = {point2[1] - point1[1]:.0f}', ha='left', fontsize=10)

# Calculate and show distance
euclidean_dist = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
ax1.text(3.5, 5.5, f'd = √(Δx² + Δy²)', fontsize=12, fontweight='bold', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))
ax1.text(3.5, 4.8, f'd = √(9 + 16) = {euclidean_dist:.1f}', fontsize=10)

ax1.set_xlim(0, 7)
ax1.set_ylim(0, 9)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# MANHATTAN DISTANCE
ax2 = axes[1]
ax2.set_title('Manhattan Distance\n"City blocks"', fontsize=14, fontweight='bold', color=colors['mlorange'])

# Plot points
ax2.scatter(*point1, s=200, c=colors['mlred'], zorder=5)
ax2.scatter(*point2, s=200, c=colors['mlgreen'], zorder=5)
ax2.text(point1[0]-0.3, point1[1], 'A', fontsize=14, fontweight='bold')
ax2.text(point2[0]+0.3, point2[1], 'B', fontsize=14, fontweight='bold')

# Draw Manhattan path (step pattern)
ax2.plot([point1[0], point2[0]], [point1[1], point1[1]], 
         'r-', linewidth=3, alpha=0.7)
ax2.plot([point2[0], point2[0]], [point1[1], point2[1]], 
         'r-', linewidth=3, alpha=0.7)

# Alternative path (showing it's the same distance)
ax2.plot([point1[0], point1[0]], [point1[1], point2[1]], 
         'r--', linewidth=2, alpha=0.3)
ax2.plot([point1[0], point2[0]], [point2[1], point2[1]], 
         'r--', linewidth=2, alpha=0.3)

# Add arrows to show movement
arrow1 = FancyArrowPatch((point1[0], point1[1]), (point2[0], point1[1]),
                        arrowstyle='->', mutation_scale=20, 
                        color=colors['mlorange'], linewidth=2)
ax2.add_patch(arrow1)
arrow2 = FancyArrowPatch((point2[0], point1[1]), (point2[0], point2[1]),
                        arrowstyle='->', mutation_scale=20, 
                        color=colors['mlorange'], linewidth=2)
ax2.add_patch(arrow2)

# Calculate and show distance
manhattan_dist = abs(point2[0] - point1[0]) + abs(point2[1] - point1[1])
ax2.text(3.5, 5.5, f'd = |Δx| + |Δy|', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))
ax2.text(3.5, 4.8, f'd = 3 + 4 = {manhattan_dist:.0f}', fontsize=10)

# Add grid to show city blocks
for i in range(8):
    ax2.axhline(y=i, color='gray', linestyle=':', alpha=0.3)
    ax2.axvline(x=i, color='gray', linestyle=':', alpha=0.3)

ax2.set_xlim(0, 7)
ax2.set_ylim(0, 9)
ax2.set_aspect('equal')

# COSINE SIMILARITY
ax3 = axes[2]
ax3.set_title('Cosine Similarity\n"Direction matters"', fontsize=14, fontweight='bold', color=colors['mlgreen'])

# Use vectors from origin
origin = np.array([0, 0])
vec1 = np.array([3, 2])
vec2 = np.array([2, 4])

# Plot vectors
ax3.arrow(0, 0, vec1[0], vec1[1], head_width=0.2, head_length=0.2, 
          fc=colors['mlred'], ec=colors['mlred'], linewidth=2)
ax3.arrow(0, 0, vec2[0], vec2[1], head_width=0.2, head_length=0.2, 
          fc=colors['mlgreen'], ec=colors['mlgreen'], linewidth=2)

ax3.text(vec1[0]+0.2, vec1[1], 'A', fontsize=14, fontweight='bold')
ax3.text(vec2[0]+0.2, vec2[1], 'B', fontsize=14, fontweight='bold')

# Draw angle arc
angle1 = np.arctan2(vec1[1], vec1[0])
angle2 = np.arctan2(vec2[1], vec2[0])
angles = np.linspace(angle1, angle2, 50)
arc_r = 1.5
arc_x = arc_r * np.cos(angles)
arc_y = arc_r * np.sin(angles)
ax3.plot(arc_x, arc_y, 'b-', linewidth=2, alpha=0.7)

# Add angle symbol
mid_angle = (angle1 + angle2) / 2
ax3.text(1.8 * np.cos(mid_angle), 1.8 * np.sin(mid_angle), 'θ', 
         fontsize=14, fontweight='bold', color=colors['mlblue'])

# Calculate cosine similarity
cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
angle_deg = np.degrees(np.arccos(cos_sim))

ax3.text(2, 5, f'cos(θ) = A·B / (|A||B|)', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))
ax3.text(2, 4.3, f'cos(θ) = {cos_sim:.2f}', fontsize=10)
ax3.text(2, 3.8, f'θ = {angle_deg:.1f}°', fontsize=10)

# Add origin point
ax3.scatter(0, 0, s=100, c='black', zorder=5)
ax3.text(-0.3, -0.3, 'Origin', fontsize=10)

ax3.set_xlim(-0.5, 5)
ax3.set_ylim(-0.5, 6)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

plt.suptitle('Distance Metrics: How We Measure "Closeness"', 
             fontsize=18, fontweight='bold', y=1.02)

plt.tight_layout()

# Save the figure
plt.savefig('../charts/distance_metrics_detailed.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/distance_metrics_detailed.png', dpi=150, bbox_inches='tight')
print("Distance metrics detailed visualization created successfully!")
plt.close()