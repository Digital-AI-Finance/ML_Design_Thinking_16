#!/usr/bin/env python3
"""
Create Algorithm Complexity Analysis Visualization for Week 1
Shows time/space complexity and scalability comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle

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

# Main complexity comparison table
ax1 = plt.subplot(2, 3, 1, frameon=False)
ax1.axis('off')

# Title
ax1.text(0.5, 0.95, 'Algorithm Complexity Analysis', fontsize=14, fontweight='bold',
        ha='center', transform=ax1.transAxes, color=colors['mlblue'])
ax1.text(0.5, 0.91, 'Big O Notation Comparison', fontsize=10,
        ha='center', transform=ax1.transAxes, style='italic', color='gray')

# Create complexity table
complexity_data = [
    ['Algorithm', 'Time Complexity', 'Space Complexity', 'Scalability'],
    ['K-means', 'O(n·k·i·d)', 'O(n·d + k·d)', 'Excellent'],
    ['DBSCAN', 'O(n²) / O(n log n)*', 'O(n)', 'Good'],
    ['Hierarchical', 'O(n³) / O(n² log n)*', 'O(n²)', 'Poor'],
    ['GMM', 'O(n·k²·i·d)', 'O(k·d²)', 'Moderate'],
    ['Spectral', 'O(n³)', 'O(n²)', 'Poor'],
]

# Draw table
for i, row in enumerate(complexity_data):
    for j, cell in enumerate(row):
        x_pos = 0.05 + j * 0.24
        y_pos = 0.80 - i * 0.12
        
        if i == 0:  # Header
            ax1.text(x_pos, y_pos, cell, fontsize=10, fontweight='bold',
                    transform=ax1.transAxes)
            # Draw header line
            ax1.plot([0.02, 0.98], [y_pos - 0.03, y_pos - 0.03],
                    'k-', linewidth=1, transform=ax1.transAxes)
        else:
            # Color code scalability
            if j == 3:  # Scalability column
                if cell == 'Excellent':
                    color = colors['mlgreen']
                elif cell == 'Good':
                    color = colors['mlorange']
                elif cell == 'Moderate':
                    color = colors['yellow']
                else:  # Poor
                    color = colors['mlred']
            else:
                color = colors['dark']
            
            weight = 'bold' if j == 0 else 'normal'
            ax1.text(x_pos, y_pos, cell, fontsize=9, fontweight=weight,
                    transform=ax1.transAxes, color=color)
        
        # Add row background
        if i > 0 and i % 2 == 0:
            rect = Rectangle((0.02, y_pos - 0.02), 0.96, 0.10,
                           facecolor=colors['light'], alpha=0.3,
                           transform=ax1.transAxes)
            ax1.add_patch(rect)

# Add notation explanation
notation_text = (
    "Notation Guide:\n"
    "n = number of data points\n"
    "k = number of clusters\n"
    "i = number of iterations\n"
    "d = number of dimensions\n"
    "* = with spatial index"
)
ax1.text(0.02, 0.15, notation_text, fontsize=8, transform=ax1.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

# Scalability visualization
ax2 = plt.subplot(2, 3, 2)
n_values = np.logspace(2, 6, 50)  # 100 to 1,000,000 data points

# Calculate time complexity curves (simplified for visualization)
kmeans_time = n_values * 5 * 10 * 10  # n*k*i*d with k=5, i=10, d=10
dbscan_time = n_values * np.log(n_values) * 10  # n log n with index
hierarchical_time = n_values**2 * np.log(n_values) / 1000  # n² log n
gmm_time = n_values * 25 * 10 * 10  # n*k²*i*d

ax2.loglog(n_values, kmeans_time, label='K-means', color=colors['mlblue'], linewidth=2)
ax2.loglog(n_values, dbscan_time, label='DBSCAN (indexed)', color=colors['mlgreen'], linewidth=2)
ax2.loglog(n_values, hierarchical_time, label='Hierarchical', color=colors['mlred'], linewidth=2)
ax2.loglog(n_values, gmm_time, label='GMM', color=colors['mlpurple'], linewidth=2)

ax2.set_xlabel('Number of Data Points (n)', fontsize=9)
ax2.set_ylabel('Time (arbitrary units)', fontsize=9)
ax2.set_title('Scalability Comparison', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

# Add performance zones
ax2.axvspan(100, 1000, alpha=0.1, color='green', label='Small data')
ax2.axvspan(1000, 100000, alpha=0.1, color='yellow', label='Medium data')
ax2.axvspan(100000, 1000000, alpha=0.1, color='red', label='Large data')

# Memory usage comparison
ax3 = plt.subplot(2, 3, 3)
algorithms = ['K-means', 'DBSCAN', 'Hierarchical', 'GMM', 'Spectral']
memory_usage = [1, 0.8, 5, 2, 4]  # Relative memory usage
colors_list = [colors['mlblue'], colors['mlgreen'], colors['mlred'], 
               colors['mlpurple'], colors['mlorange']]

bars = ax3.bar(algorithms, memory_usage, color=colors_list, alpha=0.7)
ax3.set_ylabel('Relative Memory Usage', fontsize=9)
ax3.set_title('Space Complexity Comparison', fontsize=11, fontweight='bold')
ax3.set_ylim(0, 6)

# Add values on bars
for bar, value in zip(bars, memory_usage):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{value}x', ha='center', va='bottom', fontsize=8)

# Practical recommendations
ax4 = plt.subplot(2, 3, 4)
ax4.axis('off')

ax4.text(0.5, 0.95, 'Practical Recommendations', fontsize=12, fontweight='bold',
        ha='center', transform=ax4.transAxes, color=colors['mlblue'])

recommendations = [
    ('Small Data (<10K points)', 'Any algorithm works', colors['mlgreen']),
    ('Medium Data (10K-100K)', 'K-means or DBSCAN', colors['mlorange']),
    ('Large Data (>100K)', 'MiniBatch K-means', colors['mlred']),
    ('High Dimensions (>50)', 'Consider PCA first', colors['mlpurple']),
    ('Real-time Requirements', 'Pre-computed K-means', colors['mlblue']),
    ('Memory Constrained', 'Avoid Hierarchical', colors['yellow'])
]

y_pos = 0.85
for scenario, recommendation, color in recommendations:
    # Scenario
    ax4.text(0.05, y_pos, scenario, fontsize=9, fontweight='bold',
            transform=ax4.transAxes)
    # Recommendation
    ax4.text(0.05, y_pos - 0.04, f'→ {recommendation}', fontsize=8,
            transform=ax4.transAxes, color=color, style='italic')
    y_pos -= 0.12

# Optimization techniques
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')

ax5.text(0.5, 0.95, 'Optimization Techniques', fontsize=12, fontweight='bold',
        ha='center', transform=ax5.transAxes, color=colors['mlorange'])

techniques = [
    'MiniBatch K-means:',
    '  • Samples subset of data',
    '  • 10-100x faster on large data',
    '',
    'Spatial Indexing (DBSCAN):',
    '  • KD-tree or Ball-tree',
    '  • O(n²) → O(n log n)',
    '',
    'Dimensionality Reduction:',
    '  • PCA before clustering',
    '  • Reduces d in O(n·k·i·d)',
    '',
    'Early Stopping:',
    '  • Monitor convergence',
    '  • Stop when stable'
]

y_pos = 0.85
for technique in techniques:
    if technique.startswith('  '):
        ax5.text(0.1, y_pos, technique, fontsize=8,
                transform=ax5.transAxes, color='gray')
    elif technique == '':
        pass
    else:
        ax5.text(0.05, y_pos, technique, fontsize=9, fontweight='bold',
                transform=ax5.transAxes)
    y_pos -= 0.05

# Implementation complexity
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

ax6.text(0.5, 0.95, 'Implementation Complexity', fontsize=12, fontweight='bold',
        ha='center', transform=ax6.transAxes, color=colors['mlpurple'])

impl_data = [
    ['Algorithm', 'Ease', 'Lines of Code*', 'Tuning'],
    ['K-means', 'Easy', '~50', 'Simple'],
    ['DBSCAN', 'Moderate', '~100', 'Tricky'],
    ['Hierarchical', 'Easy', '~30', 'Simple'],
    ['GMM', 'Hard', '~200', 'Complex'],
    ['Spectral', 'Hard', '~150', 'Complex']
]

for i, row in enumerate(impl_data):
    for j, cell in enumerate(row):
        x_pos = 0.05 + j * 0.23
        y_pos = 0.80 - i * 0.10
        
        if i == 0:  # Header
            ax6.text(x_pos, y_pos, cell, fontsize=9, fontweight='bold',
                    transform=ax6.transAxes)
        else:
            # Color code difficulty
            if j == 1:  # Ease column
                if cell == 'Easy':
                    color = colors['mlgreen']
                elif cell == 'Moderate':
                    color = colors['mlorange']
                else:  # Hard
                    color = colors['mlred']
            elif j == 3:  # Tuning column
                if cell == 'Simple':
                    color = colors['mlgreen']
                elif cell == 'Tricky':
                    color = colors['mlorange']
                else:  # Complex
                    color = colors['mlred']
            else:
                color = colors['dark']
            
            ax6.text(x_pos, y_pos, cell, fontsize=8,
                    transform=ax6.transAxes, color=color)

ax6.text(0.5, 0.15, '* Approximate implementation from scratch',
        fontsize=7, ha='center', transform=ax6.transAxes, style='italic', color='gray')

# Add performance tips box
tips_text = (
    "Performance Tips:\n"
    "1. Profile before optimizing\n"
    "2. Use vectorized operations\n"
    "3. Consider approximate methods\n"
    "4. Parallelize when possible"
)
ax6.text(0.5, 0.05, tips_text, fontsize=8, ha='center', transform=ax6.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))

# Overall title and footer
fig.suptitle('Clustering Algorithm Complexity & Performance Guide', 
            fontsize=16, fontweight='bold', y=0.98)
fig.text(0.5, 0.01, 'Choose algorithms based on data size, dimensionality, and performance requirements',
        fontsize=9, ha='center', style='italic', color='gray')

plt.tight_layout(rect=[0, 0.02, 1, 0.96])

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/algorithm_complexity.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/algorithm_complexity.png', 
           dpi=150, bbox_inches='tight')

print("Algorithm complexity analysis created successfully!")
print("Files saved:")
print("  - charts/algorithm_complexity.pdf")
print("  - charts/algorithm_complexity.png")
print("\nComplexity Summary:")
print("  - Time & space complexity comparison")
print("  - Scalability visualization")
print("  - Memory usage comparison")
print("  - Practical recommendations")
print("  - Optimization techniques")
print("  - Implementation complexity")