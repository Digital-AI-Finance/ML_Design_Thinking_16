#!/usr/bin/env python3
"""
Create Innovation Pattern Maps for Week 1 Part 3
Shows how different innovation categories map to market opportunities
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Define 4 different innovation pattern maps
patterns = [
    {
        'title': 'Technology Maturity Map',
        'xlabel': 'Technical Complexity',
        'ylabel': 'Market Readiness',
        'clusters': [
            {'name': 'Emerging', 'center': [2, 3], 'cov': [[0.8, 0.2], [0.2, 0.6]], 'color': '#1f77b4'},
            {'name': 'Growth', 'center': [5, 6], 'cov': [[1.0, 0.3], [0.3, 0.8]], 'color': '#ff7f0e'},
            {'name': 'Mature', 'center': [8, 8], 'cov': [[0.7, 0.1], [0.1, 0.5]], 'color': '#2ca02c'},
            {'name': 'Declining', 'center': [7, 3], 'cov': [[0.6, -0.2], [-0.2, 0.7]], 'color': '#d62728'}
        ]
    },
    {
        'title': 'Market Opportunity Map',
        'xlabel': 'Competition Intensity',
        'ylabel': 'Growth Potential',
        'clusters': [
            {'name': 'Blue Ocean', 'center': [2, 8], 'cov': [[0.9, 0.1], [0.1, 0.7]], 'color': '#17becf'},
            {'name': 'Red Ocean', 'center': [8, 3], 'cov': [[0.8, -0.2], [-0.2, 0.6]], 'color': '#e377c2'},
            {'name': 'Niche', 'center': [3, 5], 'cov': [[0.5, 0.1], [0.1, 0.8]], 'color': '#bcbd22'},
            {'name': 'Mass Market', 'center': [7, 7], 'cov': [[1.2, 0.3], [0.3, 0.9]], 'color': '#7f7f7f'}
        ]
    },
    {
        'title': 'Resource Requirement Map',
        'xlabel': 'Capital Intensity',
        'ylabel': 'Talent Requirements',
        'clusters': [
            {'name': 'Bootstrappable', 'center': [2, 2], 'cov': [[0.6, 0.1], [0.1, 0.5]], 'color': '#2ca02c'},
            {'name': 'VC-Backed', 'center': [5, 7], 'cov': [[1.0, 0.3], [0.3, 0.8]], 'color': '#ff7f0e'},
            {'name': 'Enterprise', 'center': [8, 8], 'cov': [[0.8, 0.2], [0.2, 0.7]], 'color': '#d62728'},
            {'name': 'Research', 'center': [3, 8], 'cov': [[0.7, -0.1], [-0.1, 0.9]], 'color': '#9467bd'}
        ]
    },
    {
        'title': 'Innovation Speed Map',
        'xlabel': 'Time to Market',
        'ylabel': 'Iteration Velocity',
        'clusters': [
            {'name': 'Rapid', 'center': [2, 8], 'cov': [[0.7, 0.2], [0.2, 0.6]], 'color': '#1f77b4'},
            {'name': 'Steady', 'center': [5, 5], 'cov': [[1.0, 0.1], [0.1, 0.8]], 'color': '#ff7f0e'},
            {'name': 'Slow', 'center': [8, 3], 'cov': [[0.8, -0.2], [-0.2, 0.7]], 'color': '#d62728'},
            {'name': 'Agile', 'center': [3, 7], 'cov': [[0.6, 0.3], [0.3, 0.9]], 'color': '#2ca02c'}
        ]
    }
]

# Generate and plot each pattern map
for idx, (ax, pattern) in enumerate(zip(axes, patterns)):
    # Generate data points for each cluster
    all_points = []
    all_labels = []
    
    for i, cluster in enumerate(pattern['clusters']):
        # Generate points for this cluster
        points = np.random.multivariate_normal(
            cluster['center'], cluster['cov'], 100
        )
        all_points.append(points)
        all_labels.extend([i] * 100)
    
    all_points = np.vstack(all_points)
    
    # Plot points
    for i, cluster in enumerate(pattern['clusters']):
        mask = np.array(all_labels) == i
        ax.scatter(all_points[mask, 0], all_points[mask, 1],
                  c=cluster['color'], s=20, alpha=0.4, edgecolors='none')
        
        # Add cluster center
        ax.scatter(cluster['center'][0], cluster['center'][1],
                  c='black', s=100, marker='x', linewidth=2)
        
        # Add cluster name
        ax.text(cluster['center'][0], cluster['center'][1] + 0.5,
               cluster['name'], fontsize=10, fontweight='bold',
               ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', alpha=0.8))
        
        # Add confidence ellipse
        eigenvalues, eigenvectors = np.linalg.eigh(cluster['cov'])
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues) * 2  # 2 standard deviations
        
        ellipse = Ellipse(cluster['center'], width, height,
                         angle=angle, facecolor='none',
                         edgecolor=cluster['color'], linewidth=2,
                         linestyle='--', alpha=0.7)
        ax.add_patch(ellipse)
    
    # Add opportunity zones (areas between clusters)
    if idx == 0:  # Technology Maturity Map
        ax.annotate('Innovation\nGap', xy=(3.5, 4.5), fontsize=9,
                   ha='center', color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='yellow', alpha=0.5))
    elif idx == 1:  # Market Opportunity Map
        ax.annotate('Sweet\nSpot', xy=(5, 6), fontsize=9,
                   ha='center', color='darkgreen', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='lightgreen', alpha=0.5))
    elif idx == 2:  # Resource Requirement Map
        ax.annotate('Efficiency\nZone', xy=(4, 4), fontsize=9,
                   ha='center', color='blue', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='lightblue', alpha=0.5))
    elif idx == 3:  # Innovation Speed Map
        ax.annotate('Optimal\nPace', xy=(4, 6), fontsize=9,
                   ha='center', color='purple', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='lavender', alpha=0.5))
    
    # Customize subplot
    ax.set_xlabel(pattern['xlabel'], fontsize=11, fontweight='bold')
    ax.set_ylabel(pattern['ylabel'], fontsize=11, fontweight='bold')
    ax.set_title(pattern['title'], fontsize=12, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3, linestyle='--')

# Overall title
fig.suptitle('Innovation Pattern Maps\nFour Perspectives on Innovation Categories', 
            fontsize=16, fontweight='bold', y=1.02)

# Add insights text
fig.text(0.5, -0.02, 
        'Key Insight: Each map reveals different opportunity zones where innovation categories intersect',
        ha='center', fontsize=11, fontweight='bold', style='italic')

plt.tight_layout()

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/innovation_pattern_maps.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/innovation_pattern_maps.png', 
           dpi=150, bbox_inches='tight')

print("Innovation pattern maps created successfully!")
print("Files saved:")
print("  - charts/innovation_pattern_maps.pdf")
print("  - charts/innovation_pattern_maps.png")