#!/usr/bin/env python3
"""
Create Innovation Archetypes Visualization for Week 1 Part 3
Shows different categories of innovation discovered through clustering
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Define innovation archetypes with their characteristics
archetypes = {
    'Disruptive': {
        'pos': (2, 8),
        'color': '#d62728',
        'size': 2.0,
        'examples': ['Uber', 'Airbnb', 'Netflix'],
        'characteristics': ['Low-end market entry', 'New value networks', 'Simplicity focus']
    },
    'Incremental': {
        'pos': (8, 8),
        'color': '#2ca02c',
        'size': 1.8,
        'examples': ['iPhone updates', 'Car models', 'Software patches'],
        'characteristics': ['Continuous improvement', 'Risk mitigation', 'Customer retention']
    },
    'Radical': {
        'pos': (2, 2),
        'color': '#1f77b4',
        'size': 1.6,
        'examples': ['Internet', 'Smartphone', 'CRISPR'],
        'characteristics': ['Technology breakthrough', 'New capabilities', 'Market creation']
    },
    'Architectural': {
        'pos': (8, 2),
        'color': '#ff7f0e',
        'size': 1.7,
        'examples': ['Tesla', 'Amazon AWS', 'Platform economy'],
        'characteristics': ['System reconfiguration', 'Integration focus', 'Network effects']
    },
    'Modular': {
        'pos': (5, 5),
        'color': '#9467bd',
        'size': 1.5,
        'examples': ['Microservices', 'App stores', 'Plugin systems'],
        'characteristics': ['Component innovation', 'Standardized interfaces', 'Ecosystem play']
    }
}

# Draw archetype circles
for name, info in archetypes.items():
    # Main circle
    circle = Circle(info['pos'], info['size'], 
                   color=info['color'], alpha=0.3,
                   edgecolor=info['color'], linewidth=3)
    ax.add_patch(circle)
    
    # Archetype name
    ax.text(info['pos'][0], info['pos'][1], name,
           fontsize=14, fontweight='bold', ha='center', va='center',
           color='white',
           bbox=dict(boxstyle='round,pad=0.4', 
                    facecolor=info['color'], alpha=0.9))
    
    # Examples box
    examples_text = '\n'.join(info['examples'])
    ax.text(info['pos'][0], info['pos'][1] - info['size'] - 0.5,
           examples_text, fontsize=8, ha='center',
           bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='white', alpha=0.8))

# Add connecting lines showing relationships
connections = [
    ((2, 8), (5, 5), 'Disrupts\nmodules'),
    ((8, 8), (5, 5), 'Improves\nmodules'),
    ((2, 2), (8, 2), 'Enables\narchitecture'),
    ((8, 2), (5, 5), 'Integrates\nmodules'),
    ((2, 8), (2, 2), 'Can become\nradical')
]

for start, end, label in connections:
    ax.annotate('', xy=end, xytext=start,
               arrowprops=dict(arrowstyle='->', color='gray', 
                             alpha=0.5, lw=1.5, connectionstyle="arc3,rad=0.2"))
    # Add relationship label
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    ax.text(mid_x, mid_y, label, fontsize=8, ha='center',
           color='gray', style='italic', alpha=0.8)

# Add innovation dimensions axes
ax.axhline(y=5, color='black', alpha=0.2, linestyle='--', linewidth=1)
ax.axvline(x=5, color='black', alpha=0.2, linestyle='--', linewidth=1)

# Axis labels
ax.text(0.5, 5, 'Market\nDisruption', fontsize=10, fontweight='bold',
       rotation=90, va='center', ha='center')
ax.text(9.5, 5, 'Market\nEvolution', fontsize=10, fontweight='bold',
       rotation=90, va='center', ha='center')
ax.text(5, 0.5, 'Technology Push', fontsize=10, fontweight='bold',
       ha='center', va='center')
ax.text(5, 9.5, 'Market Pull', fontsize=10, fontweight='bold',
       ha='center', va='center')

# Add quadrant labels
quadrant_labels = [
    (2.5, 9, 'TRANSFORM\n(New markets,\nexisting tech)'),
    (7.5, 9, 'OPTIMIZE\n(Existing markets,\nexisting tech)'),
    (2.5, 1, 'PIONEER\n(New markets,\nnew tech)'),
    (7.5, 1, 'EVOLVE\n(Existing markets,\nnew tech)')
]

for x, y, label in quadrant_labels:
    ax.text(x, y, label, fontsize=9, ha='center', va='center',
           style='italic', color='darkblue', alpha=0.7)

# Add characteristics legend
char_text = (
    "Innovation Characteristics:\n"
    "• Size = Market impact\n"
    "• Color = Innovation type\n"
    "• Position = Strategic approach\n"
    "• Connections = Evolution paths"
)
ax.text(0.2, 0.5, char_text, fontsize=9,
       bbox=dict(boxstyle='round,pad=0.4', 
                facecolor='lightyellow', alpha=0.9))

# Title
ax.set_title('Innovation Archetypes Discovery\nFive Distinct Patterns from Clustering Analysis', 
            fontsize=16, fontweight='bold', pad=20)

# Remove axes
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Add legend for archetype types
legend_elements = [
    mpatches.Circle((0, 0), 0.1, color='#d62728', alpha=0.5, label='Disruptive'),
    mpatches.Circle((0, 0), 0.1, color='#2ca02c', alpha=0.5, label='Incremental'),
    mpatches.Circle((0, 0), 0.1, color='#1f77b4', alpha=0.5, label='Radical'),
    mpatches.Circle((0, 0), 0.1, color='#ff7f0e', alpha=0.5, label='Architectural'),
    mpatches.Circle((0, 0), 0.1, color='#9467bd', alpha=0.5, label='Modular')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
         title='Innovation Types', framealpha=0.95)

plt.tight_layout()

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/innovation_archetypes.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/innovation_archetypes.png', 
           dpi=150, bbox_inches='tight')

print("Innovation archetypes visualization created successfully!")
print("Files saved:")
print("  - charts/innovation_archetypes.pdf")
print("  - charts/innovation_archetypes.png")