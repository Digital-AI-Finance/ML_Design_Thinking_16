#!/usr/bin/env python3
"""
Create Innovation Taxonomy Visualization for Week 1
Shows different types of innovations and their relationships
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import matplotlib.patches as mpatches

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Define innovation taxonomy with hierarchical structure
taxonomy = {
    'Product Innovation': {
        'pos': (3, 7),
        'color': '#1f77b4',
        'subtypes': ['New Features', 'Performance', 'Design'],
        'examples': ['iPhone', 'Tesla', 'Dyson']
    },
    'Process Innovation': {
        'pos': (7, 7),
        'color': '#ff7f0e',
        'subtypes': ['Automation', 'Efficiency', 'Quality'],
        'examples': ['Lean Mfg', 'Six Sigma', 'Agile']
    },
    'Business Model': {
        'pos': (11, 7),
        'color': '#2ca02c',
        'subtypes': ['Revenue', 'Delivery', 'Value Chain'],
        'examples': ['Subscription', 'Platform', 'Freemium']
    },
    'Marketing Innovation': {
        'pos': (3, 3),
        'color': '#d62728',
        'subtypes': ['Channels', 'Pricing', 'Promotion'],
        'examples': ['Viral', 'Dynamic Pricing', 'Content']
    },
    'Organizational': {
        'pos': (7, 3),
        'color': '#9467bd',
        'subtypes': ['Structure', 'Culture', 'Partnerships'],
        'examples': ['Holacracy', 'Remote', 'Open Innovation']
    },
    'Service Innovation': {
        'pos': (11, 3),
        'color': '#8c564b',
        'subtypes': ['Experience', 'Delivery', 'Support'],
        'examples': ['Concierge', 'Self-Service', 'AI Support']
    }
}

# Draw main innovation categories
for category, info in taxonomy.items():
    x, y = info['pos']
    
    # Main box
    box = FancyBboxPatch((x - 1.3, y - 0.6), 2.6, 1.2,
                         boxstyle="round,pad=0.05",
                         facecolor=info['color'], alpha=0.3,
                         edgecolor=info['color'], linewidth=3)
    ax.add_patch(box)
    
    # Category name
    ax.text(x, y + 0.2, category, fontsize=12, fontweight='bold',
           ha='center', va='center')
    
    # Subtypes
    subtypes_text = ' • '.join(info['subtypes'])
    ax.text(x, y - 0.2, subtypes_text, fontsize=8,
           ha='center', va='center', style='italic')
    
    # Examples below
    examples_text = ', '.join(info['examples'])
    ax.text(x, y - 0.9, f"Ex: {examples_text}", fontsize=7,
           ha='center', va='center', color='gray')

# Add connections showing innovation combinations
connections = [
    # Product + Process = Breakthrough
    ((3, 7), (7, 7), 'Manufacturing\nInnovation'),
    # Process + Business Model = Disruption
    ((7, 7), (11, 7), 'Digital\nTransformation'),
    # Business Model + Service = Platform
    ((11, 7), (11, 3), 'Platform\nEconomy'),
    # Marketing + Organizational = Growth
    ((3, 3), (7, 3), 'Growth\nHacking'),
    # Organizational + Service = Experience
    ((7, 3), (11, 3), 'Customer\nCentricity'),
    # Product + Marketing = Launch
    ((3, 7), (3, 3), 'Go-to-Market'),
]

for start, end, label in connections:
    arrow = FancyArrowPatch(start, end,
                          arrowstyle='<->', mutation_scale=20,
                          color='gray', alpha=0.5, linewidth=1.5,
                          connectionstyle="arc3,rad=0.2")
    ax.add_patch(arrow)
    
    # Connection label
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    ax.text(mid_x, mid_y, label, fontsize=8,
           ha='center', va='center', style='italic',
           bbox=dict(boxstyle='round,pad=0.2', 
                    facecolor='white', alpha=0.8))

# Add innovation impact levels
impact_levels = [
    (2, 9, 'RADICAL', 'New markets, new technology', '#e74c3c'),
    (7, 9, 'DISRUPTIVE', 'Change industry rules', '#f39c12'),
    (12, 9, 'INCREMENTAL', 'Continuous improvement', '#27ae60')
]

for x, y, level, desc, color in impact_levels:
    circle = Circle((x, y), 0.6, color=color, alpha=0.3)
    ax.add_patch(circle)
    ax.text(x, y + 0.1, level, fontsize=10, fontweight='bold',
           ha='center', va='center')
    ax.text(x, y - 0.2, desc, fontsize=7,
           ha='center', va='center', style='italic')

# Add innovation drivers at the bottom
drivers = [
    (3, 1, 'Technology\nPush', '#3498db'),
    (5.5, 1, 'Market\nPull', '#e67e22'),
    (8, 1, 'Regulatory\nChange', '#9b59b6'),
    (10.5, 1, 'Social\nTrends', '#1abc9c')
]

for x, y, driver, color in drivers:
    box = FancyBboxPatch((x - 0.7, y - 0.3), 1.4, 0.6,
                         boxstyle="round,pad=0.02",
                         facecolor=color, alpha=0.2,
                         edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, driver, fontsize=9,
           ha='center', va='center')

# Add arrows from drivers to innovation types
driver_connections = [
    ((3, 1.3), (3, 2.4)),  # Tech to Marketing
    ((5.5, 1.3), (7, 2.4)),  # Market to Organizational
    ((8, 1.3), (7, 2.4)),  # Regulatory to Organizational
    ((10.5, 1.3), (11, 2.4))  # Social to Service
]

for start, end in driver_connections:
    arrow = FancyArrowPatch(start, end,
                          arrowstyle='->', mutation_scale=15,
                          color='lightgray', alpha=0.5, linewidth=1,
                          connectionstyle="arc3,rad=0.1")
    ax.add_patch(arrow)

# Title and subtitle
ax.text(7, 10, 'Innovation Taxonomy Framework',
       fontsize=16, fontweight='bold', ha='center')
ax.text(7, 9.5, 'Types, Relationships, and Impact Levels',
       fontsize=11, ha='center', style='italic', color='gray')

# Add legend for impact levels
legend_elements = [
    mpatches.Circle((0, 0), 0.1, color='#e74c3c', alpha=0.3, label='Radical'),
    mpatches.Circle((0, 0), 0.1, color='#f39c12', alpha=0.3, label='Disruptive'),
    mpatches.Circle((0, 0), 0.1, color='#27ae60', alpha=0.3, label='Incremental')
]
ax.legend(handles=legend_elements, loc='upper left', 
         title='Impact Levels', fontsize=9)

# Add clustering insight box
insight_text = (
    "ML Clustering reveals:\n"
    "• Natural innovation groupings\n"
    "• Hidden relationships\n"
    "• White space opportunities\n"
    "• Evolution patterns"
)
ax.text(0.5, 5, insight_text, fontsize=9,
       bbox=dict(boxstyle='round,pad=0.4', 
                facecolor='lightyellow', alpha=0.9))

# Remove axes
ax.set_xlim(0, 14)
ax.set_ylim(0, 10.5)
ax.axis('off')

plt.tight_layout()

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/innovation_taxonomy.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/innovation_taxonomy.png', 
           dpi=150, bbox_inches='tight')

print("Innovation taxonomy created successfully!")
print("Files saved:")
print("  - charts/innovation_taxonomy.pdf")
print("  - charts/innovation_taxonomy.png")