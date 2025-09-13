#!/usr/bin/env python3
"""
Create Preprocessing Pipeline Diagram for Week 1
Shows the complete data preparation workflow for clustering
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Define pipeline stages with positions
stages = [
    {
        'name': 'Raw Innovation Data',
        'pos': (2, 8),
        'color': '#e74c3c',
        'icon': 'database',
        'examples': ['Text descriptions', 'Numerical metrics', 'Categories', 'Timestamps'],
        'issues': ['Missing values', 'Different scales', 'Mixed types']
    },
    {
        'name': 'Data Cleaning',
        'pos': (5, 8),
        'color': '#3498db',
        'icon': 'clean',
        'examples': ['Handle missing', 'Remove duplicates', 'Fix errors'],
        'issues': ['Imputation strategy', 'Outlier detection']
    },
    {
        'name': 'Feature Engineering',
        'pos': (8, 8),
        'color': '#9b59b6',
        'icon': 'gear',
        'examples': ['Create ratios', 'Time features', 'Interactions'],
        'issues': ['Feature selection', 'Domain knowledge']
    },
    {
        'name': 'Encoding',
        'pos': (11, 8),
        'color': '#2ecc71',
        'icon': 'encode',
        'examples': ['One-hot encode', 'Label encode', 'Target encode'],
        'issues': ['High cardinality', 'New categories']
    },
    {
        'name': 'Scaling',
        'pos': (11, 5),
        'color': '#f39c12',
        'icon': 'scale',
        'examples': ['StandardScaler', 'MinMaxScaler', 'RobustScaler'],
        'issues': ['Outlier impact', 'Distribution']
    },
    {
        'name': 'Dimensionality Reduction',
        'pos': (8, 5),
        'color': '#1abc9c',
        'icon': 'reduce',
        'examples': ['PCA', 't-SNE', 'Feature selection'],
        'issues': ['Information loss', 'Interpretability']
    },
    {
        'name': 'Validation Split',
        'pos': (5, 5),
        'color': '#34495e',
        'icon': 'split',
        'examples': ['Train/test', 'Cross-validation', 'Time series'],
        'issues': ['Data leakage', 'Stratification']
    },
    {
        'name': 'Ready for Clustering',
        'pos': (2, 5),
        'color': '#27ae60',
        'icon': 'check',
        'examples': ['Normalized', 'No missing', 'Numeric only'],
        'issues': ['Final checks', 'Quality metrics']
    }
]

# Draw pipeline stages
for i, stage in enumerate(stages):
    x, y = stage['pos']
    
    # Main box
    box = FancyBboxPatch((x - 1.2, y - 0.6), 2.4, 1.2,
                         boxstyle="round,pad=0.05",
                         facecolor=stage['color'], alpha=0.3,
                         edgecolor=stage['color'], linewidth=2)
    ax.add_patch(box)
    
    # Stage name
    ax.text(x, y + 0.2, stage['name'], fontsize=10, fontweight='bold',
           ha='center', va='center')
    
    # Examples
    examples_text = ' | '.join(stage['examples'][:2])
    ax.text(x, y - 0.2, examples_text, fontsize=7,
           ha='center', va='center', style='italic')
    
    # Issues/considerations
    issues_text = stage['issues'][0]
    ax.text(x, y - 0.9, f"Key: {issues_text}", fontsize=6,
           ha='center', va='center', color='gray')

# Draw connections
connections = [
    ((3.2, 8), (3.8, 8)),  # Raw -> Cleaning
    ((6.2, 8), (6.8, 8)),  # Cleaning -> Feature Eng
    ((9.2, 8), (9.8, 8)),  # Feature Eng -> Encoding
    ((11, 7.4), (11, 5.6)),  # Encoding -> Scaling
    ((9.8, 5), (9.2, 5)),  # Scaling -> Dim Reduction
    ((6.8, 5), (6.2, 5)),  # Dim Reduction -> Validation
    ((3.8, 5), (3.2, 5))   # Validation -> Ready
]

for start, end in connections:
    arrow = FancyArrowPatch(start, end,
                          arrowstyle='->', mutation_scale=20,
                          color='gray', alpha=0.7, linewidth=2)
    ax.add_patch(arrow)

# Add transformation examples with before/after
transformations = [
    {
        'title': 'Scaling Example',
        'pos': (2, 3),
        'before': 'Revenue: 1M-100M\nEmployees: 5-5000',
        'after': 'Revenue: -1.2 to 2.3\nEmployees: -0.8 to 3.1',
        'method': 'StandardScaler'
    },
    {
        'title': 'Encoding Example',
        'pos': (5, 3),
        'before': 'Type: [Product, Service, Platform]',
        'after': 'Type_Product: [1,0,0]\nType_Service: [0,1,0]',
        'method': 'One-Hot Encoding'
    },
    {
        'title': 'Missing Data Example',
        'pos': (8, 3),
        'before': 'Score: [8, NaN, 6, NaN, 9]',
        'after': 'Score: [8, 7.5, 6, 7.5, 9]',
        'method': 'Mean Imputation'
    },
    {
        'title': 'Dimension Reduction',
        'pos': (11, 3),
        'before': '50 features',
        'after': '2 principal components\n(85% variance retained)',
        'method': 'PCA'
    }
]

for trans in transformations:
    x, y = trans['pos']
    
    # Title
    ax.text(x, y + 0.4, trans['title'], fontsize=9, fontweight='bold',
           ha='center', va='center')
    
    # Before box
    before_box = FancyBboxPatch((x - 1.2, y - 0.2), 1.1, 0.5,
                               boxstyle="round,pad=0.02",
                               facecolor='lightcoral', alpha=0.3)
    ax.add_patch(before_box)
    ax.text(x - 0.65, y + 0.05, 'Before:', fontsize=7, fontweight='bold')
    ax.text(x - 0.65, y - 0.15, trans['before'], fontsize=6, ha='center', va='center')
    
    # Arrow
    arrow = FancyArrowPatch((x - 0.05, y + 0.05), (x + 0.05, y + 0.05),
                          arrowstyle='->', mutation_scale=15,
                          color='black', linewidth=1.5)
    ax.add_patch(arrow)
    
    # After box
    after_box = FancyBboxPatch((x + 0.1, y - 0.2), 1.1, 0.5,
                              boxstyle="round,pad=0.02",
                              facecolor='lightgreen', alpha=0.3)
    ax.add_patch(after_box)
    ax.text(x + 0.65, y + 0.05, 'After:', fontsize=7, fontweight='bold')
    ax.text(x + 0.65, y - 0.15, trans['after'], fontsize=6, ha='center', va='center')
    
    # Method
    ax.text(x, y - 0.5, f"({trans['method']})", fontsize=6,
           ha='center', va='center', style='italic', color='blue')

# Add quality checks panel
quality_text = (
    "Quality Checks at Each Stage:\n"
    "* Data types correct\n"
    "* No missing values\n"
    "* Scaled to similar range\n"
    "* No high correlation\n"
    "* Sufficient variance\n"
    "* No data leakage"
)
ax.text(0.5, 6.5, quality_text, fontsize=8,
       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

# Add common pitfalls
pitfalls_text = (
    "Common Pitfalls:\n"
    "X Scaling before split\n"
    "X Leaking test data\n"
    "X Over-engineering features\n"
    "X Ignoring outliers\n"
    "X Wrong imputation\n"
    "X Losing interpretability"
)
ax.text(12.5, 6.5, pitfalls_text, fontsize=8,
       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.7))

# Add pipeline code snippet
code_text = (
    "from sklearn.pipeline import Pipeline\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.decomposition import PCA\n\n"
    "pipeline = Pipeline([\n"
    "    ('scaler', StandardScaler()),\n"
    "    ('pca', PCA(n_components=2)),\n"
    "    ('clustering', KMeans(n_clusters=5))\n"
    "])"
)
ax.text(6.5, 1.5, code_text, fontsize=7, family='monospace',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.3))

# Title and subtitle
ax.text(6.5, 9.5, 'Data Preprocessing Pipeline for Innovation Clustering',
       fontsize=16, fontweight='bold', ha='center')
ax.text(6.5, 9, 'From Raw Data to Clustering-Ready Features',
       fontsize=11, ha='center', style='italic', color='gray')

# Add time estimate
time_text = "Typical Time: 60-80% of project"
ax.text(6.5, 0.5, time_text, fontsize=9, ha='center',
       bbox=dict(boxstyle='round,pad=0.2', facecolor='orange', alpha=0.3))

# Remove axes
ax.set_xlim(0, 13)
ax.set_ylim(0, 10)
ax.axis('off')

plt.tight_layout()

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/preprocessing_pipeline.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/preprocessing_pipeline.png', 
           dpi=150, bbox_inches='tight')

print("Preprocessing pipeline diagram created successfully!")
print("Files saved:")
print("  - charts/preprocessing_pipeline.pdf")
print("  - charts/preprocessing_pipeline.png")