#!/usr/bin/env python3
"""
Create Parameter Tuning Guidelines for Week 1
Shows recommended parameter ranges and tuning strategies for clustering
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Algorithm parameters and recommendations
algorithms = {
    'K-Means': {
        'params': {
            'n_clusters (K)': {'range': '2-10', 'default': '3-5', 'method': 'Elbow/Silhouette'},
            'init': {'range': "['k-means++', 'random']", 'default': 'k-means++', 'method': 'Always k-means++'},
            'n_init': {'range': '10-100', 'default': '10', 'method': 'More for stability'},
            'max_iter': {'range': '100-1000', 'default': '300', 'method': 'Increase if no convergence'},
            'tol': {'range': '1e-6 to 1e-2', 'default': '1e-4', 'method': 'Smaller for precision'}
        },
        'color': '#3498db'
    },
    'DBSCAN': {
        'params': {
            'eps': {'range': '0.01-2.0', 'default': '0.5', 'method': 'k-distance plot'},
            'min_samples': {'range': '3-20', 'default': '2*dims', 'method': 'Domain knowledge'},
            'metric': {'range': "['euclidean', 'manhattan']", 'default': 'euclidean', 'method': 'Data dependent'},
            'algorithm': {'range': "['auto', 'ball_tree']", 'default': 'auto', 'method': 'Auto is fine'},
            'leaf_size': {'range': '10-50', 'default': '30', 'method': 'Memory vs speed'}
        },
        'color': '#2ecc71'
    },
    'GMM': {
        'params': {
            'n_components': {'range': '2-10', 'default': '3-5', 'method': 'BIC/AIC'},
            'covariance_type': {'range': "['full', 'diag', 'spherical']", 'default': 'full', 'method': 'Start full, simplify'},
            'max_iter': {'range': '50-500', 'default': '100', 'method': 'Monitor convergence'},
            'n_init': {'range': '1-10', 'default': '1', 'method': 'More for stability'},
            'init_params': {'range': "['kmeans', 'random']", 'default': 'kmeans', 'method': 'kmeans faster'}
        },
        'color': '#9b59b6'
    }
}

# Plot parameter tables for each algorithm
for idx, (algo_name, algo_info) in enumerate(algorithms.items()):
    ax = axes[0, idx]
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, algo_name, fontsize=14, fontweight='bold',
           ha='center', transform=ax.transAxes, color=algo_info['color'])
    
    # Parameter table
    y_start = 0.85
    row_height = 0.15
    
    # Headers
    headers = ['Parameter', 'Range', 'Default', 'Tuning Method']
    header_positions = [0.05, 0.25, 0.45, 0.65]
    
    for i, header in enumerate(headers):
        ax.text(header_positions[i], y_start, header, fontsize=10, 
               fontweight='bold', transform=ax.transAxes)
    
    # Draw header line
    ax.plot([0.02, 0.98], [y_start - 0.03, y_start - 0.03], 
           'k-', linewidth=1, transform=ax.transAxes)
    
    # Parameters
    y_pos = y_start - 0.05
    for param_name, param_info in algo_info['params'].items():
        y_pos -= row_height
        
        # Parameter name
        ax.text(0.05, y_pos, param_name, fontsize=9, 
               transform=ax.transAxes, fontweight='bold')
        
        # Range
        ax.text(0.25, y_pos, param_info['range'], fontsize=8,
               transform=ax.transAxes, family='monospace')
        
        # Default
        ax.text(0.45, y_pos, str(param_info['default']), fontsize=8,
               transform=ax.transAxes, color='darkgreen')
        
        # Method
        ax.text(0.65, y_pos, param_info['method'], fontsize=8,
               transform=ax.transAxes, style='italic')
        
        # Alternate row coloring
        if list(algo_info['params'].keys()).index(param_name) % 2 == 0:
            rect = Rectangle((0.02, y_pos - 0.02), 0.96, row_height - 0.02,
                           facecolor=algo_info['color'], alpha=0.1,
                           transform=ax.transAxes)
            ax.add_patch(rect)

# Tuning strategies visualization
ax = axes[1, 0]
ax.axis('off')

strategies = {
    'Grid Search': {
        'pros': ['Exhaustive', 'Reproducible', 'Simple'],
        'cons': ['Slow', 'Curse of dimensionality'],
        'when': 'Small parameter space',
        'color': '#e74c3c'
    },
    'Random Search': {
        'pros': ['Faster', 'Better for many params', 'Parallelizable'],
        'cons': ['May miss optimum', 'Not reproducible'],
        'when': 'Large parameter space',
        'color': '#f39c12'
    },
    'Bayesian Opt': {
        'pros': ['Efficient', 'Learns from history', 'Fewer iterations'],
        'cons': ['Complex', 'Overhead for simple problems'],
        'when': 'Expensive evaluations',
        'color': '#3498db'
    }
}

y_pos = 0.9
for strategy_name, strategy_info in strategies.items():
    # Strategy name
    ax.text(0.05, y_pos, strategy_name, fontsize=11, fontweight='bold',
           transform=ax.transAxes, color=strategy_info['color'])
    
    # Pros
    pros_text = 'Pros: ' + ', '.join(strategy_info['pros'])
    ax.text(0.05, y_pos - 0.05, pros_text, fontsize=8,
           transform=ax.transAxes, color='darkgreen')
    
    # Cons
    cons_text = 'Cons: ' + ', '.join(strategy_info['cons'])
    ax.text(0.05, y_pos - 0.1, cons_text, fontsize=8,
           transform=ax.transAxes, color='darkred')
    
    # When to use
    ax.text(0.05, y_pos - 0.15, f"Use when: {strategy_info['when']}", fontsize=8,
           transform=ax.transAxes, style='italic')
    
    y_pos -= 0.25

ax.text(0.5, 0.98, 'Tuning Strategies', fontsize=12, fontweight='bold',
       ha='center', transform=ax.transAxes)

# Validation metrics for tuning
ax = axes[1, 1]
ax.axis('off')

metrics_info = [
    ['Metric', 'Range', 'Interpretation', 'Use For'],
    ['Silhouette', '[-1, 1]', 'Higher is better', 'General quality'],
    ['Davies-Bouldin', '[0, ∞)', 'Lower is better', 'Cluster separation'],
    ['Calinski-Harabasz', '[0, ∞)', 'Higher is better', 'Dense clusters'],
    ['Inertia', '[0, ∞)', 'Lower is better', 'K-means only'],
    ['BIC/AIC', '(-∞, ∞)', 'Lower is better', 'GMM selection']
]

ax.text(0.5, 0.95, 'Validation Metrics', fontsize=12, fontweight='bold',
       ha='center', transform=ax.transAxes)

# Draw metrics table
for i, row in enumerate(metrics_info):
    y_pos = 0.85 - i * 0.12
    for j, cell in enumerate(row):
        x_pos = 0.05 + j * 0.23
        
        if i == 0:  # Header
            ax.text(x_pos, y_pos, cell, fontsize=9, fontweight='bold',
                   transform=ax.transAxes)
            ax.plot([0.02, 0.98], [y_pos - 0.02, y_pos - 0.02],
                   'k-', linewidth=1, transform=ax.transAxes)
        else:
            if j == 0:  # Metric name
                ax.text(x_pos, y_pos, cell, fontsize=9, fontweight='bold',
                       transform=ax.transAxes)
            elif j == 2:  # Interpretation
                color = 'darkgreen' if 'better' in cell else 'black'
                ax.text(x_pos, y_pos, cell, fontsize=8,
                       transform=ax.transAxes, color=color)
            else:
                ax.text(x_pos, y_pos, cell, fontsize=8,
                       transform=ax.transAxes)

# Best practices and tips
ax = axes[1, 2]
ax.axis('off')

ax.text(0.5, 0.95, 'Tuning Best Practices', fontsize=12, fontweight='bold',
       ha='center', transform=ax.transAxes)

practices = [
    "1. Start with defaults, then tune",
    "2. Use cross-validation when possible",
    "3. Consider computational budget",
    "4. Log all experiments",
    "5. Visualize parameter effects",
    "6. Use domain knowledge",
    "7. Check stability across runs",
    "8. Don't overfit to metrics"
]

y_pos = 0.85
for practice in practices:
    ax.text(0.05, y_pos, practice, fontsize=9, transform=ax.transAxes)
    y_pos -= 0.08
    
# Add important note box
note_text = (
    "IMPORTANT:\n"
    "No metric is perfect!\n"
    "Always validate with:\n"
    "• Visual inspection\n"
    "• Domain expertise\n"
    "• Business goals"
)
ax.text(0.5, 0.25, note_text, fontsize=9, transform=ax.transAxes,
       ha='center', bbox=dict(boxstyle='round,pad=0.5', 
                             facecolor='lightyellow', alpha=0.9))

# Overall title
fig.suptitle('Clustering Parameter Tuning Guidelines', 
            fontsize=16, fontweight='bold', y=0.98)
fig.text(0.5, 0.94, 'Recommended Ranges, Methods, and Best Practices',
        fontsize=11, ha='center', style='italic', color='gray')

# Add workflow diagram at bottom
workflow_text = (
    "TUNING WORKFLOW: "
    "1. Default Parameters → 2. Coarse Grid Search → 3. Fine-tune Best Region → "
    "4. Validate Stability → 5. Production Settings"
)
fig.text(0.5, 0.02, workflow_text, fontsize=9, ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))

plt.tight_layout(rect=[0, 0.04, 1, 0.93])

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/parameter_tuning_guide.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/parameter_tuning_guide.png', 
           dpi=150, bbox_inches='tight')

print("Parameter tuning guidelines created successfully!")
print("Files saved:")
print("  - charts/parameter_tuning_guide.pdf")
print("  - charts/parameter_tuning_guide.png")