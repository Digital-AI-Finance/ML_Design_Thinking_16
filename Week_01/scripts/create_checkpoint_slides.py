#!/usr/bin/env python3
"""
Create Knowledge Checkpoint Visualizations for Week 1
Interactive-style checkpoint questions and progress indicators
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

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

# Create figure with 3 checkpoint slides
fig = plt.figure(figsize=(16, 10))

# Checkpoint 1: After Part 1 (Foundation)
ax1 = plt.subplot(1, 3, 1)
ax1.axis('off')

# Title and progress
ax1.text(0.5, 0.95, 'Knowledge Check: Part 1', fontsize=14, fontweight='bold',
        ha='center', transform=ax1.transAxes, color=colors['mlblue'])
ax1.text(0.5, 0.91, 'Innovation Discovery Foundation', fontsize=10,
        ha='center', transform=ax1.transAxes, style='italic', color='gray')

# Progress bar
progress_bg = Rectangle((0.1, 0.85), 0.8, 0.03, 
                        facecolor=colors['light'], edgecolor=colors['dark'])
ax1.add_patch(progress_bg)
progress_fill = Rectangle((0.1, 0.85), 0.8 * 0.33, 0.03,
                         facecolor=colors['mlgreen'], edgecolor=None)
ax1.add_patch(progress_fill)
ax1.text(0.5, 0.865, '1 of 3 Parts Complete', fontsize=8,
        ha='center', transform=ax1.transAxes, color='white', fontweight='bold')

# Questions
questions1 = [
    {
        'q': 'Q1: What is the main goal of clustering in innovation?',
        'options': [
            'A) To reduce data size',
            'B) To discover hidden patterns',
            'C) To predict outcomes',
            'D) To clean data'
        ],
        'correct': 1
    },
    {
        'q': 'Q2: Which metric measures cluster cohesion?',
        'options': [
            'A) Accuracy',
            'B) Precision',
            'C) Silhouette Score',
            'D) F1 Score'
        ],
        'correct': 2
    },
    {
        'q': 'Q3: Empathy mapping helps identify:',
        'options': [
            'A) Technical requirements',
            'B) User pain points',
            'C) System architecture',
            'D) Database schema'
        ],
        'correct': 1
    }
]

y_pos = 0.75
for i, q_data in enumerate(questions1):
    # Question
    ax1.text(0.05, y_pos, q_data['q'], fontsize=9, fontweight='bold',
            transform=ax1.transAxes)
    
    # Options
    for j, option in enumerate(q_data['options']):
        y_opt = y_pos - 0.03 - (j * 0.025)
        color = colors['mlgreen'] if j == q_data['correct'] else colors['dark']
        symbol = '✓' if j == q_data['correct'] else '○'
        ax1.text(0.07, y_opt, f'{symbol} {option}', fontsize=8,
                transform=ax1.transAxes, color=color)
    
    y_pos -= 0.2

# Key Concepts Box
key_concepts = [
    "• Unsupervised learning",
    "• Pattern discovery",
    "• User segmentation", 
    "• Innovation opportunities"
]
ax1.text(0.5, 0.12, 'Key Concepts Covered:', fontsize=9, fontweight='bold',
        ha='center', transform=ax1.transAxes)
for i, concept in enumerate(key_concepts):
    ax1.text(0.5, 0.08 - i*0.02, concept, fontsize=8,
            ha='center', transform=ax1.transAxes, color=colors['mlblue'])

# Checkpoint 2: After Part 2 (Technical)
ax2 = plt.subplot(1, 3, 2)
ax2.axis('off')

ax2.text(0.5, 0.95, 'Knowledge Check: Part 2', fontsize=14, fontweight='bold',
        ha='center', transform=ax2.transAxes, color=colors['mlorange'])
ax2.text(0.5, 0.91, 'Clustering Algorithms Deep Dive', fontsize=10,
        ha='center', transform=ax2.transAxes, style='italic', color='gray')

# Progress bar
progress_bg2 = Rectangle((0.1, 0.85), 0.8, 0.03,
                         facecolor=colors['light'], edgecolor=colors['dark'])
ax2.add_patch(progress_bg2)
progress_fill2 = Rectangle((0.1, 0.85), 0.8 * 0.66, 0.03,
                          facecolor=colors['mlorange'], edgecolor=None)
ax2.add_patch(progress_fill2)
ax2.text(0.5, 0.865, '2 of 3 Parts Complete', fontsize=8,
        ha='center', transform=ax2.transAxes, color='white', fontweight='bold')

# Questions
questions2 = [
    {
        'q': 'Q1: K-means time complexity is:',
        'options': [
            'A) O(n)',
            'B) O(n log n)',
            'C) O(n*k*i*d)',
            'D) O(n²)'
        ],
        'correct': 2
    },
    {
        'q': 'Q2: DBSCAN is best for:',
        'options': [
            'A) Spherical clusters',
            'B) Arbitrary shapes',
            'C) Fixed K clusters',
            'D) Linear data'
        ],
        'correct': 1
    },
    {
        'q': 'Q3: GMM provides:',
        'options': [
            'A) Hard clustering',
            'B) Soft clustering',
            'C) No clustering',
            'D) Random clustering'
        ],
        'correct': 1
    }
]

y_pos = 0.75
for i, q_data in enumerate(questions2):
    ax2.text(0.05, y_pos, q_data['q'], fontsize=9, fontweight='bold',
            transform=ax2.transAxes)
    
    for j, option in enumerate(q_data['options']):
        y_opt = y_pos - 0.03 - (j * 0.025)
        color = colors['mlgreen'] if j == q_data['correct'] else colors['dark']
        symbol = '✓' if j == q_data['correct'] else '○'
        ax2.text(0.07, y_opt, f'{symbol} {option}', fontsize=8,
                transform=ax2.transAxes, color=color)
    
    y_pos -= 0.2

# Algorithm Comparison Table
ax2.text(0.5, 0.15, 'Algorithm Quick Reference:', fontsize=9, fontweight='bold',
        ha='center', transform=ax2.transAxes)

algorithms = [
    ['Algorithm', 'Best For', 'Weakness'],
    ['K-means', 'Speed', 'Assumes spherical'],
    ['DBSCAN', 'Shapes', 'Parameter sensitive'],
    ['GMM', 'Overlap', 'Computationally heavy']
]

for i, row in enumerate(algorithms):
    for j, cell in enumerate(row):
        x_pos = 0.2 + j * 0.3
        y_table = 0.10 - i * 0.025
        weight = 'bold' if i == 0 else 'normal'
        size = 8 if i == 0 else 7
        ax2.text(x_pos, y_table, cell, fontsize=size, fontweight=weight,
                transform=ax2.transAxes, ha='center')

# Checkpoint 3: After Part 3 (Design Integration)
ax3 = plt.subplot(1, 3, 3)
ax3.axis('off')

ax3.text(0.5, 0.95, 'Knowledge Check: Part 3', fontsize=14, fontweight='bold',
        ha='center', transform=ax3.transAxes, color=colors['mlpurple'])
ax3.text(0.5, 0.91, 'Human-Centered Application', fontsize=10,
        ha='center', transform=ax3.transAxes, style='italic', color='gray')

# Progress bar
progress_bg3 = Rectangle((0.1, 0.85), 0.8, 0.03,
                         facecolor=colors['light'], edgecolor=colors['dark'])
ax3.add_patch(progress_bg3)
progress_fill3 = Rectangle((0.1, 0.85), 0.8, 0.03,
                          facecolor=colors['mlpurple'], edgecolor=None)
ax3.add_patch(progress_fill3)
ax3.text(0.5, 0.865, '3 of 3 Parts Complete!', fontsize=8,
        ha='center', transform=ax3.transAxes, color='white', fontweight='bold')

# Questions
questions3 = [
    {
        'q': 'Q1: User archetypes are created from:',
        'options': [
            'A) Random assignment',
            'B) Cluster analysis',
            'C) Manual labeling',
            'D) Predictions'
        ],
        'correct': 1
    },
    {
        'q': 'Q2: Innovation opportunities emerge from:',
        'options': [
            'A) Cluster gaps',
            'B) Dense regions',
            'C) Outliers',
            'D) All of above'
        ],
        'correct': 3
    },
    {
        'q': 'Q3: Validation should include:',
        'options': [
            'A) Only metrics',
            'B) Domain experts',
            'C) Random checks',
            'D) Code review'
        ],
        'correct': 1
    }
]

y_pos = 0.75
for i, q_data in enumerate(questions3):
    ax3.text(0.05, y_pos, q_data['q'], fontsize=9, fontweight='bold',
            transform=ax3.transAxes)
    
    for j, option in enumerate(q_data['options']):
        y_opt = y_pos - 0.03 - (j * 0.025)
        color = colors['mlgreen'] if j == q_data['correct'] else colors['dark']
        symbol = '✓' if j == q_data['correct'] else '○'
        ax3.text(0.07, y_opt, f'{symbol} {option}', fontsize=8,
                transform=ax3.transAxes, color=color)
    
    y_pos -= 0.2

# Ready for Practice Box
practice_box = FancyBboxPatch((0.1, 0.02), 0.8, 0.12,
                             boxstyle="round,pad=0.02",
                             facecolor=colors['mlgreen'], alpha=0.2,
                             edgecolor=colors['mlgreen'], linewidth=2,
                             transform=ax3.transAxes)
ax3.add_patch(practice_box)

ax3.text(0.5, 0.10, 'Ready for Practice!', fontsize=10, fontweight='bold',
        ha='center', transform=ax3.transAxes, color=colors['mlgreen'])
ax3.text(0.5, 0.07, 'You now have the knowledge to:', fontsize=8,
        ha='center', transform=ax3.transAxes)
ax3.text(0.5, 0.04, '1. Choose algorithms  2. Apply clustering  3. Extract insights',
        fontsize=7, ha='center', transform=ax3.transAxes, style='italic')

# Overall title
fig.suptitle('Interactive Knowledge Checkpoints', fontsize=16, fontweight='bold', y=0.98)
fig.text(0.5, 0.01, 'Green checkmarks indicate correct answers | Use these to review key concepts',
        fontsize=8, ha='center', style='italic', color='gray')

plt.tight_layout(rect=[0, 0.02, 1, 0.96])

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/checkpoint_slides.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/checkpoint_slides.png', 
           dpi=150, bbox_inches='tight')

print("Knowledge checkpoint slides created successfully!")
print("Files saved:")
print("  - charts/checkpoint_slides.pdf")
print("  - charts/checkpoint_slides.png")
print("\nCheckpoint Summary:")
print("  - 3 checkpoints (after each major part)")
print("  - 3 questions per checkpoint")
print("  - Progress indicators included")
print("  - Correct answers marked with green checkmarks")