#!/usr/bin/env python3
"""
Create Simple Practice Example for BSc Students
Using a relatable scenario without complexity
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches

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

# Title for the whole figure
fig.suptitle('Practice Example: Grouping Students by Study Habits', 
            fontsize=14, fontweight='bold', y=0.98)
fig.text(0.5, 0.95, 'A Simple Clustering Exercise', 
        fontsize=11, ha='center', style='italic', color='gray')

# Panel 1: The Problem
ax1 = plt.subplot(2, 4, 1)
ax1.set_title('The Problem', fontsize=11, fontweight='bold')
ax1.axis('off')

problem_text = [
    'A teacher wants to understand',
    'different study patterns in class.',
    '',
    'Data collected:',
    '• Hours studied per week',
    '• Number of questions asked',
    '',
    '50 students total',
    '',
    'Goal: Find study groups'
]

y_pos = 0.9
for line in problem_text:
    if 'Data' in line or 'Goal' in line:
        weight = 'bold'
        color = colors['mlblue']
    else:
        weight = 'normal'
        color = 'black'
    ax1.text(0.05, y_pos, line, fontsize=9, transform=ax1.transAxes,
            fontweight=weight, color=color)
    y_pos -= 0.09

# Panel 2: The Data (Visual)
ax2 = plt.subplot(2, 4, 2)
ax2.set_title('Student Data', fontsize=11, fontweight='bold')

# Generate realistic student data
np.random.seed(42)
# Group 1: Low study, low questions (struggling students)
group1_hours = np.random.normal(3, 0.8, 15)
group1_questions = np.random.normal(2, 0.5, 15)

# Group 2: High study, high questions (engaged students)
group2_hours = np.random.normal(8, 1.0, 20)
group2_questions = np.random.normal(8, 1.2, 20)

# Group 3: Medium study, low questions (independent learners)
group3_hours = np.random.normal(6, 0.8, 15)
group3_questions = np.random.normal(3, 0.7, 15)

# Combine all data
all_hours = np.concatenate([group1_hours, group2_hours, group3_hours])
all_questions = np.concatenate([group1_questions, group2_questions, group3_questions])
X = np.column_stack([all_hours, all_questions])

# Plot raw data
ax2.scatter(all_hours, all_questions, c='gray', alpha=0.6, s=40)
ax2.set_xlabel('Hours Studied/Week', fontsize=9)
ax2.set_ylabel('Questions Asked/Week', fontsize=9)
ax2.set_xlim(0, 12)
ax2.set_ylim(0, 12)
ax2.grid(True, alpha=0.3)
ax2.text(0.5, -0.18, 'Each dot = 1 student', fontsize=8, 
        ha='center', transform=ax2.transAxes, style='italic')

# Panel 3: Apply Clustering
ax3 = plt.subplot(2, 4, 3)
ax3.set_title('After Clustering', fontsize=11, fontweight='bold')

# Apply K-means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Plot clustered data
cluster_colors = [colors['mlred'], colors['mlgreen'], colors['mlblue']]
for i in range(3):
    mask = labels == i
    ax3.scatter(X[mask, 0], X[mask, 1], c=cluster_colors[i], 
               alpha=0.6, s=40, label=f'Group {i+1}')

ax3.set_xlabel('Hours Studied/Week', fontsize=9)
ax3.set_ylabel('Questions Asked/Week', fontsize=9)
ax3.set_xlim(0, 12)
ax3.set_ylim(0, 12)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left', fontsize=8)

# Panel 4: Interpret Results
ax4 = plt.subplot(2, 4, 4)
ax4.set_title('What We Found', fontsize=11, fontweight='bold')
ax4.axis('off')

# Calculate cluster characteristics
cluster_stats = []
for i in range(3):
    mask = labels == i
    avg_hours = X[mask, 0].mean()
    avg_questions = X[mask, 1].mean()
    cluster_stats.append((avg_hours, avg_questions))

# Sort by total engagement (hours + questions)
cluster_order = sorted(range(3), key=lambda i: sum(cluster_stats[i]))

interpretations = [
    (colors['mlred'], 'Group 1:', 'Need extra help', '• Low hours, few questions'),
    (colors['mlblue'], 'Group 2:', 'Independent learners', '• Good hours, few questions'),
    (colors['mlgreen'], 'Group 3:', 'Highly engaged', '• Many hours, many questions')
]

y_pos = 0.85
for color, title, desc, detail in interpretations:
    ax4.text(0.05, y_pos, title, fontsize=10, fontweight='bold', 
            color=color, transform=ax4.transAxes)
    ax4.text(0.05, y_pos - 0.08, desc, fontsize=9, 
            transform=ax4.transAxes)
    ax4.text(0.05, y_pos - 0.15, detail, fontsize=8, 
            style='italic', color='gray', transform=ax4.transAxes)
    y_pos -= 0.28

# Panel 5: Your Turn - Step 1
ax5 = plt.subplot(2, 4, 5)
ax5.set_title('Your Turn: Step 1', fontsize=11, fontweight='bold')
ax5.axis('off')

step1_text = [
    'Load the data:',
    '',
    'import pandas as pd',
    'data = pd.read_csv("students.csv")',
    '',
    'Look at it:',
    'print(data.head())',
    '',
    'You should see:',
    '  hours  questions',
    '0  3.2    2.1',
    '1  8.5    7.9',
    '...'
]

y_pos = 0.9
for line in step1_text:
    if line.startswith('import') or line.startswith('data') or line.startswith('print'):
        family = 'monospace'
        size = 8
        color = colors['mlblue']
    elif 'Load' in line or 'Look' in line or 'You should' in line:
        family = 'sans-serif'
        size = 9
        color = 'black'
    else:
        family = 'monospace'
        size = 8
        color = 'gray'
    
    ax5.text(0.05, y_pos, line, fontsize=size, transform=ax5.transAxes,
            family=family, color=color)
    y_pos -= 0.065

# Panel 6: Your Turn - Step 2
ax6 = plt.subplot(2, 4, 6)
ax6.set_title('Your Turn: Step 2', fontsize=11, fontweight='bold')
ax6.axis('off')

step2_text = [
    'Prepare the data:',
    '',
    'from sklearn.preprocessing import StandardScaler',
    'scaler = StandardScaler()',
    'X_scaled = scaler.fit_transform(data)',
    '',
    'Why scale?',
    'Makes features comparable',
    '',
    'Apply clustering:',
    'from sklearn.cluster import KMeans',
    'kmeans = KMeans(n_clusters=3)',
    'labels = kmeans.fit_predict(X_scaled)'
]

y_pos = 0.9
for line in step2_text:
    if line.startswith('from') or line.startswith('scaler') or line.startswith('X_scaled') or line.startswith('kmeans') or line.startswith('labels'):
        family = 'monospace'
        size = 8
        color = colors['mlblue']
    elif 'Prepare' in line or 'Why' in line or 'Apply' in line:
        family = 'sans-serif'
        size = 9
        color = 'black'
    else:
        family = 'sans-serif'
        size = 8
        color = 'gray'
    
    ax6.text(0.05, y_pos, line, fontsize=size, transform=ax6.transAxes,
            family=family, color=color)
    y_pos -= 0.065

# Panel 7: Your Turn - Step 3
ax7 = plt.subplot(2, 4, 7)
ax7.set_title('Your Turn: Step 3', fontsize=11, fontweight='bold')
ax7.axis('off')

step3_text = [
    'See the results:',
    '',
    'import matplotlib.pyplot as plt',
    'plt.scatter(data["hours"], ',
    '           data["questions"],',
    '           c=labels)',
    'plt.xlabel("Hours")',
    'plt.ylabel("Questions")',
    'plt.show()',
    '',
    'Count groups:',
    'for i in range(3):',
    '    count = (labels == i).sum()',
    '    print(f"Group {i}: {count} students")'
]

y_pos = 0.9
for line in step3_text:
    if 'import' in line or 'plt.' in line or 'c=' in line or 'for' in line or 'count' in line or 'print' in line:
        family = 'monospace'
        size = 8
        color = colors['mlblue']
    elif 'See' in line or 'Count' in line:
        family = 'sans-serif'
        size = 9
        color = 'black'
    else:
        family = 'monospace'
        size = 8
        color = 'gray'
    
    ax7.text(0.05, y_pos, line, fontsize=size, transform=ax7.transAxes,
            family=family, color=color)
    y_pos -= 0.065

# Panel 8: Check Your Work
ax8 = plt.subplot(2, 4, 8)
ax8.set_title('Check Your Work', fontsize=11, fontweight='bold')
ax8.axis('off')

checklist = [
    '✓ Did you find 3 groups?',
    '✓ Are groups visually separated?',
    '✓ Do groups make sense?',
    '',
    'What to look for:',
    '• Clear differences between groups',
    '• Similar students in same group',
    '• Groups tell a story',
    '',
    'Next: Try with K=2 or K=4',
    'Which is better? Why?'
]

y_pos = 0.9
for line in checklist:
    if '✓' in line:
        color = colors['mlgreen']
        weight = 'normal'
    elif 'What' in line or 'Next' in line:
        color = colors['mlblue']
        weight = 'bold'
    else:
        color = 'black'
        weight = 'normal'
    
    ax8.text(0.05, y_pos, line, fontsize=9, transform=ax8.transAxes,
            color=color, fontweight=weight)
    y_pos -= 0.08

# Footer with tips
tips_text = (
    'Tips: Start simple → Check results → Try different settings → Pick what makes sense'
)
fig.text(0.5, 0.02, tips_text, fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 0.94])

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/simple_practice_example.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/simple_practice_example.png', 
           dpi=150, bbox_inches='tight')

print("Simple practice example created successfully!")
print("Files saved:")
print("  - charts/simple_practice_example.pdf")
print("  - charts/simple_practice_example.png")