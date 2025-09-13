import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Define the empathy map quadrants
quadrants = {
    'THINK': {'pos': (0, 0.5), 'color': '#3498db'},
    'FEEL': {'pos': (0.5, 0.5), 'color': '#e74c3c'},
    'SAY': {'pos': (0, 0), 'color': '#2ecc71'},
    'DO': {'pos': (0.5, 0), 'color': '#f39c12'}
}

# Draw quadrants
for name, props in quadrants.items():
    rect = patches.Rectangle(props['pos'], 0.5, 0.5,
                            linewidth=2, edgecolor='white',
                            facecolor=props['color'], alpha=0.3)
    ax.add_patch(rect)
    
    # Add quadrant labels
    ax.text(props['pos'][0] + 0.25, props['pos'][1] + 0.45,
           name, fontsize=16, fontweight='bold',
           ha='center', va='center')

# Add center circle for user/cluster
circle = patches.Circle((0.5, 0.5), 0.15, 
                       facecolor='white', edgecolor='black', linewidth=3)
ax.add_patch(circle)
ax.text(0.5, 0.5, 'CLUSTER\nPROFILE', fontsize=12, fontweight='bold',
       ha='center', va='center')

# Add insights for each quadrant based on cluster analysis
insights = {
    'THINK': [
        'Need efficiency',
        'Value time-saving',
        'Compare options',
        'Research features'
    ],
    'FEEL': [
        'Frustrated by complexity',
        'Excited by automation',
        'Anxious about learning',
        'Confident with tools'
    ],
    'SAY': [
        '"Too many clicks"',
        '"Where is that feature?"',
        '"This saves time"',
        '"Finally, it works!"'
    ],
    'DO': [
        'Use shortcuts',
        'Skip tutorials',
        'Share discoveries',
        'Customize settings'
    ]
}

# Add insights to quadrants
y_offsets = {'THINK': 0.75, 'FEEL': 0.75, 'SAY': 0.25, 'DO': 0.25}
x_offsets = {'THINK': 0.25, 'FEEL': 0.75, 'SAY': 0.25, 'DO': 0.75}

for quadrant, items in insights.items():
    y_start = y_offsets[quadrant]
    x_pos = x_offsets[quadrant]
    
    for i, item in enumerate(items):
        y_pos = y_start - i * 0.08
        ax.text(x_pos, y_pos, f'• {item}', fontsize=10,
               ha='center', va='center', 
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', alpha=0.7))

# Add data sources as small labels
sources = {
    'THINK': 'From search queries',
    'FEEL': 'From sentiment analysis',
    'SAY': 'From support tickets',
    'DO': 'From clickstream data'
}

for quadrant, source in sources.items():
    x_pos = x_offsets[quadrant]
    y_pos = 0.05 if quadrant in ['SAY', 'DO'] else 0.95
    ax.text(x_pos, y_pos, source, fontsize=8, style='italic',
           ha='center', va='center', alpha=0.7)

# Remove axes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')

# Add title
plt.title('Empathy Map: Data-Driven User Understanding', 
         fontsize=16, fontweight='bold', pad=20)

# Add subtitle
plt.text(0.5, -0.05, 'Each quadrant filled by analyzing different data sources from user clusters',
        transform=ax.transAxes, ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('empathy_map_clusters.pdf', dpi=300, bbox_inches='tight')
plt.savefig('empathy_map_clusters.png', dpi=150, bbox_inches='tight')
plt.close()

print("Empathy map visualization created!")