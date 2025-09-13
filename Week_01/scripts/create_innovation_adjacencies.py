#!/usr/bin/env python3
"""
Create Innovation Adjacencies Network for Week 1 Part 3
Shows connections and relationships between innovation opportunities
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import matplotlib.patches as mpatches

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Create a graph for innovation adjacencies
G = nx.Graph()

# Define innovation nodes (core innovations)
core_innovations = {
    'AI_Platform': {'pos': (5, 5), 'color': '#1f77b4', 'size': 3000, 'label': 'AI\nPlatform'},
    'IoT_Sensors': {'pos': (2, 7), 'color': '#ff7f0e', 'size': 2500, 'label': 'IoT\nSensors'},
    'Cloud_Analytics': {'pos': (8, 7), 'color': '#2ca02c', 'size': 2800, 'label': 'Cloud\nAnalytics'},
    'Mobile_App': {'pos': (2, 3), 'color': '#d62728', 'size': 2200, 'label': 'Mobile\nApp'},
    'Blockchain': {'pos': (8, 3), 'color': '#9467bd', 'size': 2000, 'label': 'Blockchain\nSecurity'},
}

# Add nodes to graph
for node, attrs in core_innovations.items():
    G.add_node(node, **attrs)

# Define adjacent innovations (opportunities)
adjacent_innovations = {
    'Smart_Home': {'pos': (3.5, 6), 'color': '#17becf', 'size': 1500, 'label': 'Smart\nHome'},
    'Predictive_Maint': {'pos': (6.5, 6), 'color': '#bcbd22', 'size': 1400, 'label': 'Predictive\nMaintenance'},
    'Edge_Computing': {'pos': (5, 8), 'color': '#e377c2', 'size': 1600, 'label': 'Edge\nComputing'},
    'AR_Interface': {'pos': (3.5, 4), 'color': '#7f7f7f', 'size': 1300, 'label': 'AR\nInterface'},
    'Digital_Twin': {'pos': (6.5, 4), 'color': '#8c564b', 'size': 1700, 'label': 'Digital\nTwin'},
    'Federated_Learn': {'pos': (5, 2), 'color': '#1f77b4', 'size': 1200, 'label': 'Federated\nLearning'},
    'Voice_Assistant': {'pos': (1, 5), 'color': '#ff7f0e', 'size': 1100, 'label': 'Voice\nAssistant'},
    'Supply_Chain': {'pos': (9, 5), 'color': '#2ca02c', 'size': 1800, 'label': 'Supply\nChain'},
}

# Add adjacent nodes
for node, attrs in adjacent_innovations.items():
    G.add_node(node, **attrs)

# Define innovation connections (weighted by synergy potential)
innovation_edges = [
    # Strong synergies (thick lines)
    ('AI_Platform', 'Cloud_Analytics', 0.9),
    ('AI_Platform', 'IoT_Sensors', 0.85),
    ('AI_Platform', 'Mobile_App', 0.8),
    ('Cloud_Analytics', 'IoT_Sensors', 0.88),
    ('Mobile_App', 'Blockchain', 0.75),
    
    # Adjacent opportunities
    ('AI_Platform', 'Smart_Home', 0.7),
    ('AI_Platform', 'Predictive_Maint', 0.75),
    ('AI_Platform', 'Digital_Twin', 0.8),
    ('IoT_Sensors', 'Smart_Home', 0.85),
    ('IoT_Sensors', 'Edge_Computing', 0.9),
    ('Cloud_Analytics', 'Predictive_Maint', 0.85),
    ('Cloud_Analytics', 'Edge_Computing', 0.8),
    ('Cloud_Analytics', 'Supply_Chain', 0.82),
    ('Mobile_App', 'AR_Interface', 0.78),
    ('Mobile_App', 'Voice_Assistant', 0.72),
    ('Blockchain', 'Digital_Twin', 0.7),
    ('Blockchain', 'Supply_Chain', 0.88),
    ('Blockchain', 'Federated_Learn', 0.65),
    
    # Cross-connections
    ('Smart_Home', 'Voice_Assistant', 0.68),
    ('Predictive_Maint', 'Digital_Twin', 0.82),
    ('Edge_Computing', 'Federated_Learn', 0.6),
    ('AR_Interface', 'Digital_Twin', 0.65),
]

# Add edges with weights
for source, target, weight in innovation_edges:
    G.add_edge(source, target, weight=weight)

# Extract positions
pos = nx.get_node_attributes(G, 'pos')

# Draw edges with varying thickness based on weight
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]

# Normalize weights for visualization
min_weight = min(weights)
max_weight = max(weights)
normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]

# Draw edges
for (u, v), weight, norm_weight in zip(edges, weights, normalized_weights):
    # Color based on strength
    if weight >= 0.85:
        edge_color = 'darkgreen'
        style = '-'
        alpha = 0.7
    elif weight >= 0.7:
        edge_color = 'orange'
        style = '-'
        alpha = 0.5
    else:
        edge_color = 'gray'
        style = '--'
        alpha = 0.3
    
    # Draw edge
    nx.draw_networkx_edges(G, pos, [(u, v)], 
                          width=2 + norm_weight * 4,
                          edge_color=edge_color,
                          style=style,
                          alpha=alpha,
                          ax=ax)

# Draw nodes
node_colors = [G.nodes[node]['color'] for node in G.nodes()]
node_sizes = [G.nodes[node]['size'] for node in G.nodes()]

nx.draw_networkx_nodes(G, pos, 
                      node_color=node_colors,
                      node_size=node_sizes,
                      alpha=0.8,
                      edgecolors='white',
                      linewidths=2,
                      ax=ax)

# Add labels
labels = {node: G.nodes[node]['label'] for node in G.nodes()}
for node, (x, y) in pos.items():
    # Core innovations get bold labels
    if node in core_innovations:
        ax.text(x, y, labels[node], 
               fontsize=11, fontweight='bold',
               ha='center', va='center',
               color='white')
    else:
        ax.text(x, y, labels[node], 
               fontsize=9, fontweight='normal',
               ha='center', va='center',
               color='white')

# Add innovation zones
# Zone 1: High Integration Area
zone1 = FancyBboxPatch((4, 4.5), 2, 2,
                       boxstyle="round,pad=0.1",
                       facecolor='yellow', alpha=0.1,
                       edgecolor='goldenrod', linewidth=2,
                       linestyle='--')
ax.add_patch(zone1)
ax.text(5, 5.8, 'High Integration\nZone', fontsize=10, 
       fontweight='bold', ha='center', color='goldenrod')

# Zone 2: Emerging Tech Area
zone2 = FancyBboxPatch((4, 1.5), 2, 1.5,
                       boxstyle="round,pad=0.1",
                       facecolor='lightblue', alpha=0.1,
                       edgecolor='blue', linewidth=2,
                       linestyle='--')
ax.add_patch(zone2)
ax.text(5, 1, 'Emerging\nTechnologies', fontsize=10,
       fontweight='bold', ha='center', color='blue')

# Add innovation metrics
metrics_text = (
    "Innovation Synergy Metrics:\n"
    "• 5 Core Technologies\n"
    "• 8 Adjacent Opportunities\n"
    "• 18 Strong Connections (>0.7)\n"
    "• 3 Innovation Clusters"
)
ax.text(0.2, 8.5, metrics_text, fontsize=10,
       bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='lightyellow', alpha=0.9))

# Add strategic insights
strategy_text = (
    "Strategic Adjacencies:\n"
    "1. AI + IoT + Cloud = Smart Systems\n"
    "2. Blockchain + Supply = Trust Layer\n"
    "3. Mobile + AR = New Interfaces"
)
ax.text(7, 1, strategy_text, fontsize=10,
       bbox=dict(boxstyle='round,pad=0.5',
                facecolor='lightgreen', alpha=0.9))

# Set title and labels
ax.set_title('Innovation Adjacency Network\nDiscovering Synergies Through Connection Analysis', 
            fontsize=16, fontweight='bold', pad=20)

# Remove axes
ax.set_xlim(0, 10)
ax.set_ylim(0, 9)
ax.axis('off')

# Add legend
legend_elements = [
    mpatches.Patch(color='darkgreen', alpha=0.7, label='Strong Synergy (>0.85)'),
    mpatches.Patch(color='orange', alpha=0.5, label='Good Synergy (0.7-0.85)'),
    mpatches.Patch(color='gray', alpha=0.3, label='Potential Synergy (<0.7)'),
    mpatches.Circle((0, 0), 0.1, color='#1f77b4', alpha=0.8, label='Core Innovation'),
    mpatches.Circle((0, 0), 0.08, color='#17becf', alpha=0.8, label='Adjacent Opportunity')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)

plt.tight_layout()

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/innovation_adjacencies.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/innovation_adjacencies.png', 
           dpi=150, bbox_inches='tight')

print("Innovation adjacencies network created successfully!")
print("Files saved:")
print("  - charts/innovation_adjacencies.pdf")
print("  - charts/innovation_adjacencies.png")