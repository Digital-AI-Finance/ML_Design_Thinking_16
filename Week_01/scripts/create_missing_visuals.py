#!/usr/bin/env python3
"""
Create missing visual elements for discovery-based learning materials.
Visual 5: Theory Map - Flowchart connecting discoveries to formal concepts
Visual 8: Math Icons - Visual equation representations
Visual 9: Personal Framework - Blank synthesis template
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.patches import ConnectionPatch
import matplotlib.lines as mlines

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define color palette
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e',
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'gray': '#7f7f7f',
    'lightgray': '#e0e0e0'
}

def create_theory_map():
    """Visual 5: Theory Map - Connecting student discoveries to formal concepts"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Title
    ax.text(0.5, 0.95, 'Your Discovery Journey → Formal Theory', 
            fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Left column: Student discoveries
    discoveries = [
        "Dots close together\nbelong in groups",
        "Different ways to\nmeasure 'close'",
        "Groups have\ncenter points",
        "Some dots don't\nfit any group",
        "Need to simplify\n1000 → 20 → 5"
    ]
    
    # Right column: Formal concepts
    concepts = [
        "Clustering:\nUnsupervised Learning",
        "Distance Metrics:\nEuclidean, Manhattan",
        "Centroids:\nCluster Representatives",
        "Outliers:\nAnomaly Detection",
        "Dimensionality\nReduction"
    ]
    
    # Middle column: Algorithms
    algorithms = [
        "K-means",
        "DBSCAN",
        "Hierarchical",
        "GMM",
        "PCA"
    ]
    
    # Draw boxes and connections
    y_positions = np.linspace(0.8, 0.15, 5)
    
    for i, (disc, algo, conc) in enumerate(zip(discoveries, algorithms, concepts)):
        y = y_positions[i]
        
        # Discovery box (left)
        disc_box = FancyBboxPatch((0.05, y-0.06), 0.22, 0.1,
                                   boxstyle="round,pad=0.01",
                                   facecolor=colors['mlblue'], alpha=0.3,
                                   edgecolor=colors['mlblue'], linewidth=2,
                                   transform=ax.transAxes)
        ax.add_patch(disc_box)
        ax.text(0.16, y, disc, ha='center', va='center', 
                fontsize=10, transform=ax.transAxes)
        
        # Algorithm circle (middle)
        algo_circle = Circle((0.5, y), 0.05, 
                            facecolor=colors['mlorange'], alpha=0.3,
                            edgecolor=colors['mlorange'], linewidth=2,
                            transform=ax.transAxes)
        ax.add_patch(algo_circle)
        ax.text(0.5, y, algo, ha='center', va='center',
                fontsize=11, fontweight='bold', transform=ax.transAxes)
        
        # Concept box (right)
        conc_box = FancyBboxPatch((0.73, y-0.06), 0.22, 0.1,
                                   boxstyle="round,pad=0.01",
                                   facecolor=colors['mlgreen'], alpha=0.3,
                                   edgecolor=colors['mlgreen'], linewidth=2,
                                   transform=ax.transAxes)
        ax.add_patch(conc_box)
        ax.text(0.84, y, conc, ha='center', va='center',
                fontsize=10, transform=ax.transAxes)
        
        # Arrows
        arrow1 = FancyArrowPatch((0.28, y), (0.44, y),
                                connectionstyle="arc3,rad=0", 
                                arrowstyle='->', mutation_scale=20,
                                color=colors['gray'], linewidth=1.5,
                                transform=ax.transAxes)
        ax.add_patch(arrow1)
        
        arrow2 = FancyArrowPatch((0.56, y), (0.72, y),
                                connectionstyle="arc3,rad=0",
                                arrowstyle='->', mutation_scale=20,
                                color=colors['gray'], linewidth=1.5,
                                transform=ax.transAxes)
        ax.add_patch(arrow2)
    
    # Headers
    ax.text(0.16, 0.88, 'YOUR DISCOVERIES', fontsize=12, fontweight='bold',
            ha='center', color=colors['mlblue'], transform=ax.transAxes)
    ax.text(0.5, 0.88, 'ALGORITHMS', fontsize=12, fontweight='bold',
            ha='center', color=colors['mlorange'], transform=ax.transAxes)
    ax.text(0.84, 0.88, 'FORMAL THEORY', fontsize=12, fontweight='bold',
            ha='center', color=colors['mlgreen'], transform=ax.transAxes)
    
    # Innovation pipeline at bottom
    ax.text(0.5, 0.05, 'Innovation Pipeline: Data → Patterns → Insights → Action',
            fontsize=12, fontstyle='italic', ha='center',
            color=colors['mlpurple'], transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../handouts/visual5_theory_map.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../handouts/visual5_theory_map.png', dpi=150, bbox_inches='tight')
    print("Created: visual5_theory_map.pdf")

def create_math_icons():
    """Visual 8: Mathematical equations using visual icons"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('The Mathematics Hidden in Your Process', fontsize=18, fontweight='bold')
    
    # 1. K-means objective (top left)
    ax = axes[0, 0]
    ax.set_title('What You Minimize (K-means)', fontsize=14, fontweight='bold')
    
    # Visual equation
    ax.text(0.1, 0.5, 'Goal =', fontsize=20, transform=ax.transAxes)
    ax.text(0.3, 0.5, 'Σ', fontsize=30, transform=ax.transAxes)
    
    # Draw dots and center
    for i, x in enumerate([0.45, 0.5, 0.55]):
        for j, y in enumerate([0.3, 0.5, 0.7]):
            if not (x == 0.5 and y == 0.5):
                circle = Circle((x, y), 0.02, color=colors['mlblue'], 
                              transform=ax.transAxes)
                ax.add_patch(circle)
    
    # Center point
    center = Circle((0.5, 0.5), 0.03, color=colors['mlred'], 
                   transform=ax.transAxes)
    ax.add_patch(center)
    
    # Distance lines
    for x, y in [(0.45, 0.3), (0.55, 0.7)]:
        line = mlines.Line2D([x, 0.5], [y, 0.5], color=colors['gray'],
                           linestyle='--', transform=ax.transAxes)
        ax.add_line(line)
    
    ax.text(0.65, 0.5, '(distance)²', fontsize=16, transform=ax.transAxes)
    ax.text(0.5, 0.1, 'Minimize total squared distance\nfrom points to centers',
            fontsize=12, ha='center', fontstyle='italic', transform=ax.transAxes)
    
    # 2. Silhouette score (top right)
    ax = axes[0, 1]
    ax.set_title('What You Maximize (Quality)', fontsize=14, fontweight='bold')
    
    ax.text(0.1, 0.5, 'Quality =', fontsize=20, transform=ax.transAxes)
    ax.text(0.38, 0.65, 'b - a', fontsize=18, transform=ax.transAxes)
    ax.text(0.38, 0.5, '————', fontsize=16, transform=ax.transAxes)
    ax.text(0.36, 0.35, 'max(a,b)', fontsize=16, transform=ax.transAxes)
    
    # Visual representation
    # Cluster A
    for x, y in [(0.65, 0.5), (0.67, 0.48), (0.66, 0.52)]:
        circle = Circle((x, y), 0.015, color=colors['mlblue'], 
                       transform=ax.transAxes)
        ax.add_patch(circle)
    
    # Cluster B
    for x, y in [(0.8, 0.5), (0.82, 0.48), (0.81, 0.52)]:
        circle = Circle((x, y), 0.015, color=colors['mlorange'], 
                       transform=ax.transAxes)
        ax.add_patch(circle)
    
    # Distance arrows
    ax.annotate('', xy=(0.68, 0.5), xytext=(0.64, 0.5),
                arrowprops=dict(arrowstyle='<->', color=colors['mlgreen']),
                transform=ax.transAxes)
    ax.text(0.66, 0.42, 'a', fontsize=12, ha='center', color=colors['mlgreen'],
            transform=ax.transAxes)
    
    ax.annotate('', xy=(0.79, 0.5), xytext=(0.68, 0.5),
                arrowprops=dict(arrowstyle='<->', color=colors['mlred']),
                transform=ax.transAxes)
    ax.text(0.735, 0.42, 'b', fontsize=12, ha='center', color=colors['mlred'],
            transform=ax.transAxes)
    
    ax.text(0.5, 0.1, 'Maximize separation relative to cohesion',
            fontsize=12, ha='center', fontstyle='italic', transform=ax.transAxes)
    
    # 3. Elbow method (bottom left)
    ax = axes[1, 0]
    ax.set_title('Finding Optimal K (Elbow)', fontsize=14, fontweight='bold')
    
    # Draw elbow curve
    k_values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    inertia = np.array([100, 50, 30, 25, 24, 23.5, 23, 22.8])
    
    ax.plot(k_values, inertia, 'o-', color=colors['mlpurple'], linewidth=2, markersize=8)
    ax.plot(3, 30, 'o', color=colors['mlred'], markersize=12)
    ax.annotate('Elbow Point\n(Optimal K)', xy=(3, 30), xytext=(4.5, 50),
                arrowprops=dict(arrowstyle='->', color=colors['mlred']),
                fontsize=12, color=colors['mlred'])
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Within-Cluster Sum of Squares', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 4. Innovation formula (bottom right)
    ax = axes[1, 1]
    ax.set_title('Your Innovation Equation', fontsize=14, fontweight='bold')
    
    # Visual formula
    elements = ['Many\nIdeas', '+', 'Smart\nGrouping', '+', 'Human\nInsight', '=', 'Strategic\nInnovation']
    x_positions = [0.1, 0.22, 0.34, 0.46, 0.58, 0.7, 0.85]
    colors_seq = [colors['mlblue'], 'black', colors['mlorange'], 'black', 
                  colors['mlgreen'], 'black', colors['mlpurple']]
    
    for x, elem, col in zip(x_positions, elements, colors_seq):
        if elem in ['+', '=']:
            ax.text(x, 0.5, elem, fontsize=24, ha='center', va='center',
                   transform=ax.transAxes)
        else:
            box = FancyBboxPatch((x-0.06, 0.35), 0.12, 0.3,
                                 boxstyle="round,pad=0.01",
                                 facecolor=col, alpha=0.2,
                                 edgecolor=col, linewidth=2,
                                 transform=ax.transAxes)
            ax.add_patch(box)
            ax.text(x, 0.5, elem, fontsize=11, ha='center', va='center',
                   fontweight='bold', transform=ax.transAxes)
    
    ax.text(0.5, 0.15, '1000 → Cluster → 20 themes → Prioritize → 5 actions',
            fontsize=12, ha='center', fontstyle='italic', 
            color=colors['gray'], transform=ax.transAxes)
    
    # Turn off axes for icon plots
    for ax in [axes[0, 0], axes[0, 1], axes[1, 1]]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../handouts/visual8_math_icons.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../handouts/visual8_math_icons.png', dpi=150, bbox_inches='tight')
    print("Created: visual8_math_icons.pdf")

def create_personal_framework():
    """Visual 9: Personal Framework - Blank template for students"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Title
    ax.text(0.5, 0.95, 'Build Your Personal Clustering & Innovation Framework',
            fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Three main sections
    sections = [
        {'title': 'Theory Foundation', 'x': 0.2, 'color': colors['mlblue']},
        {'title': 'Your Rules', 'x': 0.5, 'color': colors['mlorange']},
        {'title': 'Applications', 'x': 0.8, 'color': colors['mlgreen']}
    ]
    
    for section in sections:
        # Section header
        header_box = FancyBboxPatch((section['x']-0.12, 0.78), 0.24, 0.08,
                                    boxstyle="round,pad=0.01",
                                    facecolor=section['color'], alpha=0.3,
                                    edgecolor=section['color'], linewidth=2,
                                    transform=ax.transAxes)
        ax.add_patch(header_box)
        ax.text(section['x'], 0.82, section['title'], 
                fontsize=12, fontweight='bold', ha='center',
                transform=ax.transAxes)
        
        # Content boxes (empty for students to fill)
        for i, y in enumerate([0.65, 0.50, 0.35]):
            content_box = Rectangle((section['x']-0.12, y-0.05), 0.24, 0.1,
                                   facecolor='white', 
                                   edgecolor=section['color'], linewidth=1,
                                   linestyle='--', transform=ax.transAxes)
            ax.add_patch(content_box)
            
            # Placeholder text
            placeholder = ['Concept 1:', 'Concept 2:', 'Concept 3:'][i]
            ax.text(section['x']-0.10, y+0.03, placeholder,
                   fontsize=9, color='gray', fontstyle='italic',
                   transform=ax.transAxes)
    
    # Connection arrows
    for x in [0.33, 0.63]:
        arrow = FancyArrowPatch((x, 0.5), (x+0.04, 0.5),
                               connectionstyle="arc3,rad=0",
                               arrowstyle='->', mutation_scale=25,
                               color=colors['gray'], linewidth=2,
                               transform=ax.transAxes)
        ax.add_patch(arrow)
    
    # Bottom section: Your definitions
    definition_box = FancyBboxPatch((0.1, 0.08), 0.8, 0.15,
                                   boxstyle="round,pad=0.01",
                                   facecolor=colors['lightgray'], alpha=0.2,
                                   edgecolor=colors['mlpurple'], linewidth=2,
                                   transform=ax.transAxes)
    ax.add_patch(definition_box)
    
    ax.text(0.5, 0.19, 'Your Definitions', fontsize=12, fontweight='bold',
            ha='center', color=colors['mlpurple'], transform=ax.transAxes)
    
    ax.text(0.15, 0.14, 'Clustering is: _' + '_'*60,
            fontsize=10, transform=ax.transAxes)
    ax.text(0.15, 0.10, 'Innovation happens when: _' + '_'*48,
            fontsize=10, transform=ax.transAxes)
    
    # Questions for reflection
    questions = [
        "What patterns appeared across all exercises?",
        "Which of your initial theories held true?",
        "What surprised you most?",
        "How will you apply this in your field?"
    ]
    
    ax.text(0.05, 0.28, 'Reflection Questions:', fontsize=11, fontweight='bold',
            transform=ax.transAxes)
    
    for i, q in enumerate(questions):
        ax.text(0.05, 0.24-i*0.03, f"{i+1}. {q}", fontsize=9,
                transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../handouts/visual9_personal_framework.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../handouts/visual9_personal_framework.png', dpi=150, bbox_inches='tight')
    print("Created: visual9_personal_framework.pdf")

def main():
    """Create all missing visual elements"""
    print("Creating missing visual elements for discovery-based learning...")
    
    # Create Visual 5: Theory Map
    create_theory_map()
    
    # Create Visual 8: Math Icons
    create_math_icons()
    
    # Create Visual 9: Personal Framework
    create_personal_framework()
    
    print("\nAll missing visuals created successfully!")
    print("Files saved in ../handouts/ directory")

if __name__ == "__main__":
    main()