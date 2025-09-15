#!/usr/bin/env python3
"""
Create Visual Elements for Discovery-Based Learning Handouts
Focus on innovation and pattern discovery
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib.patches as patches
from scipy.cluster.hierarchy import dendrogram, linkage

# Set style and random seed
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Color palette
colors = {
    'mlblue': '#1f77b4',
    'mlorange': '#ff7f0e', 
    'mlgreen': '#2ca02c',
    'mlred': '#d62728',
    'mlpurple': '#9467bd',
    'gray': '#7f7f7f',
    'light': '#f0f0f0',
    'dark': '#2c2c2c'
}

# ============ VISUAL 1: The Dot Cloud Mystery ============
def create_dot_cloud_mystery():
    """Create unlabeled scatter plot with hidden structure"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate data with hidden clusters
    centers = [[2, 2], [8, 2], [5, 8], [8, 8], [2, 8]]
    X, _ = make_blobs(n_samples=500, centers=centers, n_features=2, 
                      cluster_std=0.8, random_state=42)
    
    # Plot without any labels or colors
    ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, s=20)
    
    ax.set_title('What Patterns Do You See?', fontsize=16, fontweight='bold')
    ax.set_xlabel('Innovation Dimension 1', fontsize=12)
    ax.set_ylabel('Innovation Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Remove tick labels to make it more abstract
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.tight_layout()
    plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/handouts/visual1_dot_cloud.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/handouts/visual1_dot_cloud.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

# ============ VISUAL 2: Three Distance Types ============
def create_distance_visualizations():
    """Show three different ways to measure distance"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Points for demonstration
    point1 = np.array([2, 2])
    point2 = np.array([6, 5])
    
    # Panel A: Euclidean Distance
    ax = axes[0]
    ax.scatter(*point1, s=100, c=colors['mlblue'], zorder=5)
    ax.scatter(*point2, s=100, c=colors['mlred'], zorder=5)
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 
            'k--', linewidth=2, label='Euclidean')
    ax.annotate('', xy=point2, xytext=point1,
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.set_title('A: Straight Line Distance\n"As the crow flies"', fontsize=12)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 7)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Panel B: Manhattan Distance
    ax = axes[1]
    ax.scatter(*point1, s=100, c=colors['mlblue'], zorder=5)
    ax.scatter(*point2, s=100, c=colors['mlred'], zorder=5)
    ax.plot([point1[0], point2[0], point2[0]], 
            [point1[1], point1[1], point2[1]], 
            'k--', linewidth=2, label='Manhattan')
    ax.set_title('B: City Block Distance\n"Walking on a grid"', fontsize=12)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 7)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Panel C: Cosine Similarity (as angle)
    ax = axes[2]
    origin = [0, 0]
    ax.scatter(*origin, s=100, c='black', zorder=5)
    ax.arrow(0, 0, 4, 2, head_width=0.2, head_length=0.2, 
             fc=colors['mlblue'], ec=colors['mlblue'])
    ax.arrow(0, 0, 3, 4, head_width=0.2, head_length=0.2, 
             fc=colors['mlred'], ec=colors['mlred'])
    # Draw angle arc
    angle = patches.Arc((0, 0), 2, 2, angle=0, 
                        theta1=np.degrees(np.arctan2(2, 4)), 
                        theta2=np.degrees(np.arctan2(4, 3)), 
                        color='black', linewidth=2)
    ax.add_patch(angle)
    ax.set_title('C: Angular Distance\n"Direction similarity"', fontsize=12)
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.suptitle('Three Ways to Measure "How Close?"', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/handouts/visual2_distances.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/handouts/visual2_distances.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

# ============ VISUAL 3: Evolution of Understanding ============
def create_evolution_panels():
    """Show progression from chaos to order with blank panels"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    np.random.seed(42)
    
    # Panel 1: Chaos - 100 innovation ideas
    ax = axes[0]
    ideas = np.random.randn(100, 2) * 2 + 5
    ax.scatter(ideas[:, 0], ideas[:, 1], c='gray', alpha=0.4, s=30)
    ax.set_title('1. Innovation Ideas\n(100 items)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: ??? - Blank for students
    ax = axes[1]
    ax.text(0.5, 0.5, '?', fontsize=72, ha='center', va='center',
            transform=ax.transAxes, color='gray', alpha=0.3)
    ax.set_title('2. Your Process Here\n(? items)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: ??? - Another blank
    ax = axes[2]
    ax.text(0.5, 0.5, '?', fontsize=72, ha='center', va='center',
            transform=ax.transAxes, color='gray', alpha=0.3)
    ax.set_title('3. Your Process Here\n(? items)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Order - 5 strategic initiatives
    ax = axes[3]
    centers = np.array([[2, 5], [5, 8], [8, 7], [5, 2], [8, 3]])
    for i, center in enumerate(centers):
        circle = plt.Circle(center, 1.2, color=list(colors.values())[i], 
                           alpha=0.3)
        ax.add_patch(circle)
        ax.scatter(*center, s=200, c=list(colors.values())[i], 
                  edgecolors='black', linewidth=2, zorder=5)
        ax.text(center[0], center[1]-2, f'Strategy {i+1}', 
               ha='center', fontsize=10)
    
    ax.set_title('4. Strategic Initiatives\n(5 items)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('The Innovation Journey: From Ideas to Action', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/handouts/visual3_evolution.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/handouts/visual3_evolution.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

# ============ VISUAL 4: The Shape Challenge ============
def create_shape_challenge():
    """Different clustering scenarios for discovery"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Scenario 1: Circular clusters
    X1, _ = make_blobs(n_samples=150, centers=3, n_features=2, 
                       cluster_std=0.5, random_state=42)
    axes[0].scatter(X1[:, 0], X1[:, 1], c='gray', alpha=0.6, s=20)
    axes[0].set_title('Pattern A: Circular Groups', fontsize=11, fontweight='bold')
    
    # Scenario 2: Elongated clusters
    X2, _ = make_blobs(n_samples=150, centers=3, n_features=2, 
                       cluster_std=[1.0, 0.3, 0.5], random_state=42)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X2 = np.dot(X2, transformation)
    axes[1].scatter(X2[:, 0], X2[:, 1], c='gray', alpha=0.6, s=20)
    axes[1].set_title('Pattern B: Elongated Groups', fontsize=11, fontweight='bold')
    
    # Scenario 3: Nested circles
    X3, _ = make_circles(n_samples=150, factor=0.5, noise=0.05, random_state=42)
    axes[2].scatter(X3[:, 0], X3[:, 1], c='gray', alpha=0.6, s=20)
    axes[2].set_title('Pattern C: Nested Groups', fontsize=11, fontweight='bold')
    
    # Scenario 4: Moons
    X4, _ = make_moons(n_samples=150, noise=0.1, random_state=42)
    axes[3].scatter(X4[:, 0], X4[:, 1], c='gray', alpha=0.6, s=20)
    axes[3].set_title('Pattern D: Curved Groups', fontsize=11, fontweight='bold')
    
    # Scenario 5: Different densities
    X5_1, _ = make_blobs(n_samples=50, centers=[[2, 2]], n_features=2, 
                         cluster_std=0.2, random_state=42)
    X5_2, _ = make_blobs(n_samples=100, centers=[[6, 2]], n_features=2, 
                         cluster_std=0.8, random_state=42)
    X5 = np.vstack([X5_1, X5_2])
    axes[4].scatter(X5[:, 0], X5[:, 1], c='gray', alpha=0.6, s=20)
    axes[4].set_title('Pattern E: Different Densities', fontsize=11, fontweight='bold')
    
    # Scenario 6: With outliers
    X6, _ = make_blobs(n_samples=130, centers=2, n_features=2, 
                       cluster_std=0.5, random_state=42)
    # Add outliers
    outliers = np.random.uniform(-3, 8, (20, 2))
    X6 = np.vstack([X6, outliers])
    axes[5].scatter(X6[:, 0], X6[:, 1], c='gray', alpha=0.6, s=20)
    axes[5].set_title('Pattern F: Groups with Outliers', fontsize=11, fontweight='bold')
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    plt.suptitle('The Shape Challenge: What Rules Would Find These Groups?', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/handouts/visual4_shapes.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/handouts/visual4_shapes.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

# ============ VISUAL 6: Algorithm Behavior Grid ============
def create_algorithm_grid():
    """Show how different algorithms handle different data"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Create three different datasets
    datasets = []
    
    # Dataset 1: Well-separated blobs
    X1, y1 = make_blobs(n_samples=150, centers=3, n_features=2, 
                        cluster_std=0.5, random_state=42)
    datasets.append((X1, "Well-Separated"))
    
    # Dataset 2: Moons
    X2, y2 = make_moons(n_samples=150, noise=0.1, random_state=42)
    datasets.append((X2, "Non-Convex"))
    
    # Dataset 3: Different densities
    X3_1, _ = make_blobs(n_samples=50, centers=[[2, 2]], n_features=2, 
                         cluster_std=0.2, random_state=42)
    X3_2, _ = make_blobs(n_samples=100, centers=[[5, 2]], n_features=2, 
                         cluster_std=0.8, random_state=42)
    X3 = np.vstack([X3_1, X3_2])
    datasets.append((X3, "Varying Density"))
    
    algorithms = [
        ("K-means", KMeans(n_clusters=2, random_state=42)),
        ("DBSCAN", DBSCAN(eps=0.3, min_samples=5)),
        ("Hierarchical", AgglomerativeClustering(n_clusters=2))
    ]
    
    for i, (algo_name, algo) in enumerate(algorithms):
        for j, (data, data_name) in enumerate(datasets):
            ax = axes[i, j]
            
            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data)
            
            # Apply algorithm
            if algo_name == "DBSCAN":
                labels = algo.fit_predict(X_scaled)
            else:
                labels = algo.fit_predict(X_scaled)
            
            # Plot results
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:  # Noise points in DBSCAN
                    ax.scatter(data[labels == label, 0], 
                             data[labels == label, 1],
                             c='black', alpha=0.3, s=10, label='Noise')
                else:
                    ax.scatter(data[labels == label, 0], 
                             data[labels == label, 1],
                             alpha=0.6, s=30)
            
            if i == 0:
                ax.set_title(f'{data_name}', fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{algo_name}', fontsize=12, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    
    plt.suptitle('Algorithm Behavior Patterns: How Different Methods Handle Different Data', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/handouts/visual6_algorithm_grid.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/handouts/visual6_algorithm_grid.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

# ============ VISUAL 7: Innovation Landscape ============
def create_innovation_landscape():
    """Topographical map metaphor for innovation space"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a landscape
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create multiple peaks (solution clusters) and valleys (problem spaces)
    Z = (2 * np.exp(-((X-2)**2 + (Y-2)**2)/2) +     # Solution cluster 1
         3 * np.exp(-((X+2)**2 + (Y-1)**2)/3) +      # Solution cluster 2
         1.5 * np.exp(-((X)**2 + (Y+3)**2)/2) +      # Solution cluster 3
         -1 * np.exp(-((X-1)**2 + (Y+1)**2)/4) +     # Problem valley 1
         -0.8 * np.exp(-((X+3)**2 + (Y+2)**2)/3))    # Problem valley 2
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.8, 
                           linewidth=0, antialiased=True)
    
    # Add contour lines at the bottom
    ax.contour(X, Y, Z, zdir='z', offset=-3, cmap='viridis', alpha=0.5)
    
    # Mark peaks and valleys
    ax.scatter([2, -2, 0], [2, -1, -3], [2.0, 2.8, 1.4], 
              c='red', s=100, marker='^', label='Solution Clusters')
    ax.scatter([1, -3], [-1, -2], [-0.8, -0.6], 
              c='blue', s=100, marker='v', label='Problem Spaces')
    
    ax.set_xlabel('Innovation Dimension 1', fontsize=10)
    ax.set_ylabel('Innovation Dimension 2', fontsize=10)
    ax.set_zlabel('Value/Impact', fontsize=10)
    ax.set_title('The Innovation Landscape:\nValleys (Problems) → Peaks (Solutions)', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.legend()
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/handouts/visual7_landscape.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/handouts/visual7_landscape.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

# ============ Main Execution ============
def main():
    """Generate all handout visuals"""
    print("Creating handout visuals...")
    
    # Create output directory
    import os
    output_dir = 'D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/handouts'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all visuals
    print("1. Creating Dot Cloud Mystery...")
    create_dot_cloud_mystery()
    
    print("2. Creating Distance Visualizations...")
    create_distance_visualizations()
    
    print("3. Creating Evolution Panels...")
    create_evolution_panels()
    
    print("4. Creating Shape Challenge...")
    create_shape_challenge()
    
    print("5. Creating Algorithm Grid...")
    create_algorithm_grid()
    
    print("6. Creating Innovation Landscape...")
    create_innovation_landscape()
    
    print("\nAll visuals created successfully!")
    print(f"Files saved in: {output_dir}")

if __name__ == "__main__":
    main()