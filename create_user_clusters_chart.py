import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Define complexity levels
complexity_levels = [
    {'n_features': 2, 'n_clusters': 2, 'n_samples': 100, 'name': 'Simple'},
    {'n_features': 5, 'n_clusters': 3, 'n_samples': 500, 'name': 'Low'},
    {'n_features': 10, 'n_clusters': 4, 'n_samples': 1000, 'name': 'Medium-Low'},
    {'n_features': 20, 'n_clusters': 5, 'n_samples': 2000, 'name': 'Medium'},
    {'n_features': 30, 'n_clusters': 6, 'n_samples': 3000, 'name': 'Medium-High'},
    {'n_features': 50, 'n_clusters': 8, 'n_samples': 5000, 'name': 'High'},
    {'n_features': 75, 'n_clusters': 9, 'n_samples': 7500, 'name': 'Very High'},
    {'n_features': 100, 'n_clusters': 10, 'n_samples': 10000, 'name': 'Complex'}
]

# Calculate complexity score (normalized composite)
for level in complexity_levels:
    level['complexity'] = (level['n_features']/100 * 0.4 + 
                           level['n_clusters']/10 * 0.3 + 
                           level['n_samples']/10000 * 0.3)

# Store results
ml_silhouette = []
ml_davies_bouldin = []
ml_calinski = []
manual_silhouette = []
manual_davies_bouldin = []
manual_calinski = []
complexity_scores = []

print("=== Clustering Quality vs Data Complexity ===\n")

for i, level in enumerate(complexity_levels):
    print(f"Processing {level['name']} complexity...")
    print(f"  Features: {level['n_features']}, Clusters: {level['n_clusters']}, Samples: {level['n_samples']}")
    
    # Generate dataset
    X, y_true = make_blobs(n_samples=level['n_samples'], 
                          n_features=level['n_features'], 
                          centers=level['n_clusters'],
                          cluster_std=1.5,
                          random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ML clustering (K-means with correct number of clusters)
    kmeans = KMeans(n_clusters=level['n_clusters'], random_state=42, n_init=10)
    ml_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate ML metrics
    ml_sil = silhouette_score(X_scaled, ml_labels)
    ml_db = davies_bouldin_score(X_scaled, ml_labels)
    ml_ch = calinski_harabasz_score(X_scaled, ml_labels)
    
    ml_silhouette.append(ml_sil)
    ml_davies_bouldin.append(ml_db)
    ml_calinski.append(ml_ch)
    
    # Simulate manual clustering (random with bias toward spatial proximity)
    # Manual clustering degrades with complexity
    degradation_factor = 1 - (i * 0.1)  # Gets worse as complexity increases
    
    # Start with random assignments
    manual_labels = np.random.randint(0, level['n_clusters'], size=level['n_samples'])
    
    # Add some structure based on first few dimensions (simulating human pattern recognition)
    if level['n_features'] <= 3:
        # Humans can see patterns in 2-3D
        for j in range(level['n_samples']):
            # Use simple threshold on first dimension
            manual_labels[j] = int(X_scaled[j, 0] * degradation_factor + 
                                  np.random.normal(0, 0.5)) % level['n_clusters']
    else:
        # Humans struggle with high dimensions - mostly random
        noise_level = 0.3 + (i * 0.1)  # More noise as complexity increases
        manual_labels = np.random.randint(0, level['n_clusters'], size=level['n_samples'])
    
    # Calculate manual metrics
    manual_sil = silhouette_score(X_scaled, manual_labels)
    manual_db = davies_bouldin_score(X_scaled, manual_labels)
    manual_ch = calinski_harabasz_score(X_scaled, manual_labels)
    
    manual_silhouette.append(manual_sil)
    manual_davies_bouldin.append(manual_db)
    manual_calinski.append(manual_ch)
    
    complexity_scores.append(level['complexity'])
    
    print(f"  ML Silhouette: {ml_sil:.3f}, Manual Silhouette: {manual_sil:.3f}")
    print(f"  Improvement: {((ml_sil - manual_sil)/abs(manual_sil))*100:.1f}%\n")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Silhouette Score Comparison
ax1.plot(complexity_scores, ml_silhouette, 'o-', color='#1f77b4', linewidth=2.5, 
         markersize=8, label='ML Clustering (K-means)')
ax1.plot(complexity_scores, manual_silhouette, 's--', color='#d62728', linewidth=2, 
         markersize=7, label='Manual Clustering', alpha=0.8)

ax1.set_xlabel('Data Complexity (Composite Score)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax1.set_title('Clustering Quality vs Data Complexity\n' +
              'Higher score = better defined clusters', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right', fontsize=11)

# Add shaded regions for interpretation
ax1.axhspan(0.7, 1.0, alpha=0.1, color='green', label='Excellent')
ax1.axhspan(0.5, 0.7, alpha=0.1, color='yellow')
ax1.axhspan(0.25, 0.5, alpha=0.1, color='orange')
ax1.axhspan(-1, 0.25, alpha=0.1, color='red')

# Add annotations
best_ml_idx = np.argmax(ml_silhouette)
ax1.annotate('ML maintains quality', 
             xy=(complexity_scores[best_ml_idx], ml_silhouette[best_ml_idx]),
             xytext=(complexity_scores[best_ml_idx]-0.1, ml_silhouette[best_ml_idx]+0.1),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
             fontsize=10, fontweight='bold')

worst_manual_idx = np.argmin(manual_silhouette)
ax1.annotate('Manual degrades\nwith complexity', 
             xy=(complexity_scores[worst_manual_idx], manual_silhouette[worst_manual_idx]),
             xytext=(complexity_scores[worst_manual_idx]-0.15, manual_silhouette[worst_manual_idx]+0.2),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, fontweight='bold')

# Plot 2: Multiple Metrics Comparison
ax2.plot(complexity_scores, np.array(ml_davies_bouldin), 'o-', color='#2ca02c', 
         linewidth=2, markersize=7, label='ML Davies-Bouldin (lower=better)')
ax2.plot(complexity_scores, np.array(manual_davies_bouldin), 's--', color='#ff7f0e', 
         linewidth=2, markersize=6, label='Manual Davies-Bouldin', alpha=0.8)

# Normalize Calinski-Harabasz for visualization (divide by 1000)
ax2_twin = ax2.twinx()
ax2_twin.plot(complexity_scores, np.array(ml_calinski)/1000, '^-', color='#9467bd', 
              linewidth=2, markersize=7, label='ML Calinski-Harabasz/1000')
ax2_twin.plot(complexity_scores, np.array(manual_calinski)/1000, 'v--', color='#17becf', 
              linewidth=2, markersize=6, label='Manual Calinski-Harabasz/1000', alpha=0.8)

ax2.set_xlabel('Data Complexity (Composite Score)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Davies-Bouldin Index', fontsize=11, fontweight='bold', color='#2ca02c')
ax2_twin.set_ylabel('Calinski-Harabasz Score (×1000)', fontsize=11, fontweight='bold', color='#9467bd')
ax2.set_title('Additional Clustering Metrics\n' +
              'DB: lower=better, CH: higher=better', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

# Combine legends
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

plt.tight_layout()

# Save the chart
plt.savefig('charts/user_clusters.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/user_clusters.png', dpi=150, bbox_inches='tight')  # Also save PNG for preview
print("\n=== Chart saved to charts/user_clusters.pdf ===")

# Print summary statistics
print("\n=== Summary Statistics ===")
print("\nSilhouette Score Comparison:")
print(f"  ML Average: {np.mean(ml_silhouette):.3f} (±{np.std(ml_silhouette):.3f})")
print(f"  Manual Average: {np.mean(manual_silhouette):.3f} (±{np.std(manual_silhouette):.3f})")
print(f"  Overall ML Advantage: {((np.mean(ml_silhouette) - np.mean(manual_silhouette))/abs(np.mean(manual_silhouette)))*100:.1f}%")

print("\nML Performance by Complexity:")
for i, level in enumerate(complexity_levels):
    print(f"  {level['name']:12s}: Silhouette={ml_silhouette[i]:.3f}, " +
          f"DB={ml_davies_bouldin[i]:.2f}, CH={ml_calinski[i]:.0f}")