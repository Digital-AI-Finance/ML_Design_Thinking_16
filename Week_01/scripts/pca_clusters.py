import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Generate high-dimensional data
n_samples = 1000
n_features = 10
n_clusters = 4

# Create data with more features
X, y_true = make_blobs(n_samples=n_samples, 
                       n_features=n_features,
                       centers=n_clusters,
                       cluster_std=1.5,
                       random_state=42)

# Perform K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# Reduce to 2D using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centers_pca = pca.transform(kmeans.cluster_centers_)

# Create the visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Define colors and labels
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
labels = ['Power Users', 'Regular Users', 'New Users', 'Casual Users']

# Plot each cluster
for i in range(n_clusters):
    mask = clusters == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
              c=colors[i], label=labels[i],
              alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

# Plot cluster centers
ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
          c='black', marker='X', s=300, 
          edgecolors='white', linewidth=2,
          label='Cluster Centers', zorder=5)

# Add confidence ellipses
from matplotlib.patches import Ellipse

for i in range(n_clusters):
    mask = clusters == i
    cluster_data = X_pca[mask]
    
    # Calculate covariance and ellipse parameters
    cov = np.cov(cluster_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # Draw ellipse
    ellipse = Ellipse(centers_pca[i], 
                      2 * np.sqrt(eigenvalues[0]),
                      2 * np.sqrt(eigenvalues[1]),
                      angle=angle,
                      facecolor=colors[i], alpha=0.1,
                      edgecolor=colors[i], linewidth=2, linestyle='--')
    ax.add_patch(ellipse)

# Labels and styling
ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
             fontsize=12, fontweight='bold')
ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
             fontsize=12, fontweight='bold')
ax.set_title('User Clusters Visualization (PCA Reduced from 10D to 2D)', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)

# Add text box with explanation
textstr = f'PCA reduces {n_features} dimensions to 2\nwhile preserving {pca.explained_variance_ratio_.sum():.1%} of variance'
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('pca_clusters.pdf', dpi=300, bbox_inches='tight')
plt.savefig('pca_clusters.png', dpi=150, bbox_inches='tight')
plt.close()

print("PCA clusters visualization created!")