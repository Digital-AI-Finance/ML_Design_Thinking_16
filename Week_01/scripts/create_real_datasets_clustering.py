#!/usr/bin/env python3
"""
Create Real Dataset Clustering Examples for Week 1
Shows clustering on actual datasets: Iris, Wine, Customer Segmentation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.datasets import load_iris, load_wine
import pandas as pd

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

# Create figure with subplots for different datasets
fig = plt.figure(figsize=(16, 12))

# Dataset 1: Iris Dataset
print("Processing Iris dataset...")
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
feature_names_iris = iris.feature_names
target_names_iris = iris.target_names

# Standardize the data
scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)

# Apply PCA for visualization
pca_iris = PCA(n_components=2)
X_iris_pca = pca_iris.fit_transform(X_iris_scaled)

# Apply different clustering algorithms
kmeans_iris = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans_iris = kmeans_iris.fit_predict(X_iris_scaled)

dbscan_iris = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan_iris = dbscan_iris.fit_predict(X_iris_scaled)

gmm_iris = GaussianMixture(n_components=3, random_state=42)
labels_gmm_iris = gmm_iris.fit_predict(X_iris_scaled)

# Calculate metrics
metrics_iris = {
    'K-Means': {
        'Silhouette': silhouette_score(X_iris_scaled, labels_kmeans_iris),
        'Davies-Bouldin': davies_bouldin_score(X_iris_scaled, labels_kmeans_iris),
        'Calinski-Harabasz': calinski_harabasz_score(X_iris_scaled, labels_kmeans_iris)
    },
    'GMM': {
        'Silhouette': silhouette_score(X_iris_scaled, labels_gmm_iris),
        'Davies-Bouldin': davies_bouldin_score(X_iris_scaled, labels_gmm_iris),
        'Calinski-Harabasz': calinski_harabasz_score(X_iris_scaled, labels_gmm_iris)
    }
}

# Plot Iris results
ax1 = plt.subplot(3, 4, 1)
ax1.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=y_iris, 
           cmap='viridis', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax1.set_title('Iris: True Labels', fontsize=10, fontweight='bold')
ax1.set_xlabel('PC1', fontsize=8)
ax1.set_ylabel('PC2', fontsize=8)

ax2 = plt.subplot(3, 4, 2)
ax2.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=labels_kmeans_iris,
           cmap='viridis', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax2.set_title(f'Iris: K-Means (Sil={metrics_iris["K-Means"]["Silhouette"]:.2f})', 
             fontsize=10, fontweight='bold')
ax2.set_xlabel('PC1', fontsize=8)
ax2.set_ylabel('PC2', fontsize=8)

ax3 = plt.subplot(3, 4, 3)
ax3.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=labels_dbscan_iris,
           cmap='viridis', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax3.set_title('Iris: DBSCAN', fontsize=10, fontweight='bold')
ax3.set_xlabel('PC1', fontsize=8)
ax3.set_ylabel('PC2', fontsize=8)

ax4 = plt.subplot(3, 4, 4)
ax4.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=labels_gmm_iris,
           cmap='viridis', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax4.set_title(f'Iris: GMM (Sil={metrics_iris["GMM"]["Silhouette"]:.2f})', 
             fontsize=10, fontweight='bold')
ax4.set_xlabel('PC1', fontsize=8)
ax4.set_ylabel('PC2', fontsize=8)

# Dataset 2: Wine Dataset
print("Processing Wine dataset...")
wine = load_wine()
X_wine = wine.data
y_wine = wine.target

# Standardize and reduce dimensions
scaler_wine = StandardScaler()
X_wine_scaled = scaler_wine.fit_transform(X_wine)
pca_wine = PCA(n_components=2)
X_wine_pca = pca_wine.fit_transform(X_wine_scaled)

# Apply clustering
kmeans_wine = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans_wine = kmeans_wine.fit_predict(X_wine_scaled)

dbscan_wine = DBSCAN(eps=1.5, min_samples=5)
labels_dbscan_wine = dbscan_wine.fit_predict(X_wine_scaled)

gmm_wine = GaussianMixture(n_components=3, random_state=42)
labels_gmm_wine = gmm_wine.fit_predict(X_wine_scaled)

# Calculate metrics
metrics_wine = {
    'K-Means': {
        'Silhouette': silhouette_score(X_wine_scaled, labels_kmeans_wine),
        'Davies-Bouldin': davies_bouldin_score(X_wine_scaled, labels_kmeans_wine),
    },
    'GMM': {
        'Silhouette': silhouette_score(X_wine_scaled, labels_gmm_wine),
        'Davies-Bouldin': davies_bouldin_score(X_wine_scaled, labels_gmm_wine),
    }
}

# Plot Wine results
ax5 = plt.subplot(3, 4, 5)
ax5.scatter(X_wine_pca[:, 0], X_wine_pca[:, 1], c=y_wine,
           cmap='plasma', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax5.set_title('Wine: True Labels', fontsize=10, fontweight='bold')
ax5.set_xlabel('PC1', fontsize=8)
ax5.set_ylabel('PC2', fontsize=8)

ax6 = plt.subplot(3, 4, 6)
ax6.scatter(X_wine_pca[:, 0], X_wine_pca[:, 1], c=labels_kmeans_wine,
           cmap='plasma', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax6.set_title(f'Wine: K-Means (Sil={metrics_wine["K-Means"]["Silhouette"]:.2f})', 
             fontsize=10, fontweight='bold')
ax6.set_xlabel('PC1', fontsize=8)
ax6.set_ylabel('PC2', fontsize=8)

ax7 = plt.subplot(3, 4, 7)
ax7.scatter(X_wine_pca[:, 0], X_wine_pca[:, 1], c=labels_dbscan_wine,
           cmap='plasma', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax7.set_title('Wine: DBSCAN', fontsize=10, fontweight='bold')
ax7.set_xlabel('PC1', fontsize=8)
ax7.set_ylabel('PC2', fontsize=8)

ax8 = plt.subplot(3, 4, 8)
ax8.scatter(X_wine_pca[:, 0], X_wine_pca[:, 1], c=labels_gmm_wine,
           cmap='plasma', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax8.set_title(f'Wine: GMM (Sil={metrics_wine["GMM"]["Silhouette"]:.2f})', 
             fontsize=10, fontweight='bold')
ax8.set_xlabel('PC1', fontsize=8)
ax8.set_ylabel('PC2', fontsize=8)

# Dataset 3: Customer Segmentation (Synthetic but realistic)
print("Creating Customer Segmentation dataset...")
np.random.seed(42)
n_customers = 200

# Create realistic customer features
# Feature 1: Annual Spending (in thousands)
# Feature 2: Frequency of Purchase
# Feature 3: Customer Lifetime Value Score

# Create 4 customer segments
segment1 = np.random.multivariate_normal([30, 12, 40], [[100, 20, 30], [20, 25, 15], [30, 15, 50]], 50)
segment2 = np.random.multivariate_normal([80, 6, 120], [[150, -30, 40], [-30, 20, -20], [40, -20, 100]], 50)
segment3 = np.random.multivariate_normal([50, 24, 60], [[80, 15, 25], [15, 30, 20], [25, 20, 60]], 50)
segment4 = np.random.multivariate_normal([120, 3, 200], [[200, -40, 60], [-40, 15, -30], [60, -30, 150]], 50)

X_customers = np.vstack([segment1, segment2, segment3, segment4])
y_customers = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)

# Standardize
scaler_cust = StandardScaler()
X_customers_scaled = scaler_cust.fit_transform(X_customers)
pca_cust = PCA(n_components=2)
X_customers_pca = pca_cust.fit_transform(X_customers_scaled)

# Apply clustering
kmeans_cust = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_kmeans_cust = kmeans_cust.fit_predict(X_customers_scaled)

dbscan_cust = DBSCAN(eps=0.8, min_samples=5)
labels_dbscan_cust = dbscan_cust.fit_predict(X_customers_scaled)

gmm_cust = GaussianMixture(n_components=4, random_state=42)
labels_gmm_cust = gmm_cust.fit_predict(X_customers_scaled)

# Calculate metrics
metrics_cust = {
    'K-Means': silhouette_score(X_customers_scaled, labels_kmeans_cust),
    'GMM': silhouette_score(X_customers_scaled, labels_gmm_cust)
}

# Plot Customer Segmentation results
ax9 = plt.subplot(3, 4, 9)
scatter9 = ax9.scatter(X_customers_pca[:, 0], X_customers_pca[:, 1], c=y_customers,
                      cmap='coolwarm', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax9.set_title('Customers: True Segments', fontsize=10, fontweight='bold')
ax9.set_xlabel('PC1', fontsize=8)
ax9.set_ylabel('PC2', fontsize=8)

ax10 = plt.subplot(3, 4, 10)
ax10.scatter(X_customers_pca[:, 0], X_customers_pca[:, 1], c=labels_kmeans_cust,
            cmap='coolwarm', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax10.set_title(f'Customers: K-Means (Sil={metrics_cust["K-Means"]:.2f})', 
              fontsize=10, fontweight='bold')
ax10.set_xlabel('PC1', fontsize=8)
ax10.set_ylabel('PC2', fontsize=8)

ax11 = plt.subplot(3, 4, 11)
ax11.scatter(X_customers_pca[:, 0], X_customers_pca[:, 1], c=labels_dbscan_cust,
            cmap='coolwarm', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax11.set_title('Customers: DBSCAN', fontsize=10, fontweight='bold')
ax11.set_xlabel('PC1', fontsize=8)
ax11.set_ylabel('PC2', fontsize=8)

ax12 = plt.subplot(3, 4, 12)
ax12.scatter(X_customers_pca[:, 0], X_customers_pca[:, 1], c=labels_gmm_cust,
            cmap='coolwarm', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax12.set_title(f'Customers: GMM (Sil={metrics_cust["GMM"]:.2f})', 
              fontsize=10, fontweight='bold')
ax12.set_xlabel('PC1', fontsize=8)
ax12.set_ylabel('PC2', fontsize=8)

# Add dataset information text
fig.text(0.5, 0.98, 'Real Dataset Clustering Comparison', 
         fontsize=16, fontweight='bold', ha='center')
fig.text(0.5, 0.96, 'Iris (150 samples, 4 features) | Wine (178 samples, 13 features) | Customers (200 samples, 3 features)',
         fontsize=10, ha='center', style='italic', color='gray')

# Add legend for customer segments
customer_labels = ['Budget Conscious', 'Premium', 'Frequent Buyers', 'VIP']
fig.text(0.5, 0.02, 'Customer Segments: ' + ' | '.join(customer_labels),
         fontsize=9, ha='center', bbox=dict(boxstyle='round,pad=0.3', 
                                           facecolor='lightblue', alpha=0.3))

# Add metrics summary box
metrics_text = (
    "Performance Summary:\n"
    f"Iris K-Means Silhouette: {metrics_iris['K-Means']['Silhouette']:.3f}\n"
    f"Wine K-Means Silhouette: {metrics_wine['K-Means']['Silhouette']:.3f}\n"
    f"Customer K-Means Silhouette: {metrics_cust['K-Means']:.3f}"
)
fig.text(0.02, 0.15, metrics_text, fontsize=8, 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

# Add data source citations
citations = (
    "Data Sources:\n"
    "• Iris: Fisher, R.A. (1936) UCI ML Repository\n"
    "• Wine: Forina, M. et al. (1991) UCI ML Repository\n"
    "• Customers: Simulated retail segmentation data"
)
fig.text(0.02, 0.02, citations, fontsize=7, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.04, 1, 0.95])

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/real_datasets_clustering.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/real_datasets_clustering.png', 
           dpi=150, bbox_inches='tight')

print("\nReal dataset clustering examples created successfully!")
print("Files saved:")
print("  - charts/real_datasets_clustering.pdf")
print("  - charts/real_datasets_clustering.png")
print("\nDataset Summary:")
print(f"  - Iris: {X_iris.shape[0]} samples, {X_iris.shape[1]} features")
print(f"  - Wine: {X_wine.shape[0]} samples, {X_wine.shape[1]} features")
print(f"  - Customers: {X_customers.shape[0]} samples, {X_customers.shape[1]} features")
print("\nKey Metrics:")
print(f"  - Best Iris clustering: K-Means (Silhouette={metrics_iris['K-Means']['Silhouette']:.3f})")
print(f"  - Best Wine clustering: GMM (Silhouette={metrics_wine['GMM']['Silhouette']:.3f})")
print(f"  - Best Customer clustering: K-Means (Silhouette={metrics_cust['K-Means']:.3f})")