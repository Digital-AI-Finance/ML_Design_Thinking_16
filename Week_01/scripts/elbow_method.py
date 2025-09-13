import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Generate sample data with 4 natural clusters
X, _ = make_blobs(n_samples=500, centers=4, n_features=2, 
                  cluster_std=1.0, random_state=42)

# Calculate inertias for different K values
K_range = range(1, 11)
inertias = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the elbow curve
ax.plot(K_range, inertias, 'b-', linewidth=2, label='Total Within-Cluster Sum of Squares')
ax.plot(K_range, inertias, 'bo', markersize=8)

# Highlight the elbow point
elbow_k = 4
elbow_idx = elbow_k - 1
ax.plot(elbow_k, inertias[elbow_idx], 'r*', markersize=20, 
        label=f'Optimal K = {elbow_k}', zorder=5)

# Add annotation for the elbow
ax.annotate('Elbow Point\n(Optimal K)', 
           xy=(elbow_k, inertias[elbow_idx]),
           xytext=(elbow_k + 1.5, inertias[elbow_idx] + 100),
           arrowprops=dict(arrowstyle='->', color='red', lw=2),
           fontsize=12, fontweight='bold', color='red')

# Add shaded regions to show improvement zones
ax.axvspan(1, 3, alpha=0.2, color='red', label='High improvement')
ax.axvspan(3, 5, alpha=0.2, color='orange', label='Good improvement')
ax.axvspan(5, 10, alpha=0.2, color='gray', label='Diminishing returns')

# Labels and styling
ax.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Within-Cluster Sum of Squares', fontsize=12, fontweight='bold')
ax.set_title('The Elbow Method: Finding Optimal Number of Clusters', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10)

# Add text boxes with insights
ax.text(2, inertias[0] - 50, 'Big drops:\nMajor structure', 
        fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.5))
ax.text(4, inertias[3] - 50, 'Elbow:\nNatural K', 
        fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.5))
ax.text(7.5, inertias[7] + 50, 'Flattening:\nOver-clustering', 
        fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='#cccccc', alpha=0.5))

# Set x-axis to show all K values
ax.set_xticks(K_range)
ax.set_xlim(0.5, 10.5)

plt.tight_layout()
plt.savefig('elbow_method.pdf', dpi=300, bbox_inches='tight')
plt.savefig('elbow_method.png', dpi=150, bbox_inches='tight')
plt.close()

print("Elbow method visualization created!")