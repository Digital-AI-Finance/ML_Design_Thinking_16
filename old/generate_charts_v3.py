import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from itertools import combinations
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Create charts directory if it doesn't exist
os.makedirs('charts', exist_ok=True)

# Set style for consistency
plt.style.use('seaborn-v0_8-whitegrid')

# Define beautiful color palette
BLUE = '#0066CC'
GREEN = '#00994C'
RED = '#CC0000'
ORANGE = '#FF9900'
PURPLE = '#9933CC'
GRAY = '#808080'
CYAN = '#00CCCC'
MAGENTA = '#CC00CC'

# Set default font sizes for readability
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def create_neural_network_learning():
    """Chart 1: Neural Network Learning Dynamics - Beautiful visualization"""
    np.random.seed(42)
    
    # Generate synthetic spiral dataset (visually interesting)
    n_samples = 1500
    noise = 0.1
    
    # Create two spirals
    theta = np.sqrt(np.random.rand(n_samples // 2)) * 4 * np.pi
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    data_a += np.random.randn(n_samples // 2, 2) * noise
    
    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    data_b += np.random.randn(n_samples // 2, 2) * noise
    
    X = np.vstack([data_a, data_b])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train neural network and track performance
    epochs = 100
    train_losses = []
    test_accuracies = []
    models_at_epochs = {}
    
    for epoch in range(1, epochs + 1):
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=epoch, 
                           random_state=42, alpha=0.001, learning_rate_init=0.01)
        mlp.fit(X_train, y_train)
        
        # Calculate loss (approximated by 1 - score)
        train_score = mlp.score(X_train, y_train)
        train_losses.append(1 - train_score)
        test_accuracies.append(mlp.score(X_test, y_test))
        
        # Save models at specific epochs for visualization
        if epoch in [1, 10, 50, 100]:
            models_at_epochs[epoch] = mlp
    
    # Create visualization
    fig = plt.figure(figsize=(15, 7))
    
    # Left panel: Learning curves
    ax1 = plt.subplot(1, 2, 1)
    
    # Plot with gradient fill
    epochs_range = np.arange(1, epochs + 1)
    
    # Training loss
    ax1.plot(epochs_range, train_losses, color=RED, linewidth=2, label='Training Loss')
    ax1.fill_between(epochs_range, train_losses, alpha=0.3, color=RED)
    
    # Test accuracy (on secondary axis)
    ax2 = ax1.twinx()
    ax2.plot(epochs_range, test_accuracies, color=GREEN, linewidth=2, label='Test Accuracy')
    ax2.fill_between(epochs_range, test_accuracies, alpha=0.3, color=GREEN)
    
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Loss', color=RED)
    ax2.set_ylabel('Accuracy', color=GREEN)
    ax1.set_title('Neural Network Learning Dynamics')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    # Right panel: Decision boundary evolution
    for idx, (epoch, model) in enumerate(models_at_epochs.items()):
        ax = plt.subplot(2, 4, 5 + idx)
        
        # Create mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Predict on mesh
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.8)
        
        # Plot data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', 
                           edgecolors='white', linewidth=0.5, s=20, alpha=0.7)
        
        ax.set_title(f'Epoch {epoch}')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('How Neural Networks Learn Complex Patterns', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('charts/neural_network_learning.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("- Created neural_network_learning.pdf (Beautiful NN visualization)")

def create_tsne_visualization():
    """Chart 2: t-SNE Dimensionality Reduction - Stunning cluster discovery"""
    np.random.seed(42)
    
    # Generate high-dimensional data with hidden structure
    n_samples = 3000
    n_features = 50
    n_clusters = 6
    
    # Create clusters in high-dimensional space
    X_high, y_true = make_blobs(n_samples=n_samples, n_features=n_features,
                                centers=n_clusters, cluster_std=2.0, random_state=42)
    
    # Add noise to make it more realistic
    X_high += np.random.normal(0, 0.5, X_high.shape)
    
    # Apply PCA for comparison
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_high)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_high)
    
    # Define cluster names for visualization
    cluster_names = ['Power Users', 'Explorers', 'Casual Users', 
                    'New Users', 'Champions', 'At Risk']
    
    # Create beautiful visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    # Left panel: PCA projection
    for i in range(n_clusters):
        mask = y_true == i
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[i]], 
                   label=cluster_names[i], alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    ax1.set_title('Linear Projection (PCA)', fontsize=14)
    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Add explained variance
    var_explained = pca.explained_variance_ratio_
    ax1.text(0.02, 0.98, f'Variance explained: {var_explained.sum():.1%}',
            transform=ax1.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Right panel: t-SNE projection
    for i in range(n_clusters):
        mask = y_true == i
        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[colors[i]], 
                   label=cluster_names[i], alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
    
    ax2.set_title('Non-linear Projection (t-SNE)', fontsize=14)
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Add description
    ax2.text(0.02, 0.98, 'Hidden structure revealed!',
            transform=ax2.transAxes, fontsize=10, va='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('Discovering Hidden Structure in High-Dimensional Data', fontsize=16)
    plt.tight_layout()
    plt.savefig('charts/tsne_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("- Created tsne_visualization.pdf (Stunning dimensionality reduction)")

def create_deep_learning_advantage():
    """Chart 7: The Deep Learning Advantage - Data scaling visualization"""
    np.random.seed(42)
    
    # Data points on log scale
    data_sizes = np.logspace(1, 6, 50)  # 10 to 1,000,000
    
    # Traditional ML performance (plateaus)
    traditional_ml = 0.7 + 0.15 * (1 - np.exp(-data_sizes / 5000))
    traditional_ml += np.random.normal(0, 0.01, len(data_sizes))
    
    # Shallow learning (moderate improvement)
    shallow_ml = 0.65 + 0.25 * (1 - np.exp(-data_sizes / 20000))
    shallow_ml += np.random.normal(0, 0.01, len(data_sizes))
    
    # Deep learning (continuous improvement)
    deep_learning = 0.60 + 0.35 * (1 - np.exp(-data_sizes / 100000))
    deep_learning += 0.05 * np.log10(data_sizes) / np.log10(1000000)
    deep_learning += np.random.normal(0, 0.01, len(data_sizes))
    
    # Ensure values stay in valid range
    traditional_ml = np.clip(traditional_ml, 0, 1)
    shallow_ml = np.clip(shallow_ml, 0, 1)
    deep_learning = np.clip(deep_learning, 0, 1)
    
    # Create confidence bands
    confidence_width = 0.02
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot with gradient fills
    ax.semilogx(data_sizes, traditional_ml, color=GRAY, linewidth=2.5, label='Traditional ML')
    ax.fill_between(data_sizes, traditional_ml - confidence_width, 
                    traditional_ml + confidence_width, alpha=0.2, color=GRAY)
    
    ax.semilogx(data_sizes, shallow_ml, color=ORANGE, linewidth=2.5, label='Shallow Neural Networks')
    ax.fill_between(data_sizes, shallow_ml - confidence_width, 
                    shallow_ml + confidence_width, alpha=0.2, color=ORANGE)
    
    ax.semilogx(data_sizes, deep_learning, color=BLUE, linewidth=2.5, label='Deep Learning')
    ax.fill_between(data_sizes, deep_learning - confidence_width, 
                    deep_learning + confidence_width, alpha=0.2, color=BLUE)
    
    # Add performance zones
    ax.axhspan(0.9, 1.0, alpha=0.05, color='green')
    ax.axhspan(0.8, 0.9, alpha=0.05, color='yellow')
    ax.axhspan(0.7, 0.8, alpha=0.05, color='orange')
    
    # Add annotations
    ax.annotate('Traditional ML Plateau', 
               xy=(50000, traditional_ml[np.argmin(np.abs(data_sizes - 50000))]),
               xytext=(5000, 0.82),
               arrowprops=dict(arrowstyle='->', color=GRAY, alpha=0.7))
    
    ax.annotate('Deep Learning Advantage', 
               xy=(500000, deep_learning[np.argmin(np.abs(data_sizes - 500000))]),
               xytext=(50000, 0.95),
               arrowprops=dict(arrowstyle='->', color=BLUE, alpha=0.7, lw=2))
    
    # Vertical lines for data scale milestones
    milestones = [100, 1000, 10000, 100000, 1000000]
    for milestone in milestones:
        ax.axvline(milestone, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('Training Data Size (log scale)', fontsize=12)
    ax.set_ylabel('Model Performance', fontsize=12)
    ax.set_title('The Deep Learning Advantage: Performance Scales with Data', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.55, 1.0])
    
    # Add text labels for zones
    ax.text(15, 0.95, 'State of the Art', fontsize=9, style='italic', color='green')
    ax.text(15, 0.85, 'Production Ready', fontsize=9, style='italic', color='orange')
    ax.text(15, 0.75, 'Acceptable', fontsize=9, style='italic', color='red')
    
    plt.tight_layout()
    plt.savefig('charts/deep_learning_advantage.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("- Created deep_learning_advantage.pdf (Beautiful scaling visualization)")

def create_emotion_wheel():
    """Chart 8: AI Understanding Emotions at Scale - Emotion wheel visualization"""
    np.random.seed(42)
    
    # Define 8 primary emotions
    emotions = ['Joy', 'Trust', 'Fear', 'Surprise', 
               'Sadness', 'Disgust', 'Anger', 'Anticipation']
    
    # Generate data for 10,000 reviews (AI analysis)
    n_reviews = 10000
    # Create realistic emotion distribution (some emotions more common)
    ai_distribution = np.array([
        np.random.gamma(4, 2, n_reviews),  # Joy - common
        np.random.gamma(3, 2, n_reviews),  # Trust
        np.random.gamma(2, 1.5, n_reviews),  # Fear
        np.random.gamma(2.5, 1.5, n_reviews),  # Surprise
        np.random.gamma(2, 2, n_reviews),  # Sadness
        np.random.gamma(1.5, 1, n_reviews),  # Disgust - less common
        np.random.gamma(2, 1.5, n_reviews),  # Anger
        np.random.gamma(3.5, 2, n_reviews),  # Anticipation
    ])
    
    # Normalize to percentages
    ai_percentages = (ai_distribution.mean(axis=1) / ai_distribution.mean(axis=1).sum()) * 100
    
    # Human annotation (100 samples) - more variable
    human_percentages = ai_percentages + np.random.normal(0, 3, len(emotions))
    human_percentages = np.clip(human_percentages, 0, 100)
    human_percentages = (human_percentages / human_percentages.sum()) * 100
    
    # Create polar plot
    fig = plt.figure(figsize=(14, 7))
    
    # Left: Radial emotion wheel
    ax1 = plt.subplot(121, projection='polar')
    
    # Set up angles for each emotion
    angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist()
    
    # Close the plot
    ai_values = ai_percentages.tolist()
    human_values = human_percentages.tolist()
    angles += angles[:1]
    ai_values += ai_values[:1]
    human_values += human_values[:1]
    
    # Plot
    ax1.plot(angles, ai_values, 'o-', linewidth=2, color=BLUE, label='AI Analysis (10K samples)')
    ax1.fill(angles, ai_values, alpha=0.25, color=BLUE)
    
    ax1.plot(angles, human_values, 's-', linewidth=2, color=ORANGE, label='Human Analysis (100 samples)')
    ax1.fill(angles, human_values, alpha=0.25, color=ORANGE)
    
    # Set emotion labels
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(emotions)
    ax1.set_ylim(0, max(max(ai_values), max(human_values)) + 5)
    
    # Add grid
    ax1.grid(True)
    ax1.set_title('Emotion Distribution Analysis', fontsize=14, pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Right: Detailed comparison
    ax2 = plt.subplot(122)
    
    x = np.arange(len(emotions))
    width = 0.35
    
    # Create bars with gradient effect
    bars1 = ax2.bar(x - width/2, ai_percentages, width, label='AI (10,000 reviews)', 
                    color=BLUE, alpha=0.7, edgecolor='navy', linewidth=2)
    bars2 = ax2.bar(x + width/2, human_percentages, width, label='Human (100 reviews)', 
                    color=ORANGE, alpha=0.7, edgecolor='darkorange', linewidth=2)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Emotions')
    ax2.set_ylabel('Percentage of Reviews (%)')
    ax2.set_title('AI vs Human Emotion Detection', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(emotions, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add insight box
    ax2.text(0.02, 0.95, f'AI processes 100x more data\nwith consistent accuracy',
            transform=ax2.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Understanding Human Emotions at Scale', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('charts/emotion_wheel.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("- Created emotion_wheel.pdf (Beautiful emotion analysis visualization)")

# Keep these existing functions unchanged
def create_clustering_results():
    """Chart 3: User Segmentation via K-means Clustering"""
    np.random.seed(42)
    
    # Generate synthetic user data with clear clusters
    n_users = 1000
    
    # Create 5 distinct user segments
    segments = []
    labels = []
    
    # Segment 1: Young tech-savvy (high engagement, high feature use)
    seg1 = np.random.multivariate_normal([7, 8], [[0.5, 0.2], [0.2, 0.5]], 200)
    segments.append(seg1)
    labels.extend(['Tech Enthusiasts'] * 200)
    
    # Segment 2: Casual users (medium engagement, low feature use)
    seg2 = np.random.multivariate_normal([5, 3], [[0.6, 0.1], [0.1, 0.6]], 300)
    segments.append(seg2)
    labels.extend(['Casual Users'] * 300)
    
    # Segment 3: Power users (high engagement, high feature use)
    seg3 = np.random.multivariate_normal([8, 9], [[0.4, 0.15], [0.15, 0.4]], 150)
    segments.append(seg3)
    labels.extend(['Power Users'] * 150)
    
    # Segment 4: Struggling users (low engagement, low feature use)
    seg4 = np.random.multivariate_normal([2, 2], [[0.5, 0.1], [0.1, 0.5]], 200)
    segments.append(seg4)
    labels.extend(['Need Support'] * 200)
    
    # Segment 5: Explorer users (variable engagement, high exploration)
    seg5 = np.random.multivariate_normal([6, 7], [[1.2, 0.3], [0.3, 0.8]], 150)
    segments.append(seg5)
    labels.extend(['Explorers'] * 150)
    
    # Combine all segments
    X = np.vstack(segments)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Actual clustering results
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', 
                         alpha=0.6, s=30)
    ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
               c='red', marker='*', s=300, edgecolors='black', linewidth=2)
    ax1.set_xlabel('Engagement Score')
    ax1.set_ylabel('Feature Usage Score')
    ax1.set_title('K-means Clustering: 5 User Segments (n=1000)')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Segment sizes and characteristics
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    colors_bar = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    bars = ax2.bar(unique_labels, counts, color=colors_bar)
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Number of Users')
    ax2.set_title('Cluster Sizes and Characteristics')
    ax2.set_xticks(unique_labels)
    
    # Add percentages
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        ax2.text(i, count + 10, f'{count/10:.1f}%', ha='center')
    
    # Add legend with segment descriptions
    segment_names = ['Segment 0: Low Activity', 'Segment 1: Moderate Use', 
                    'Segment 2: High Engagement', 'Segment 3: Power Users', 
                    'Segment 4: Exploratory']
    ax2.legend(bars, segment_names, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('charts/clustering_results.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("- Created clustering_results.pdf")

def create_ml_accuracy_timeline():
    """Chart 4: ML Model Accuracy Improvement Over Time"""
    # Simulate model performance over iterations
    iterations = np.arange(0, 100, 5)
    
    # Simulate accuracy improvement with diminishing returns
    accuracy = 0.65 + 0.30 * (1 - np.exp(-iterations/25))
    precision = 0.60 + 0.35 * (1 - np.exp(-iterations/20))
    recall = 0.70 + 0.25 * (1 - np.exp(-iterations/30))
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Add some realistic noise
    np.random.seed(42)
    accuracy += np.random.normal(0, 0.01, len(iterations))
    precision += np.random.normal(0, 0.015, len(iterations))
    recall += np.random.normal(0, 0.01, len(iterations))
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(iterations, accuracy, 'o-', label='Accuracy', color=BLUE, linewidth=2)
    ax.plot(iterations, precision, 's-', label='Precision', color=GREEN, linewidth=2)
    ax.plot(iterations, recall, '^-', label='Recall', color=ORANGE, linewidth=2)
    ax.plot(iterations, f1_score, 'd-', label='F1 Score', color=RED, linewidth=2)
    
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics Over Training Time')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.55, 1.0])
    
    # Add annotations for key milestones
    ax.annotate('Initial Model', xy=(0, accuracy[0]), xytext=(10, 0.58),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))
    ax.annotate('Production Ready', xy=(50, accuracy[10]), xytext=(60, 0.85),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))
    
    # Add performance zones
    ax.axhspan(0.9, 1.0, alpha=0.1, color='green', label='Excellent')
    ax.axhspan(0.8, 0.9, alpha=0.1, color='yellow')
    ax.axhspan(0.7, 0.8, alpha=0.1, color='orange')
    
    plt.tight_layout()
    plt.savefig('charts/ml_accuracy_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("- Created ml_accuracy_timeline.pdf")

def create_topic_modeling_results():
    """Chart 5: LDA Topic Discovery from 5000 Documents"""
    np.random.seed(42)
    
    # Simulate topic modeling results
    topics = ['User Experience', 'Performance', 'Pricing', 'Features', 
              'Customer Service', 'Quality', 'Innovation', 'Reliability']
    
    # Document distribution across topics (realistic distribution)
    doc_counts = [850, 720, 680, 630, 520, 480, 420, 700]
    
    # Top words per topic (simulated coherence scores)
    coherence_scores = [0.72, 0.68, 0.75, 0.71, 0.69, 0.73, 0.67, 0.70]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Document distribution
    colors = plt.cm.Set3(np.linspace(0, 1, len(topics)))
    bars = ax1.barh(topics, doc_counts, color=colors)
    ax1.set_xlabel('Number of Documents')
    ax1.set_title('Topic Distribution Across 5000 Documents')
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Add counts on bars
    for i, (topic, count) in enumerate(zip(topics, doc_counts)):
        ax1.text(count + 20, i, f'{count} ({count/50:.1f}%)', va='center')
    
    # Right: Topic coherence scores
    ax2.scatter(coherence_scores, topics, s=200, alpha=0.6, c=colors)
    ax2.set_xlabel('Topic Coherence Score')
    ax2.set_title('Topic Quality Metrics')
    ax2.set_xlim([0.6, 0.8])
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0.7, color='red', linestyle='--', alpha=0.5, label='Quality Threshold')
    ax2.legend()
    
    # Add interpretation note
    fig.text(0.5, -0.02, 'Higher coherence = more meaningful topic grouping', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('charts/topic_modeling.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("- Created topic_modeling.pdf")

def create_cost_analysis():
    """Chart 6: Cost Analysis - Traditional vs AI-Powered Research"""
    # Cost components (in USD)
    components = ['Labor\nHours', 'Tools &\nSoftware', 'Data\nCollection', 
                  'Analysis\nTime', 'Validation', 'Total']
    
    # Traditional costs (manual process)
    traditional = [15000, 500, 3000, 8000, 4000, 30500]
    
    # AI-powered costs
    ai_powered = [2000, 2500, 500, 1000, 1500, 7500]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Side-by-side comparison
    x = np.arange(len(components))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, traditional, width, label='Traditional', color=GRAY)
    bars2 = ax1.bar(x + width/2, ai_powered, width, label='AI-Powered', color=BLUE)
    
    ax1.set_xlabel('Cost Component')
    ax1.set_ylabel('Cost (USD)')
    ax1.set_title('Cost Breakdown: Traditional vs AI-Powered Analysis')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=8)
    
    # Right: Savings visualization
    savings = [t - a for t, a in zip(traditional, ai_powered)]
    colors_savings = [GREEN if s > 0 else RED for s in savings]
    
    bars3 = ax2.bar(components, savings, color=colors_savings)
    ax2.set_xlabel('Cost Component')
    ax2.set_ylabel('Savings (USD)')
    ax2.set_title('Cost Savings with AI Implementation')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.axhline(0, color='black', linewidth=0.5)
    
    # Add percentage savings
    for i, (comp, save, trad) in enumerate(zip(components, savings, traditional)):
        if trad > 0:
            pct = (save / trad) * 100
            ax2.text(i, save + 500 if save > 0 else save - 500,
                    f'{pct:.0f}%', ha='center', fontsize=9)
    
    # Add ROI calculation
    total_investment = ai_powered[-1]
    total_savings = traditional[-1] - ai_powered[-1]
    roi = (total_savings / total_investment) * 100
    
    ax2.text(0.02, 0.95, f'ROI: {roi:.0f}%', transform=ax2.transAxes,
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    plt.tight_layout()
    plt.savefig('charts/cost_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("- Created cost_analysis.pdf")

def main():
    """Generate all charts"""
    print("\nGenerating stunning ML visualization charts for Week 1...")
    print("-" * 50)
    
    # New beautiful charts
    create_neural_network_learning()
    create_tsne_visualization()
    
    # Keep these good ones
    create_clustering_results()
    create_ml_accuracy_timeline()
    create_topic_modeling_results()
    create_cost_analysis()
    
    # New replacements
    create_deep_learning_advantage()
    create_emotion_wheel()
    
    print("-" * 50)
    print("All charts generated successfully!")
    print("Location: charts/")
    print("\nCharts created:")
    print("1. neural_network_learning.pdf - Beautiful NN learning dynamics")
    print("2. tsne_visualization.pdf - Stunning dimensionality reduction")
    print("3. clustering_results.pdf - K-means clustering visualization")
    print("4. ml_accuracy_timeline.pdf - Model performance over time")
    print("5. topic_modeling.pdf - LDA topic discovery")
    print("6. cost_analysis.pdf - ROI analysis")
    print("7. deep_learning_advantage.pdf - Deep learning scaling power")
    print("8. emotion_wheel.pdf - Emotion analysis at scale")

if __name__ == "__main__":
    main()