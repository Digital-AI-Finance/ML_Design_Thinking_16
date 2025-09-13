import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
import pandas as pd
import os

# Create charts directory if it doesn't exist
os.makedirs('charts', exist_ok=True)

# Set style for consistency
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define colors matching the beamer theme
BLUE = '#0066CC'
GREEN = '#00994C'
RED = '#CC0000'
LIGHTBLUE = '#ADD8E6'
GRAY = '#808080'

# Set default font sizes for readability
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def create_comparison_chart():
    """Chart 1: Traditional vs AI Empathy Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Users Reached', 'Time (days)', 'Insights Generated', 'Cost (relative)']
    traditional = [20, 35, 20, 100]
    ai_driven = [100000, 1, 500, 20]
    
    # Log scale for better visualization
    traditional_log = [np.log10(x+1) for x in traditional]
    ai_driven_log = [np.log10(x+1) for x in ai_driven]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, traditional_log, width, label='Traditional', color=GRAY)
    bars2 = ax.bar(x + width/2, ai_driven_log, width, label='AI-Driven', color=BLUE)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scale (log10)')
    ax.set_title('Traditional vs AI-Driven Empathy: Scale Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add value labels on bars
    for bar, val in zip(bars1, traditional):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}', ha='center', va='bottom', fontsize=8)
    
    for bar, val in zip(bars2, ai_driven):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('charts/comparison_chart.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_scale_visualization():
    """Chart 2: Scale Visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create circles representing scale
    scales = [10, 1000, 10000, 100000, 1000000]
    labels = ['10\nInterview\nScale', '1K\nSurvey\nScale', '10K\nAnalytics\nScale', 
              '100K\nBig Data\nScale', '1M\nAI Scale']
    colors = [GRAY, GRAY, LIGHTBLUE, BLUE, GREEN]
    
    # Position circles
    x_positions = np.linspace(1, 9, len(scales))
    
    for i, (scale, label, color, x) in enumerate(zip(scales, labels, colors, x_positions)):
        # Circle size proportional to log of scale
        size = np.log10(scale) * 1000
        circle = plt.Circle((x, 5), np.sqrt(size/np.pi), color=color, alpha=0.6)
        ax.add_patch(circle)
        ax.text(x, 5, label, ha='center', va='center', fontsize=10, weight='bold')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Scale of User Understanding: From Tens to Millions', fontsize=16, pad=20)
    
    # Add arrow showing progression
    ax.annotate('', xy=(9, 2), xytext=(1, 2),
                arrowprops=dict(arrowstyle='->', lw=2, color=BLUE))
    ax.text(5, 1.5, 'AI enables exponential scale increase', ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.savefig('charts/scale_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_nlp_pipeline():
    """Chart 3: NLP Pipeline Flowchart"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define pipeline stages
    stages = [
        ('Raw Text\nData', 0.5, 7, GRAY),
        ('Tokenization\n& Cleaning', 2, 7, LIGHTBLUE),
        ('Feature\nExtraction', 3.5, 7, LIGHTBLUE),
        ('ML\nProcessing', 5, 7, BLUE),
        ('Pattern\nDiscovery', 6.5, 7, BLUE),
        ('Insight\nGeneration', 8, 7, GREEN),
        ('User\nUnderstanding', 9.5, 7, GREEN)
    ]
    
    # Draw boxes and labels
    for label, x, y, color in stages:
        box = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black',
                             alpha=0.7, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw arrows
    arrow_pairs = [(0.5, 2), (2, 3.5), (3.5, 5), (5, 6.5), (6.5, 8), (8, 9.5)]
    for x1, x2 in arrow_pairs:
        ax.annotate('', xy=(x2-0.6, 7), xytext=(x1+0.6, 7),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add process descriptions
    processes = [
        ('10K reviews\n100K sentences', 0.5, 5.5),
        ('Split into\nwords/phrases', 2, 5.5),
        ('TF-IDF\nEmbeddings', 3.5, 5.5),
        ('Sentiment\nClustering', 5, 5.5),
        ('Topics\nSegments', 6.5, 5.5),
        ('Hypotheses\nPriorities', 8, 5.5),
        ('Actionable\nInsights', 9.5, 5.5)
    ]
    
    for label, x, y in processes:
        ax.text(x, y, label, ha='center', va='center', fontsize=8, style='italic')
    
    # Add timing information
    ax.text(5, 4, 'Traditional: 2-3 weeks', ha='center', fontsize=10, color=GRAY)
    ax.text(5, 3.5, 'AI-Powered: 2-3 hours', ha='center', fontsize=10, color=BLUE, weight='bold')
    
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(3, 8)
    ax.axis('off')
    ax.set_title('NLP Pipeline: From Raw Text to User Understanding', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig('charts/nlp_pipeline.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_pattern_heatmap():
    """Chart 4: Pattern Discovery Heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create sample data for pattern discovery
    user_segments = ['Power Users', 'Casual Users', 'New Users', 'Churning Users', 'Returning Users']
    patterns = ['Feature Usage', 'Time Patterns', 'Error Frequency', 'Support Needs', 'Satisfaction']
    
    # Create correlation matrix
    data = np.array([
        [0.9, 0.8, 0.2, 0.3, 0.85],  # Power Users
        [0.4, 0.5, 0.3, 0.4, 0.7],   # Casual Users
        [0.3, 0.3, 0.7, 0.8, 0.5],   # New Users
        [0.2, 0.2, 0.8, 0.9, 0.2],   # Churning Users
        [0.7, 0.6, 0.3, 0.4, 0.8]    # Returning Users
    ])
    
    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(patterns)))
    ax.set_yticks(np.arange(len(user_segments)))
    ax.set_xticklabels(patterns)
    ax.set_yticklabels(user_segments)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pattern Strength', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(user_segments)):
        for j in range(len(patterns)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('AI-Discovered Patterns: User Segments vs Behaviors', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('charts/pattern_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_timeline_comparison():
    """Chart 5: Timeline Comparison"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Traditional timeline
    traditional_tasks = ['Recruit Users', 'Conduct Interviews', 'Transcribe', 'Code Data', 'Analyze', 'Synthesize', 'Report']
    traditional_days = [5, 10, 5, 10, 7, 5, 3]
    traditional_start = [0, 5, 15, 20, 30, 37, 42]
    
    for task, days, start in zip(traditional_tasks, traditional_days, traditional_start):
        ax1.barh(0, days, left=start, height=0.5, color=GRAY, edgecolor='black')
        ax1.text(start + days/2, 0, task, ha='center', va='center', fontsize=8)
    
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_ylabel('Traditional', fontsize=12)
    ax1.set_yticks([])
    ax1.set_title('Research Timeline Comparison', fontsize=14)
    
    # AI timeline
    ai_tasks = ['Data Collection', 'Processing', 'Analysis', 'Insights']
    ai_hours = [2, 1, 2, 1]
    ai_start = [0, 2, 3, 5]
    
    for task, hours, start in zip(ai_tasks, ai_hours, ai_start):
        ax2.barh(0, hours, left=start, height=0.5, color=BLUE, edgecolor='black')
        ax2.text(start + hours/2, 0, task, ha='center', va='center', fontsize=8)
    
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_ylabel('AI-Driven', fontsize=12)
    ax2.set_yticks([])
    ax2.set_xlabel('Time', fontsize=12)
    
    # Adjust x-axis
    ax1.set_xlim(0, 45)
    ax2.set_xlim(0, 45)
    
    # Add time labels
    ax1.text(45, -0.3, '45 days', ha='right', fontsize=10, weight='bold')
    ax2.text(6, -0.3, '6 hours', ha='left', fontsize=10, weight='bold', color=BLUE)
    
    # Add acceleration factor
    fig.text(0.5, 0.02, '180x Faster with AI', ha='center', fontsize=14, weight='bold', color=GREEN)
    
    plt.tight_layout()
    plt.savefig('charts/timeline_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_accuracy_table():
    """Chart 6: Accuracy Comparison Table"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data for comparison
    tasks = ['Explicit\nSentiment', 'Implicit\nFrustration', 'Sarcasm\nDetection', 'Cultural\nNuance', 'Emotion\nRecognition']
    human_accuracy = [85, 45, 60, 30, 55]
    ml_accuracy = [94, 87, 78, 82, 89]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, human_accuracy, width, label='Human Analysis', color=GRAY)
    bars2 = ax.bar(x + width/2, ml_accuracy, width, label='ML Analysis', color=BLUE)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height}%', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Pattern Detection Accuracy: Human vs Machine Learning', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 100)
    
    # Add improvement indicators
    for i, (h, m) in enumerate(zip(human_accuracy, ml_accuracy)):
        improvement = ((m - h) / h) * 100
        ax.text(i, 95, f'+{improvement:.0f}%', ha='center', fontsize=8, color=GREEN, weight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('charts/accuracy_table.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_topic_clustering():
    """Chart 7: Topic Clustering Visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate sample cluster data
    np.random.seed(42)
    n_points = 150
    
    # Create 5 clusters
    clusters = []
    cluster_labels = ['Usability\nIssues', 'Feature\nRequests', 'Performance\nProblems', 
                     'Pricing\nConcerns', 'Positive\nFeedback']
    cluster_colors = [RED, BLUE, GRAY, GREEN, LIGHTBLUE]
    
    centers = [(2, 2), (8, 2), (2, 8), (8, 8), (5, 5)]
    
    for i, (center, label, color) in enumerate(zip(centers, cluster_labels, cluster_colors)):
        x = np.random.normal(center[0], 1.2, 30)
        y = np.random.normal(center[1], 1.2, 30)
        ax.scatter(x, y, alpha=0.6, s=50, color=color, label=label)
        
        # Add cluster center
        ax.scatter(center[0], center[1], s=200, color=color, edgecolor='black', 
                  linewidth=2, marker='s', alpha=0.9)
        ax.text(center[0], center[1], label, ha='center', va='center', 
               fontsize=9, weight='bold')
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.set_xlabel('Semantic Dimension 1', fontsize=12)
    ax.set_ylabel('Semantic Dimension 2', fontsize=12)
    ax.set_title('Topic Modeling: Auto-discovered User Feedback Clusters', fontsize=14)
    
    # Add annotation
    ax.text(5, -0.5, '10,000 user comments automatically categorized in minutes', 
           ha='center', fontsize=10, style='italic')
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('charts/topic_clustering.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_empathy_funnel():
    """Chart 8: Empathy Acceleration Funnel"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define funnel stages
    stages = [
        ('1M+ Data Points', 100, LIGHTBLUE),
        ('100K Processed', 80, LIGHTBLUE),
        ('10K Patterns', 60, BLUE),
        ('1K Insights', 40, BLUE),
        ('100 Personas', 25, GREEN),
        ('10 Key Actions', 10, GREEN)
    ]
    
    y_pos = 0
    for i, (label, width, color) in enumerate(stages):
        # Draw trapezoid for funnel effect
        left = 50 - width/2
        right = 50 + width/2
        
        rect = Rectangle((left, y_pos), width, 1, 
                        facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        
        # Add text
        ax.text(50, y_pos + 0.5, label, ha='center', va='center', 
               fontsize=11, weight='bold')
        
        # Add time annotation on the right
        times = ['1 hour', '2 hours', '3 hours', '4 hours', '5 hours', '6 hours']
        ax.text(105, y_pos + 0.5, times[i], ha='left', va='center', 
               fontsize=9, style='italic')
        
        y_pos += 1.2
    
    ax.set_xlim(0, 120)
    ax.set_ylim(-0.5, 7.5)
    ax.axis('off')
    
    ax.set_title('AI Empathy Funnel: From Big Data to Actionable Insights', 
                fontsize=14, pad=20)
    
    # Add side labels
    ax.text(-5, 3.5, 'Volume', rotation=90, ha='center', va='center', 
           fontsize=12, weight='bold')
    ax.text(115, 3.5, 'Time', ha='center', va='center', 
           fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/empathy_funnel.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all charts
print("Generating charts...")
create_comparison_chart()
print("  - Comparison chart created")
create_scale_visualization()
print("  - Scale visualization created")
create_nlp_pipeline()
print("  - NLP pipeline created")
create_pattern_heatmap()
print("  - Pattern heatmap created")
create_timeline_comparison()
print("  - Timeline comparison created")
create_accuracy_table()
print("  - Accuracy table created")
create_topic_clustering()
print("  - Topic clustering created")
create_empathy_funnel()
print("  - Empathy funnel created")
print("\nAll charts generated successfully in 'charts/' directory!")