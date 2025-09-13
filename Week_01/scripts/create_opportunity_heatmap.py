#!/usr/bin/env python3
"""
Create Innovation Opportunity Heatmap for Week 1 Part 3
Shows intensity of innovation opportunities across different dimensions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Define innovation dimensions and market segments
segments = ['Early\nAdopters', 'Pragmatists', 'Conservatives', 'Skeptics', 'Visionaries']
dimensions = [
    'Technology\nReadiness',
    'Market\nMaturity', 
    'Competition\nIntensity',
    'Resource\nRequirement',
    'Risk\nLevel',
    'Growth\nPotential',
    'Profit\nMargin',
    'Time to\nMarket'
]

# Create opportunity scores (higher = better opportunity)
# Using realistic patterns based on market segments
opportunity_scores = np.array([
    [9, 7, 4, 3, 8],  # Technology Readiness
    [7, 8, 6, 4, 5],  # Market Maturity
    [3, 5, 8, 9, 4],  # Competition Intensity (inverted - lower is better)
    [4, 6, 7, 8, 5],  # Resource Requirement (inverted - lower is better)
    [3, 5, 7, 8, 4],  # Risk Level (inverted - lower is better)
    [9, 7, 5, 3, 9],  # Growth Potential
    [8, 7, 6, 5, 9],  # Profit Margin
    [8, 6, 5, 4, 7],  # Time to Market (inverted - lower is better)
])

# Invert scores where lower is better
opportunity_scores[2, :] = 10 - opportunity_scores[2, :]  # Competition
opportunity_scores[3, :] = 10 - opportunity_scores[3, :]  # Resources
opportunity_scores[4, :] = 10 - opportunity_scores[4, :]  # Risk
opportunity_scores[7, :] = 10 - opportunity_scores[7, :]  # Time to Market

# LEFT PLOT: Raw Opportunity Heatmap
sns.heatmap(opportunity_scores, 
           annot=True, 
           fmt='d',
           cmap='RdYlGn',
           vmin=1, vmax=10,
           xticklabels=segments,
           yticklabels=dimensions,
           cbar_kws={'label': 'Opportunity Score (1-10)'},
           linewidths=1,
           linecolor='white',
           ax=ax1)

ax1.set_title('Innovation Opportunity Heatmap\n(Raw Scores by Segment)', 
             fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Market Segments', fontsize=12, fontweight='bold')
ax1.set_ylabel('Innovation Dimensions', fontsize=12, fontweight='bold')

# Highlight best opportunities (scores >= 8)
for i in range(len(dimensions)):
    for j in range(len(segments)):
        if opportunity_scores[i, j] >= 8:
            rect = Rectangle((j, i), 1, 1, fill=False, 
                           edgecolor='darkgreen', linewidth=3)
            ax1.add_patch(rect)

# RIGHT PLOT: Aggregated Opportunity Analysis
# Calculate composite scores
composite_scores = np.mean(opportunity_scores, axis=0)
weighted_scores = np.average(opportunity_scores, axis=0, 
                            weights=[1.5, 1.2, 1.3, 0.8, 1.0, 1.5, 1.2, 0.9])

# Create bar plot
x = np.arange(len(segments))
width = 0.35

bars1 = ax2.bar(x - width/2, composite_scores, width, 
               label='Average Score', color='#1f77b4', alpha=0.8)
bars2 = ax2.bar(x + width/2, weighted_scores, width,
               label='Weighted Score', color='#ff7f0e', alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

# Mark threshold for high opportunity
ax2.axhline(y=6.5, color='red', linestyle='--', alpha=0.5, label='High Opportunity Threshold')

# Customize right plot
ax2.set_xlabel('Market Segments', fontsize=12, fontweight='bold')
ax2.set_ylabel('Composite Opportunity Score', fontsize=12, fontweight='bold')
ax2.set_title('Aggregated Innovation Opportunities\n(Composite Analysis)', 
             fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(segments)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_ylim(0, 10)
ax2.grid(True, alpha=0.3, axis='y')

# Add strategic recommendations
recommendations = {
    'Early\nAdopters': 'Prime Target',
    'Visionaries': 'High Potential',
    'Pragmatists': 'Good Secondary'
}

for i, (segment, score) in enumerate(zip(segments, weighted_scores)):
    if segment in recommendations:
        ax2.annotate(recommendations[segment],
                    xy=(i, score), xytext=(i, score + 0.5),
                    ha='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1))

# Add insights box on left plot
insights_text = (
    "Key Insights:\n"
    "• 3 Prime opportunity zones\n"
    "• Early Adopters: Best overall\n"
    "• Low competition in emerging segments\n"
    "• High growth potential identified"
)
ax1.text(-1.5, 3, insights_text, fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', 
                 facecolor='lightyellow', alpha=0.9))

# Add opportunity matrix legend
opportunity_levels = [
    (8, 10, 'darkgreen', 'Prime Opportunity'),
    (6, 8, 'green', 'Good Opportunity'),
    (4, 6, 'yellow', 'Moderate Opportunity'),
    (1, 4, 'red', 'Low Opportunity')
]

# Add overall strategy note
fig.text(0.5, 0.02,
        'Innovation Strategy: Focus on high-scoring segments (Early Adopters, Visionaries) with low competition and high growth potential',
        ha='center', fontsize=11, fontweight='bold', style='italic')

plt.tight_layout()

# Save the figure
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/opportunity_heatmap.pdf', 
           dpi=300, bbox_inches='tight')
plt.savefig('D:/Joerg/Research/slides/ML_Design_Thinking_16/Week_01/charts/opportunity_heatmap.png', 
           dpi=150, bbox_inches='tight')

print("Innovation opportunity heatmap created successfully!")
print("Files saved:")
print("  - charts/opportunity_heatmap.pdf")
print("  - charts/opportunity_heatmap.png")