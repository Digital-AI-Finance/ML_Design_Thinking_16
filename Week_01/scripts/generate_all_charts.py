#!/usr/bin/env python3
"""
Generate all visualization charts for Week 1 slides
Run this script to create all PDFs at once
"""

import subprocess
import os
import sys
from pathlib import Path

# List of all visualization scripts to run
scripts = [
    'convergence_flow.py',
    'chaos_to_clarity.py',
    'elbow_method.py',
    'dendrogram_example.py',
    'pca_clusters.py',
    'empathy_map_clusters.py',
    'user_empathy_visual.py',
    'clustering_examples.py',
    'distance_visual.py',
    'kmeans_animation.py',
    'dendrogram_cut.py',
    'cluster_quality.py',
    'dbscan_shapes.py',
    'feature_importance.py',
    'customer_segments.py',
    'pain_points_heatmap.py',
    'behavior_patterns.py',
    'journey_map_clusters.py',
    'stakeholder_network.py',
    'persona_cards.py',
    'design_priority_matrix.py',
    'clustering_methods_comparison.py',
    'spotify_clustering.py',
    'week2_preview.py'
]

def run_script(script_name):
    """Run a single Python script"""
    if not os.path.exists(script_name):
        print(f"⚠️  {script_name} not found - skipping")
        return False
    
    print(f"Running {script_name}...")
    try:
        result = subprocess.run(['python', script_name], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✓ {script_name} completed successfully")
            return True
        else:
            print(f"✗ {script_name} failed:")
            print(result.stderr[:500])
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {script_name} timed out")
        return False
    except Exception as e:
        print(f"✗ {script_name} error: {e}")
        return False

def main():
    """Generate all charts"""
    print("="*60)
    print("GENERATING ALL WEEK 1 VISUALIZATION CHARTS")
    print("="*60)
    
    # Check if we're in the right directory
    current_dir = os.getcwd()
    if not current_dir.endswith('week01_visuals'):
        print("Warning: Not in week01_visuals directory")
        print(f"Current directory: {current_dir}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Track results
    successful = []
    failed = []
    skipped = []
    
    # Run each script
    for script in scripts:
        if os.path.exists(script):
            if run_script(script):
                successful.append(script)
            else:
                failed.append(script)
        else:
            skipped.append(script)
    
    # Report results
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Successful: {len(successful)} scripts")
    print(f"✗ Failed: {len(failed)} scripts")
    print(f"⚠ Skipped: {len(skipped)} scripts")
    
    if failed:
        print("\nFailed scripts:")
        for script in failed:
            print(f"  - {script}")
    
    if skipped:
        print("\nSkipped scripts (not found):")
        for script in skipped:
            print(f"  - {script}")
    
    # Check for generated PDFs
    pdfs = list(Path('.').glob('*.pdf'))
    print(f"\n📄 Generated {len(pdfs)} PDF files")
    
    print("\n" + "="*60)
    print("Chart generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()