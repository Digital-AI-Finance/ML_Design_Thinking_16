#!/usr/bin/env python3
"""
LaTeX Slide Compilation with Automatic Cleanup
Compiles LaTeX files and moves auxiliary files to temp folder
"""

import subprocess
import os
import shutil
import sys
import glob
from pathlib import Path

def compile_and_cleanup(tex_file, runs=2):
    """
    Compile LaTeX file and clean up auxiliary files
    
    Args:
        tex_file: Path to .tex file
        runs: Number of compilation runs (2 for TOC/references)
    """
    if not os.path.exists(tex_file):
        print(f"Error: {tex_file} not found")
        return False
    
    base_name = os.path.splitext(tex_file)[0]
    
    # Compile LaTeX (run multiple times for references)
    for i in range(runs):
        print(f"Compilation run {i+1}/{runs}...")
        result = subprocess.run(['pdflatex', tex_file], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error during compilation:")
            print(result.stdout[-2000:])  # Last 2000 chars of output
            return False
    
    # Create temp folder if it doesn't exist
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Extensions to move
    aux_extensions = ['.aux', '.log', '.nav', '.out', '.snm', '.toc', '.vrb', 
                     '.fls', '.fdb_latexmk', '.synctex.gz', '.bbl', '.blg']
    
    # Move auxiliary files
    moved_files = []
    for ext in aux_extensions:
        aux_file = base_name + ext
        if os.path.exists(aux_file):
            try:
                shutil.move(aux_file, os.path.join(temp_dir, os.path.basename(aux_file)))
                moved_files.append(os.path.basename(aux_file))
            except Exception as e:
                print(f"Warning: Could not move {aux_file}: {e}")
    
    # Report results
    pdf_file = base_name + '.pdf'
    if os.path.exists(pdf_file):
        print(f"\n✓ PDF created successfully: {os.path.abspath(pdf_file)}")
        if moved_files:
            print(f"✓ Moved {len(moved_files)} auxiliary files to {temp_dir}/")
        return True
    else:
        print(f"\n✗ PDF creation failed")
        return False

def compile_all_weeks():
    """Compile all week slides in the directory"""
    week_files = glob.glob('week*.tex')
    
    if not week_files:
        print("No week*.tex files found")
        return
    
    print(f"Found {len(week_files)} week files to compile")
    
    for tex_file in sorted(week_files):
        print(f"\n{'='*50}")
        print(f"Compiling {tex_file}...")
        print(f"{'='*50}")
        compile_and_cleanup(tex_file)

def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Compile specific file
        tex_file = sys.argv[1]
        if not tex_file.endswith('.tex'):
            tex_file += '.tex'
        compile_and_cleanup(tex_file)
    else:
        # Default: compile week01_slides.tex if it exists
        if os.path.exists('week01_slides.tex'):
            compile_and_cleanup('week01_slides.tex')
        else:
            print("Usage: python compile_slides.py [filename.tex]")
            print("Or place in directory with week01_slides.tex")
            response = input("\nCompile all week*.tex files? (y/n): ")
            if response.lower() == 'y':
                compile_all_weeks()

if __name__ == "__main__":
    main()