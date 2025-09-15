#!/usr/bin/env python3
"""
Compile Discovery Handout with Automatic Cleanup
Compiles the discovery handout and moves auxiliary files to temp folder
"""

import subprocess
import os
import shutil
import sys
from pathlib import Path

def compile_handout(tex_file='week01_discovery_handout.tex', runs=2):
    """
    Compile handout LaTeX file and clean up auxiliary files
    
    Args:
        tex_file: Name of the handout .tex file
        runs: Number of compilation runs (2 for references)
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
    aux_extensions = ['.aux', '.log', '.out', '.toc', '.fls', 
                     '.fdb_latexmk', '.synctex.gz']
    
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
        print(f"\n✓ Handout PDF created successfully: {os.path.abspath(pdf_file)}")
        if moved_files:
            print(f"✓ Moved {len(moved_files)} auxiliary files to {temp_dir}/")
        
        # Get file size
        size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
        print(f"✓ File size: {size_mb:.2f} MB")
        
        # Count pages (simple method - count Page objects in PDF)
        try:
            with open(pdf_file, 'rb') as f:
                content = f.read()
                pages = content.count(b'/Type /Page')
                print(f"✓ Number of pages: {pages}")
        except:
            pass
            
        return True
    else:
        print(f"\n✗ PDF creation failed")
        return False

def create_instructor_version():
    """
    Create a filled instructor version with sample answers
    """
    print("\nCreating instructor version with sample answers...")
    
    # Read the original handout
    with open('week01_discovery_handout.tex', 'r') as f:
        content = f.read()
    
    # Create instructor version with some filled examples
    instructor_content = content.replace(
        r'\textbf{Name:} \hrulefill',
        r'\textbf{Name:} \textcolor{red}{INSTRUCTOR VERSION}'
    )
    
    # Save instructor version
    instructor_file = 'week01_discovery_handout_instructor.tex'
    with open(instructor_file, 'w') as f:
        f.write(instructor_content)
    
    # Compile instructor version
    print("Compiling instructor version...")
    if compile_handout(instructor_file):
        print("✓ Instructor version created successfully")
        # Clean up the tex file but keep the PDF
        os.remove(instructor_file)

def main():
    """Main function"""
    print("="*50)
    print("Discovery Handout Compilation")
    print("="*50)
    
    # Compile student version
    print("\n1. Compiling student version...")
    success = compile_handout()
    
    if success:
        # Optionally create instructor version
        response = input("\nCreate instructor version? (y/n): ")
        if response.lower() == 'y':
            create_instructor_version()
    
    print("\n" + "="*50)
    print("Compilation complete!")
    print("="*50)

if __name__ == "__main__":
    main()