#!/usr/bin/env python3
"""
Font Size Consistency Checker for LaTeX Beamer Presentations

This script verifies that a LaTeX presentation uses at most 3 different font sizes.
It includes unit tests to ensure the checking logic works correctly.

USAGE:
    python check_font_sizes.py <tex_file>

EXAMPLE:
    python check_font_sizes.py ML_Design_Course/20250912_0830_course_overview_10week.tex

REQUIREMENTS:
    - Python 3.6+
    - No external dependencies

FUNCTIONALITY:
1. Scans LaTeX file for font size commands
2. Reports all unique font sizes found
3. Validates that at most 3 different sizes are used
4. Provides line numbers for each font size occurrence
5. Includes comprehensive unit tests

ALLOWED FONT SIZES (pick 3):
    - \normalsize (standard body text)
    - \Large (section headers, titles)
    - \small (tables, code, fine print)

NOT RECOMMENDED:
    - \Huge, \huge (too large)
    - \tiny, \scriptsize (too small)
    - \footnotesize (hard to read on slides)
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import unittest
from io import StringIO


class FontSizeChecker:
    """Check font size consistency in LaTeX files"""
    
    # All possible LaTeX font size commands
    FONT_SIZES = [
        'tiny', 'scriptsize', 'footnotesize', 'small', 
        'normalsize', 'large', 'Large', 'LARGE', 
        'huge', 'Huge'
    ]
    
    # Recommended font sizes for presentations (pick 3)
    RECOMMENDED_SIZES = {'normalsize', 'Large', 'small'}
    
    def __init__(self, max_sizes: int = 3):
        """
        Initialize the font size checker
        
        Args:
            max_sizes: Maximum number of different font sizes allowed
        """
        self.max_sizes = max_sizes
        self.font_pattern = re.compile(
            r'\\(' + '|'.join(self.FONT_SIZES) + r')(?=\s|{|\\|$)',
            re.IGNORECASE
        )
    
    def check_file(self, filepath: str) -> Tuple[bool, Dict[str, List[int]], Set[str]]:
        """
        Check a LaTeX file for font size consistency
        
        Args:
            filepath: Path to the LaTeX file
            
        Returns:
            Tuple of (passes_check, font_locations, unique_sizes)
            - passes_check: True if <= max_sizes different sizes
            - font_locations: Dict mapping font size to line numbers
            - unique_sizes: Set of unique font sizes found
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        font_locations = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            # Skip commented lines
            if line.strip().startswith('%'):
                continue
            
            # Find all font size commands in this line
            matches = self.font_pattern.findall(line)
            for match in matches:
                font_size = match.lower()
                if font_size not in font_locations:
                    font_locations[font_size] = []
                font_locations[font_size].append(line_num)
        
        unique_sizes = set(font_locations.keys())
        passes_check = len(unique_sizes) <= self.max_sizes
        
        return passes_check, font_locations, unique_sizes
    
    def generate_report(self, filepath: str) -> str:
        """
        Generate a detailed report of font sizes in the file
        
        Args:
            filepath: Path to the LaTeX file
            
        Returns:
            String report of font size analysis
        """
        passes, locations, unique_sizes = self.check_file(filepath)
        
        report = []
        report.append(f"Font Size Analysis for: {filepath}")
        report.append("=" * 60)
        report.append(f"Total unique font sizes: {len(unique_sizes)}")
        report.append(f"Maximum allowed: {self.max_sizes}")
        report.append(f"Status: {'PASS' if passes else 'FAIL'}")
        report.append("")
        
        if unique_sizes:
            report.append("Font sizes found:")
            for size in sorted(unique_sizes):
                count = len(locations[size])
                report.append(f"  \\{size}: {count} occurrences")
                # Show first 5 line numbers
                line_nums = locations[size][:5]
                if len(locations[size]) > 5:
                    report.append(f"    Lines: {', '.join(map(str, line_nums))}...")
                else:
                    report.append(f"    Lines: {', '.join(map(str, line_nums))}")
        
        report.append("")
        report.append("Recommendations:")
        if not passes:
            report.append(f"  - Reduce to {self.max_sizes} font sizes maximum")
            report.append(f"  - Suggested sizes: {', '.join('\\' + s for s in self.RECOMMENDED_SIZES)}")
            
            # Suggest which sizes to eliminate
            extra_sizes = unique_sizes - self.RECOMMENDED_SIZES
            if extra_sizes:
                report.append(f"  - Consider removing: {', '.join('\\' + s for s in extra_sizes)}")
        else:
            report.append("  - Font size usage is consistent!")
            
        return '\n'.join(report)
    
    def fix_font_sizes(self, filepath: str, output_path: str = None) -> str:
        """
        Automatically fix font sizes to use only recommended sizes
        
        Args:
            filepath: Input LaTeX file
            output_path: Output file path (if None, adds '_fixed' suffix)
            
        Returns:
            Path to the fixed file
        """
        filepath = Path(filepath)
        if output_path is None:
            output_path = filepath.parent / f"{filepath.stem}_fixed{filepath.suffix}"
        
        # Mapping from non-recommended to recommended sizes
        size_map = {
            'tiny': 'small',
            'scriptsize': 'small',
            'footnotesize': 'small',
            'small': 'small',
            'normalsize': 'normalsize',
            'large': 'Large',
            'Large': 'Large',
            'LARGE': 'Large',
            'huge': 'Large',
            'Huge': 'Large'
        }
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace font sizes
        def replace_size(match):
            old_size = match.group(1)
            new_size = size_map.get(old_size.lower(), 'normalsize')
            return f'\\{new_size}'
        
        pattern = re.compile(r'\\(' + '|'.join(self.FONT_SIZES) + r')(?=\s|{|\\)', re.IGNORECASE)
        fixed_content = pattern.sub(replace_size, content)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        return str(output_path)


class TestFontSizeChecker(unittest.TestCase):
    """Unit tests for FontSizeChecker"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.checker = FontSizeChecker(max_sizes=3)
        
    def test_pattern_matching(self):
        """Test that font size patterns are correctly identified"""
        test_lines = [
            (r'\Large Title', ['Large']),
            (r'\normalsize body text', ['normalsize']),
            (r'\small\texttt{code}', ['small']),
            (r'\Huge\textbf{BIG} and \tiny small', ['Huge', 'tiny']),
            (r'No font sizes here', []),
            (r'% \Large commented out', []),
        ]
        
        for line, expected in test_lines:
            matches = self.checker.font_pattern.findall(line)
            self.assertEqual(matches, expected, f"Failed for line: {line}")
    
    def test_check_consistency_pass(self):
        """Test file with <= 3 font sizes (should pass)"""
        test_content = """
\\documentclass{beamer}
\\begin{document}
\\Large Title
\\normalsize Body text
\\small Details
\\end{document}
"""
        # Create temporary test file
        test_file = Path('test_pass.tex')
        test_file.write_text(test_content)
        
        try:
            passes, locations, unique = self.checker.check_file(str(test_file))
            self.assertTrue(passes)
            self.assertEqual(len(unique), 3)
            self.assertEqual(unique, {'large', 'normalsize', 'small'})
        finally:
            test_file.unlink()
    
    def test_check_consistency_fail(self):
        """Test file with > 3 font sizes (should fail)"""
        test_content = """
\\documentclass{beamer}
\\begin{document}
\\Huge Title
\\Large Subtitle
\\normalsize Body
\\small Details
\\tiny Footer
\\end{document}
"""
        # Create temporary test file
        test_file = Path('test_fail.tex')
        test_file.write_text(test_content)
        
        try:
            passes, locations, unique = self.checker.check_file(str(test_file))
            self.assertFalse(passes)
            self.assertEqual(len(unique), 5)
        finally:
            test_file.unlink()
    
    def test_fix_font_sizes(self):
        """Test automatic font size fixing"""
        test_content = """
\\Huge Title
\\huge Subtitle
\\LARGE Section
\\tiny Footer
"""
        test_file = Path('test_fix.tex')
        test_file.write_text(test_content)
        
        try:
            fixed_path = self.checker.fix_font_sizes(str(test_file))
            fixed_file = Path(fixed_path)
            
            # Check that file was created
            self.assertTrue(fixed_file.exists())
            
            # Check content was fixed
            fixed_content = fixed_file.read_text()
            self.assertIn('\\Large Title', fixed_content)
            self.assertIn('\\Large Subtitle', fixed_content)
            self.assertIn('\\Large Section', fixed_content)
            self.assertIn('\\small Footer', fixed_content)
            
            # Clean up
            fixed_file.unlink()
        finally:
            test_file.unlink()
    
    def test_line_number_tracking(self):
        """Test that line numbers are correctly tracked"""
        test_content = """Line 1
\\Large Line 2
Line 3
\\Large Line 4
\\small Line 5
"""
        test_file = Path('test_lines.tex')
        test_file.write_text(test_content)
        
        try:
            passes, locations, unique = self.checker.check_file(str(test_file))
            self.assertEqual(locations['large'], [2, 4])
            self.assertEqual(locations['small'], [5])
        finally:
            test_file.unlink()


def main():
    """Main function for command-line usage"""
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            # Run unit tests
            unittest.main(argv=[''], verbosity=2, exit=False)
        else:
            # Check file
            checker = FontSizeChecker(max_sizes=3)
            filepath = sys.argv[1]
            
            try:
                print(checker.generate_report(filepath))
                
                # Ask if user wants to fix
                passes, _, unique = checker.check_file(filepath)
                if not passes:
                    response = input("\nWould you like to automatically fix font sizes? (y/n): ")
                    if response.lower() == 'y':
                        fixed_path = checker.fix_font_sizes(filepath)
                        print(f"\nFixed file saved to: {fixed_path}")
                        print(checker.generate_report(fixed_path))
                        
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)
    else:
        print("Usage: python check_font_sizes.py <tex_file>")
        print("       python check_font_sizes.py test    (run unit tests)")
        sys.exit(1)


if __name__ == "__main__":
    main()