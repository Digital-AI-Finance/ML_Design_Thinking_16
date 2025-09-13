import requests
import re
from urllib.parse import urlparse
import time
from typing import List, Tuple
import sys

def extract_urls_from_tex(filename: str) -> List[Tuple[str, int]]:
    """Extract URLs from LaTeX file with line numbers"""
    urls = []
    url_pattern = r'https?://[^\s\}\\]+'
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                found_urls = re.findall(url_pattern, line)
                for url in found_urls:
                    # Clean up URL (remove trailing punctuation)
                    url = url.rstrip('.,;:)')
                    urls.append((url, i))
    except FileNotFoundError:
        print(f"File {filename} not found")
        return []
    
    return urls

def check_url(url: str, timeout: int = 10) -> Tuple[bool, str]:
    """Check if URL is accessible"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        if response.status_code < 400:
            return True, f"OK ({response.status_code})"
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection Error"
    except requests.exceptions.RequestException as e:
        return False, f"Error: {str(e)[:50]}"

def verify_links_in_file(filename: str):
    """Verify all links in a LaTeX file"""
    print(f"\nVerifying links in: {filename}")
    print("-" * 60)
    
    urls = extract_urls_from_tex(filename)
    
    if not urls:
        print("No URLs found in file")
        return
    
    print(f"Found {len(urls)} URLs to verify\n")
    
    results = []
    for url, line_num in urls:
        print(f"Checking line {line_num}: {url[:60]}{'...' if len(url) > 60 else ''}")
        is_valid, status = check_url(url)
        results.append((url, line_num, is_valid, status))
        
        if is_valid:
            print(f"  [OK] {status}")
        else:
            print(f"  [FAIL] {status}")
        
        # Be polite to servers
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    valid_count = sum(1 for _, _, is_valid, _ in results if is_valid)
    invalid_count = len(results) - valid_count
    
    print(f"Total URLs checked: {len(results)}")
    print(f"Valid URLs: {valid_count}")
    print(f"Invalid URLs: {invalid_count}")
    
    if invalid_count > 0:
        print("\nFailed URLs:")
        for url, line_num, is_valid, status in results:
            if not is_valid:
                print(f"  Line {line_num}: {url}")
                print(f"    Status: {status}")
    
    return results

def main():
    """Main function to verify links in LaTeX files"""
    
    # List of common academic/reference URLs to verify
    reference_urls = [
        ("BERT Paper", "https://arxiv.org/abs/1810.04805"),
        ("Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
        ("LDA Paper", "https://www.jmlr.org/papers/v3/blei03a.html"),
        ("Design Thinking HBR", "https://hbr.org/2008/06/design-thinking"),
        ("Coursera ML Course", "https://www.coursera.org/learn/machine-learning"),
        ("Google What-If Tool", "https://pair-code.github.io/what-if-tool/"),
        ("Illustrated Transformer", "http://jalammar.github.io/illustrated-transformer/"),
        ("Kaggle", "https://www.kaggle.com/"),
        ("OpenAI GPT", "https://openai.com/research/gpt-4"),
        ("Hugging Face", "https://huggingface.co/"),
    ]
    
    print("=" * 60)
    print("LINK VERIFICATION TOOL")
    print("=" * 60)
    
    # Check if specific file is provided
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        verify_links_in_file(filename)
    else:
        # Check reference URLs
        print("\nVerifying common reference URLs...")
        print("-" * 60)
        
        for name, url in reference_urls:
            print(f"\nChecking: {name}")
            print(f"  URL: {url}")
            is_valid, status = check_url(url)
            if is_valid:
                print(f"  ✓ {status}")
            else:
                print(f"  ✗ {status}")
            time.sleep(0.5)
        
        # Check LaTeX files if they exist
        tex_files = ['week1_ai_empathy_v3.tex', 'week1_ai_empathy_v2.tex']
        
        for tex_file in tex_files:
            try:
                with open(tex_file, 'r'):
                    verify_links_in_file(tex_file)
            except FileNotFoundError:
                continue

if __name__ == "__main__":
    main()