#!/usr/bin/env python3
"""
Test the HTML to Markdown conversion functionality
"""

import html2text
import re

def remove_html_signature_patterns(html_content):
    """Remove common email signature patterns from HTML"""
    signature_patterns = [
        r'<hr[^>]*>.*$',
        r'<div[^>]*>Mit freundlichen Grüßen.*$',
        r'<p[^>]*>Mit freundlichen Grüßen.*$',
        r'<div[^>]*>Best regards.*$',
        r'<p[^>]*>Best regards.*$',
        r'--\s*<br[^>]*>.*$',
    ]
    
    cleaned_html = html_content
    for pattern in signature_patterns:
        cleaned_html = re.sub(pattern, '', cleaned_html, flags=re.DOTALL | re.IGNORECASE)
    
    return cleaned_html

def html_to_markdown(html_content):
    """Convert HTML to Markdown with signature removal"""
    # Remove signatures
    cleaned_html = remove_html_signature_patterns(html_content)
    
    # Configure html2text
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_emphasis = False
    h.body_width = 0
    h.unicode_snob = True
    h.use_automatic_links = True
    h.protect_links = True
    
    # Convert to markdown
    markdown_content = h.handle(cleaned_html)
    
    # Clean up whitespace
    markdown_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', markdown_content)
    markdown_content = markdown_content.strip()
    
    return markdown_content, cleaned_html

# Test HTML with German signature
test_html = """
<html>
<body>
<h1>Wichtige Nachricht</h1>
<p>Hallo,</p>
<p>das ist der <strong>wichtige</strong> Inhalt der E-Mail mit einem <a href="https://example.com">Link</a>.</p>
<ul>
<li>Punkt 1</li>
<li>Punkt 2</li>
</ul>
<p>Hier ist noch ein Absatz mit wichtigen Informationen.</p>
<hr>
<p>Mit freundlichen Grüßen<br>
Max Mustermann<br>
Beispiel GmbH<br>
Tel: +49 123 456789<br>
Email: max@beispiel.de</p>
</body>
</html>
"""

if __name__ == "__main__":
    print("=== HTML to Markdown Conversion Test ===")
    print("\nOriginal HTML:")
    print("=" * 50)
    print(test_html)
    
    markdown, cleaned_html = html_to_markdown(test_html)
    
    print("\nCleaned HTML (after signature removal):")
    print("=" * 50)
    print(cleaned_html)
    
    print("\nResulting Markdown:")
    print("=" * 50)
    print(markdown)
    
    print("\n=== Test completed successfully! ===")
