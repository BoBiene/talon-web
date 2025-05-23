#!/usr/bin/env python3
"""
Demo: HTML to Markdown Conversion with Signature Removal
"""

import html2text
import re

def demo_conversion():
    # Test HTML with signature
    html_with_signature = """
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
    Beispiel GmbH</p>
    </body>
    </html>
    """
    
    # Remove signature (everything after <hr>)
    cleaned_html = re.sub(r'<hr[^>]*>.*$', '', html_with_signature, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert to markdown
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    
    markdown = h.handle(cleaned_html).strip()
    
    print("=== HTML zu Markdown Konvertierung ===")
    print("\n1. Original HTML:")
    print(html_with_signature)
    
    print("\n2. HTML nach Signatur-Entfernung:")
    print(cleaned_html)
    
    print("\n3. Finales Markdown:")
    print(markdown)
    
    return markdown

if __name__ == "__main__":
    result = demo_conversion()
    print(f"\n=== Ergebnis ===")
    print(f"Markdown Zeilen: {len(result.splitlines())}")
    print("Konvertierung erfolgreich!")
