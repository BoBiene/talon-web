#!/usr/bin/env python3
"""
Einfacher Test für die HTML zu Markdown Funktionalität
"""

def test_html_markdown():
    try:
        import html2text
        import re
        
        # Test HTML
        test_html = """
        <html>
        <body>
        <h1>Test Nachricht</h1>
        <p>Das ist ein <strong>wichtiger</strong> Test mit einem <a href="https://example.com">Link</a>.</p>
        <ul>
        <li>Punkt 1</li>
        <li>Punkt 2</li>
        </ul>
        <hr>
        <p>Mit freundlichen Grüßen<br>Max Mustermann</p>
        </body>
        </html>
        """
        
        # Signatur entfernen (alles nach <hr>)
        cleaned_html = re.sub(r'<hr[^>]*>.*$', '', test_html, flags=re.DOTALL | re.IGNORECASE)
        
        # HTML zu Markdown konvertieren
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.body_width = 0
        h.unicode_snob = True
        
        markdown = h.handle(cleaned_html).strip()
        
        print("=== HTML zu Markdown Test ===")
        print("\nOriginal HTML:")
        print(test_html)
        print("\nHTML nach Signatur-Entfernung:")
        print(cleaned_html)
        print("\nResultierendes Markdown:")
        print(markdown)
        print("\n✅ Test erfolgreich!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test fehlgeschlagen: {e}")
        return False

if __name__ == "__main__":
    test_html_markdown()
