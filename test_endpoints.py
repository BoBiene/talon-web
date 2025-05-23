#!/usr/bin/env python3
"""
Test script für die neuen HTML-zu-Markdown-Endpunkte
"""

import requests
import json
import time

def test_endpoints():
    base_url = "http://localhost:5000"
    
    # Test HTML content
    test_html = """
    <h1>Wichtige Nachricht</h1>
    <p>Das ist ein <strong>wichtiger</strong> Inhalt der E-Mail mit einem <a href="https://example.com">Link</a>.</p>
    <ul>
    <li>Punkt 1</li>
    <li>Punkt 2</li>
    </ul>
    <p>Weitere Informationen.</p>
    <hr>
    <p>Mit freundlichen Grüßen<br>
    Max Mustermann<br>
    Beispiel GmbH<br>
    Tel: +49 123 456789</p>
    """
    
    print("=== Test der neuen HTML-zu-Markdown-Endpunkte ===\n")
    
    # Test 1: Talon-basierter Endpunkt
    print("1. Test: /talon/html-to-markdown (mit Talon's intelligenter Erkennung)")
    try:
        data = {
            "html": test_html.strip(),
            "sender": "max.mustermann@beispiel.de"
        }
        
        response = requests.post(f"{base_url}/talon/html-to-markdown", 
                               json=data, 
                               headers={'Content-Type': 'application/json'},
                               timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Erfolgreich!")
            print(f"Success: {result.get('success')}")
            print("\nGeneriertes Markdown:")
            print("=" * 50)
            print(result.get('markdown', 'Kein Markdown'))
            print("=" * 50)
            
            if result.get('removed_signature'):
                print(f"\nEntfernte Signatur: {result.get('removed_signature')}")
        else:
            print(f"❌ Fehler: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Fehler beim Testen: {e}")
    
    print("\n" + "="*80 + "\n")
    
    # Test 2: Direkter Endpunkt
    print("2. Test: /talon/html-to-markdown-direct (direkte Konvertierung)")
    try:
        data = {
            "html": test_html.strip()
        }
        
        response = requests.post(f"{base_url}/talon/html-to-markdown-direct", 
                               json=data, 
                               headers={'Content-Type': 'application/json'},
                               timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Erfolgreich!")
            print(f"Success: {result.get('success')}")
            print("\nGeneriertes Markdown:")
            print("=" * 50)
            print(result.get('markdown', 'Kein Markdown'))
            print("=" * 50)
        else:
            print(f"❌ Fehler: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Fehler beim Testen: {e}")
    
    print("\n" + "="*80 + "\n")
    
    # Test 3: Einfacher Health Check
    print("3. Test: Basis-Endpunkt (Health Check)")
    try:
        response = requests.get(f"{base_url}/talon/signature", timeout=5)
        print(f"Status Code: {response.status_code}")
        if response.status_code in [200, 400, 405]:  # 400/405 sind OK da wir keine Parameter senden
            print("✅ Server läuft!")
        else:
            print(f"❌ Unerwarteter Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Server nicht erreichbar: {e}")

if __name__ == "__main__":
    print("Warte 5 Sekunden bis der Server gestartet ist...")
    time.sleep(5)
    test_endpoints()
    print("\n=== Tests abgeschlossen ===")
