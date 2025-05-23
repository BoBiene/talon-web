#!/usr/bin/env python3
"""
Test script for the new HTML to Markdown endpoints
"""
import requests
import json

# Test HTML content with signature
test_html = """
<html>
<body>
<h1>Wichtige Nachricht</h1>
<p>Hallo,</p>
<p>das ist der <strong>wichtige</strong> Inhalt der E-Mail mit einem <a href="https://example.com">Link</a>.</p>
<p>Hier ist noch ein Absatz mit wichtigen Informationen.</p>
<hr>
<p>Mit freundlichen Grüßen<br>
Max Mustermann<br>
Beispiel GmbH<br>
Tel: +49 123 456789</p>
</body>
</html>
"""

# Test data
test_data = {
    "html": test_html,
    "sender": "max.mustermann@beispiel.de"
}

def test_endpoint(url, data):
    """Test an endpoint with given data"""
    print(f"\n=== Testing {url} ===")
    try:
        response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Success: {result.get('success', 'N/A')}")
        
        if 'markdown' in result:
            print("\nGenerated Markdown:")
            print("=" * 50)
            print(result['markdown'])
            print("=" * 50)
        
        if 'removed_signature' in result and result['removed_signature']:
            print(f"\nRemoved Signature: {result['removed_signature']}")
            
        if 'error' in result:
            print(f"Error: {result['error']}")
            
    except Exception as e:
        print(f"Error testing endpoint: {e}")

if __name__ == "__main__":
    base_url = "http://localhost:5000"
    
    print("Testing HTML to Markdown endpoints")
    print("Make sure the Flask app is running first!")
    
    # Test the Talon-based endpoint
    test_endpoint(f"{base_url}/talon/html-to-markdown", test_data)
    
    # Test the direct conversion endpoint
    test_endpoint(f"{base_url}/talon/html-to-markdown-direct", {"html": test_html})
    
    print("\nDone!")
