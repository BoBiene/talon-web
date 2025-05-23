# Test für /talon/html-to-markdown mit anonymisiertem HTML-Beispiel
import json
from talon.web.bootstrap import app

def test_html_to_markdown_signature_removal():
    # Anonymisiertes HTML mit typischer Signaturstruktur
    html = '''
    <div>Hello Team</div>
    <div><br></div>
    <div>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</div>
    <div><br></div>
    <div>Source Server:</div>
    <div>SERVER-1234</div>
    <div>KEY-XXXX-YYYY-ZZZZ</div>
    <div><br></div>
    <div>Destination Server:</div>
    <div>SERVER-5678 (10.0.0.1)</div>
    <div><br></div>
    <div>Can you please generate a demo license for migration?</div>
    <div><br></div>
    <div><br></div>
    <div>Best regards</div>
    <table><tbody><tr><td><p>Lorem Ipsum</p><p>Operations Engineer | Customer Solutions</p><p>lorem.ipsum@example.com | <a href=\"http://www.example.com/\">www.example.com</a></p></td></tr></tbody></table>
    <table><tbody><tr><td><p>Discover our latest <a href=\"http://www.example.com/blog/\">Blog</a></p></td></tr></tbody></table>
    <p>Our new address as of April 2024: Example GmbH, Example Street 1, 12345 Example City, Germany</p>
    <p>The information transmitted is intended only for the addressee and may contain confidential and/or privileged material. If you are not the intended recipient, please contact the sender and delete the material.</p>
    '''
    sender = "lorem.ipsum@example.com"
    with app.test_client() as client:
        response = client.post(
            "/talon/html-to-markdown",
            data=json.dumps({"html": html, "sender": sender}),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = response.get_json()
        # Die Signatur sollte entfernt sein
        assert "Best regards" not in data["markdown"]
        assert "lorem.ipsum@example.com" not in data["markdown"]
        # Die removed_signature sollte die Signatur enthalten
        assert data["removed_signature"] is not None
        # Der Haupttext sollte erhalten bleiben
        assert "Hello Team" in data["markdown"]
        assert "Source Server" in data["markdown"]
