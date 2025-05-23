# HTML to Markdown API Endpoints

## Neue Endpunkte

### 1. `/talon/html-to-markdown` (POST)

Konvertiert HTML zu Markdown unter Verwendung von Talon's intelligenter Signatur- und Zitat-Erkennung.

**Input (JSON):**
```json
{
    "html": "<html><body><h1>Titel</h1><p>Inhalt...</p><hr><p>Mit freundlichen Grüßen...</p></body></html>",
    "sender": "max@beispiel.de"  // optional
}
```

**Output:**
```json
{
    "original_html": "...",
    "markdown": "# Titel\n\nInhalt...",
    "removed_signature": "Mit freundlichen Grüßen\nMax Mustermann",
    "sender": "max@beispiel.de",
    "success": true
}
```

### 2. `/talon/html-to-markdown-direct` (POST)

Direkte HTML-zu-Markdown-Konvertierung mit einfacher Signatur-Mustererkennung.

**Input (JSON):**
```json
{
    "html": "<html><body><h1>Titel</h1><p>Inhalt...</p><hr><p>Mit freundlichen Grüßen...</p></body></html>"
}
```

**Output:**
```json
{
    "original_html": "...",
    "markdown": "# Titel\n\nInhalt...",
    "success": true
}
```

## Funktionen

- **Intelligente Signatur-Erkennung**: Nutzt Talon's ML-basierte Signatur-Erkennung
- **Zitat-Entfernung**: Entfernt automatisch E-Mail-Zitate und Antworten
- **Markdown-Konvertierung**: Verwendet html2text für saubere Markdown-Ausgabe
- **Mehrsprachig**: Unterstützt deutsche und englische Signatur-Muster
- **Flexible Eingabe**: Akzeptiert sowohl JSON als auch Form-Data

## Verwendung

### Mit curl:

```bash
# Talon-basierte Konvertierung
curl -X POST http://localhost:5000/talon/html-to-markdown \
  -H "Content-Type: application/json" \
  -d '{"html": "<h1>Test</h1><p>Inhalt</p><hr><p>Mit freundlichen Grüßen</p>", "sender": "test@example.com"}'

# Direkte Konvertierung
curl -X POST http://localhost:5000/talon/html-to-markdown-direct \
  -H "Content-Type: application/json" \
  -d '{"html": "<h1>Test</h1><p>Inhalt</p><hr><p>Mit freundlichen Grüßen</p>"}'
```

### Mit Python:

```python
import requests

data = {
    "html": "<h1>Test</h1><p>Inhalt</p><hr><p>Mit freundlichen Grüßen</p>",
    "sender": "test@example.com"
}

response = requests.post('http://localhost:5000/talon/html-to-markdown', json=data)
result = response.json()
print(result['markdown'])
```

## Signatur-Muster

Der Service erkennt folgende Signatur-Muster:

### Deutsche Muster:
- "Mit freundlichen Grüßen"
- "Freundliche Grüße"  
- "Viele Grüße"

### Englische Muster:
- "Best regards"
- "Kind regards"
- "Sincerely"

### Technische Muster:
- `<hr>` Tags
- `--` Trennzeichen
- CSS-Klassen mit "signature"
- Gmail/Outlook Signatur-Blöcke

## App starten

```bash
cd talon-web
python -m talon.web.bootstrap
```

Der Service läuft dann auf `http://localhost:5000`
