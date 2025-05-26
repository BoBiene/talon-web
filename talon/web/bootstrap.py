from talon import signature, quotations
from flask import Flask, request, jsonify, json
from werkzeug.exceptions import HTTPException, BadRequest
from bs4 import BeautifulSoup
import html2text
import re
import talon
import logging

talon.init()

log = logging.getLogger(__name__)
app = Flask(__name__)


@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response


@app.errorhandler(BadRequest)
def handle_bad_request(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response


@app.route('/talon/signature', methods=['GET', 'POST'])
def get_signature():
    email_content = request.form.get('email_content')
    email_sender = request.form.get('email_sender')
    if email_content and email_sender:
        log.debug('email content: ' + email_content)
        text, s = signature.extract(email_content, sender=email_sender)
        log.debug('text: ' + text)
        log.debug('signature: ' + str(s))
        json_response = {'email_content': email_content, 'email_body': text,
                         'email_sender': email_sender, 'email_signature': str(s)}
    else:
        raise BadRequest(
            "Required parameter 'email_content' or 'email_sender' is missing.")
    return jsonify(json_response)


@app.route('/talon/quotations/text', methods=['GET', 'POST'])
def get_reply_plain():
    email_content = request.form.get('email_content')
    email_sender = request.form.get('email_sender')
    if email_content:
        log.debug('email content: ' + email_content)
        text = quotations.extract_from_plain(email_content)
        if email_sender:
            text, s = signature.extract(text, sender=email_sender)
            log.debug('text: ' + text)
            log.debug('signature: ' + str(s))
            json_response = {'email_content': email_content, 'email_reply': text,
                             'email_sender': email_sender, 'email_signature': str(s)}
        else:
            log.debug('text: ' + text)
            json_response = {
                'email_content': email_content, 'email_reply': text}
    else:
        raise BadRequest("Required parameter 'email_content' is missing.")
    return jsonify(json_response)

@app.route('/talon/html-to-markdown', methods=['POST'])
def html_to_markdown():
    # Support both JSON and form data
    if request.is_json:
        data = request.get_json()
        html_content = data.get('html')
        sender = data.get('email_sender')
    else:
        html_content = request.form.get('html')
        sender = request.form.get('email_sender')

    if not html_content:
        raise BadRequest("Required parameter 'html' is missing.")

    try:
        # 1. Vorverarbeitung: Zitat- und Signatur-Blöcke hart entfernen
        soup = BeautifulSoup(html_content, "html.parser")

        # Entferne störende Tags
        for tag in soup(["style", "script", "footer", "nav", "header"]):
            tag.decompose()
        
        # Entferne Zitat-Blöcke und andere störende Elemente
        for sel in ["blockquote", ".gmail_quote", ".gmail_extra", ".moz-cite-prefix", "table", "hr"]:
            for el in soup.select(sel):
                el.decompose()

        # Erst zu HTML-Text konvertieren
        clean_html = str(soup)

        # 2. HTML zu Markdown konvertieren
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_emphasis = False
        h.body_width = 0
        h.unicode_snob = True
        markdown = h.handle(clean_html)

        # --- POSTPROCESSING: Whitespace-Optimierung ---
        # 1. Mehr als 2 aufeinanderfolgende Leerzeilen auf maximal 2 reduzieren
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        # 2. Zeilen mit nur Whitespace entfernen
        markdown = '\n'.join([l.rstrip() for l in markdown.splitlines()])
        # 3. Am Anfang/Ende trimmen
        markdown = markdown.strip()

        # 3. Plain text für Talon extrahieren
        plain_text = soup.get_text(separator="\n").strip()
        
        # 4. Zitat entfernen via Talon
        text = quotations.extract_from_plain(plain_text)
        sig = None
        if sender:
            text, sig = signature.extract(text, sender=sender)
        # 5. Grußformel + Name aus Markdown beibehalten, danach abschneiden
        lines = markdown.strip().splitlines()
        result = []
        # Verbesserte Regex-Patterns für deutsche und englische Grußformeln, inkl. Multi-Sprach-Kombinationen
        salute_re = re.compile(r'(?i)^(best\s+regards?|kind\s+regards?|warm\s+regards?|thanks?|cheers?|sincerely|yours?\s+truly|mit\s+freundlichen\s+grüßen|viele\s+grüße|beste\s+grüße|schöne\s+grüße|herzliche\s+grüße|liebe\s+grüße)(\s*/\s*(best\s+regards?|kind\s+regards?|warm\s+regards?|thanks?|cheers?|sincerely|yours?\s+truly|mit\s+freundlichen\s+grüßen|viele\s+grüße|beste\s+grüße|schöne\s+grüße|herzliche\s+grüße|liebe\s+grüße))?[,:\s]*$')
        name_re = re.compile(r'^[A-ZÄÖÜa-zäöüß][A-ZÄÖÜa-zäöüß\s\-\.]{1,30}$')

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                # Füge leere Zeilen nur hinzu wenn wir noch nicht bei der Grußformel sind
                if not any(salute_re.match(l.strip()) for l in result if l.strip()):
                    result.append(line)
                continue
                
            result.append(line)
            
            # Prüfe auf Grußformel
            if salute_re.match(line_stripped):
                # Schaue nach dem Namen in den nächsten Zeilen (bis zu 2 Zeilen weiter)
                for j in range(i + 1, min(i + 3, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and name_re.match(next_line) and len(next_line.split()) <= 3:
                        # Füge leere Zeile zwischen Grußformel und Name hinzu wenn nötig
                        if j > i + 1:
                            result.append("")
                        result.append(lines[j])
                        break
                break

        final_markdown = "\n".join(result).strip()
        # Entferne Markdown-hardbreaks ("  \n") und andere Spaces am Zeilenende
        final_markdown = re.sub(r'[ \t]+\n', '\n', final_markdown)
        # Nochmals: Mehr als 2 aufeinanderfolgende Leerzeilen auf maximal 2 reduzieren
        final_markdown = re.sub(r'\n{3,}', '\n\n', final_markdown)
        final_markdown = final_markdown.strip()

        return jsonify({
            'original_html': html_content,
            'markdown': final_markdown,
            'plain_text': text,
            'email_sender': sender,
            'removed_signature': str(sig) if sig else None
        })

    except Exception as e:
        log.error(f"Error processing HTML to Markdown: {str(e)}")
        raise BadRequest(f"Error processing HTML: {str(e)}")

@app.route('/talon/quotations/html', methods=['GET', 'POST'])
def get_reply_html():
    email_content = request.form.get('email_content')
    email_sender = request.form.get('email_sender')
    if email_content:
        log.debug('email content: ' + email_content)
        text = quotations.extract_from_html(email_content)

        if email_sender:
            text, s = signature.extract(text, sender=email_sender)
            log.debug('text: ' + text)
            log.debug('signature: ' + str(s))
            json_response = {'email_content': email_content, 'email_reply': text,
                             'email_sender': email_sender, 'email_signature': str(s)}
        else:
            log.debug('text: ' + text)
            json_response = {
                'email_content': email_content, 'email_reply': text}
    else:
        raise BadRequest("Required parameter 'email_content' is missing.")
    return jsonify(json_response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5505)