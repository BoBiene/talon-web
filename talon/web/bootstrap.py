from flask import Flask, request, jsonify
import html2text
from talon import signature, quotations
from werkzeug.exceptions import HTTPException, BadRequest
import talon
import logging
import re
import json

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


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring and load balancers."""
    try:
        # Basic service check
        test_text = "Test message"
        test_sender = "test@example.com"
        
        # Test signature extraction
        signature.extract(test_text, sender=test_sender)
        
        # Test quotations extraction  
        quotations.extract_from_plain(test_text)
        
        return jsonify({
            "status": "healthy",
            "service": "talon-web-api",
            "version": "1.6.0",
            "endpoints": [
                "/talon/signature",
                "/talon/quotations/text", 
                "/talon/quotations/html",
                "/talon/html-to-markdown",
                "/talon/html-to-markdown-direct"
            ]
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


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
    else:        raise BadRequest("Required parameter 'email_content' is missing.")
    return jsonify(json_response)


@app.route('/talon/html-to-markdown', methods=['POST'])
def html_to_markdown():
    """
    Converts HTML to Markdown, removes email signatures and quotations
    Expects JSON input: {"html": "HTML content", "sender": "optional sender email"}
    """
    try:
        # Get JSON data from request
        if request.is_json:
            data = request.get_json()
            html_content = data.get('html')
            sender = data.get('sender')
        else:
            # Fallback to form data for compatibility
            html_content = request.form.get('html_content')
            sender = request.form.get('email_sender')
        
        if not html_content:
            raise BadRequest("Required parameter 'html' is missing.")
        
        log.debug('HTML content received: ' + str(len(html_content)) + ' characters')
        
        # 1. Entferne HTML-Signaturmuster
        cleaned_html = _remove_html_signature_patterns(html_content)
        removed_html_signature = None
        if cleaned_html != html_content:
            removed_html_signature = 'HTML signature patterns removed.'
        
        # 2. Entferne Zitate mit Talon
        clean_text = quotations.extract_from_html(cleaned_html)
        log.debug('Text after quotation removal: ' + clean_text)
        
        # 3. Konvertiere zu Markdown
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_emphasis = False
        h.body_width = 0  # Don't wrap lines
        h.unicode_snob = True
        h.use_automatic_links = True
        h.protect_links = True
        
        if clean_text:
            simple_html = clean_text.replace('\n', '<br>\n')
            markdown_content = h.handle(simple_html)
        else:
            markdown_content = ""
        
        markdown_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', markdown_content)
        markdown_content = markdown_content.strip()
        
        # 4. Finale Talon-Signatur-Extraktion auf Markdown-Text
        removed_signature = None
        final_markdown = markdown_content
        if sender and markdown_content:
            final_markdown, removed_signature = signature.extract(markdown_content, sender=sender)
        # Wenn durch Pattern-Matching eine Signatur entfernt wurde, aber Talon keine findet, setze removed_signature entsprechend
        if not removed_signature and removed_html_signature:
            removed_signature = 'HTML signature removed'
        response_data = {
            'original_html': html_content,
            'markdown': final_markdown,
            'removed_html_signature': removed_html_signature,
            'removed_signature': str(removed_signature) if removed_signature else None,
            'sender': sender,
            'success': True
        }
        
        return jsonify(response_data)
        
    except BadRequest:
        raise
    except Exception as e:
        log.error(f"Error in html_to_markdown: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/talon/html-to-markdown-direct', methods=['POST'])
def html_to_markdown_direct():
    """
    Converts HTML directly to Markdown with basic signature removal patterns
    More direct approach without Talon's quotation extraction
    Expects JSON input: {"html": "HTML content"}
    """
    try:
        # Get JSON data from request
        if request.is_json:
            data = request.get_json()
            html_content = data.get('html')
        else:
            html_content = request.form.get('html_content')
        
        if not html_content:
            raise BadRequest("Required parameter 'html' is missing.")
        
        log.debug('HTML content received for direct conversion: ' + str(len(html_content)) + ' characters')
        
        # Remove common signature patterns directly from HTML
        cleaned_html = _remove_html_signature_patterns(html_content)
        
        # Configure html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_emphasis = False
        h.body_width = 0  # Don't wrap lines
        h.unicode_snob = True
        h.use_automatic_links = True
        h.protect_links = True
        
        # Convert to markdown
        markdown_content = h.handle(cleaned_html)
        
        # Clean up extra whitespace
        markdown_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', markdown_content)
        markdown_content = markdown_content.strip()
        
        response_data = {
            'original_html': html_content,
            'markdown': markdown_content,
            'success': True
        }
        
        return jsonify(response_data)
        
    except BadRequest:
        raise
    except Exception as e:
        log.error(f"Error in html_to_markdown_direct: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500


# Bekannte Grußformeln (deutsch/englisch, inkl. Abkürzungen)
SALUTATIONS = set([
    # Deutsch
    "Mit freundlichen Grüßen", "Freundliche Grüße", "Viele Grüße", "Beste Grüße", "Herzliche Grüße", "Liebe Grüße", "Schöne Grüße", "MfG", "VG", "Vg", "LG", "Lg", "Mit besten Grüßen", "Gruß", "Hochachtungsvoll",
    # Englisch
    "Best regards", "Kind regards", "Regards", "Sincerely", "Yours sincerely", "Yours truly", "Yours faithfully", "Thank you", "Thanks", "Cheers", "Warm regards", "With best wishes", "Respectfully", "Best wishes", "Warmest regards", "Best", "All the best", "Best regards,", "Best wishes,", "Best,"
])

# Separate short and long salutations for more restrictive matching
SHORT_SALUTATIONS = {s for s in SALUTATIONS if len(s.replace(',', '').replace('.', '').strip()) <= 4 or s in {"Best", "Best,", "Thanks", "Gruß", "LG", "Lg", "VG", "Vg", "MfG"}}
LONG_SALUTATIONS = SALUTATIONS - SHORT_SALUTATIONS


def _remove_html_signature_patterns(html_content):
    """
    Remove common email signature patterns from HTML (dynamisch aus Grußformel-Liste)
    """
    # Patterns für div, p, span, table mit Grußformeln
    salutes_regex = "|".join([re.escape(s) for s in LONG_SALUTATIONS])
    short_salutes_regex = "|".join([re.escape(s) for s in SHORT_SALUTATIONS])
    signature_patterns = [
        # Signature divs with class
        r'<div[^>]*class=["\']?[^>]*signature[^>]*["\']?[^>]*>.*?</div>',
        r'<div[^>]*class=["\']?[^>]*sig[^>]*["\']?[^>]*>.*?</div>',
        # Gmail/Outlook signature blocks
        r'<div[^>]*gmail_signature[^>]*>.*?</div>',
        r'<div[^>]*id=["\']?[^>]*signature[^>]*["\']?[^>]*>.*?</div>',
        # Tabellen mit typischen Signaturinhalten (Name, Titel, E-Mail, Links, Grußformeln)
        r'<table[^>]*>.*?(?:[A-Z][a-z]+ [A-Z][a-z]+|[\w\.-]+@[\w\.-]+|Operations Engineer|Customer Solutions|regards|Grüße|Mit freundlichen|Kind regards|Best regards|Sincerely|Thanks|VG|MfG|LG|Vg|Lg|Warm regards|Best wishes|www\\.|linkedin|instagram|youtube|XING|address as of|USt-IdNr|Managing Director|privacy agreement|Datenschutzvereinbarung).*?</table>',
        # <p>, <span>, <div> mit langen Grußformeln: NUR wenn sie am Block-Anfang stehen (restriktiv, optional Komma und <br>)
        rf'<p[^>]*>\s*({salutes_regex})\s*,?\s*(<br[^>]*>\s*)*</p>',
        rf'<span[^>]*>\s*({salutes_regex})\s*,?\s*(<br[^>]*>\s*)*</span>',
        rf'<div[^>]*>\s*({salutes_regex})\s*,?\s*(<br[^>]*>\s*)*</div>',
        # <p>, <span>, <div> mit kurzen Grußformeln: nur wenn sie allein im Block stehen (restriktiv, optional Komma und <br>)
        rf'<p[^>]*>\s*({short_salutes_regex})\s*,?\s*(<br[^>]*>\s*)*</p>',
        rf'<span[^>]*>\s*({short_salutes_regex})\s*,?\s*(<br[^>]*>\s*)*</span>',
        rf'<div[^>]*>\s*({short_salutes_regex})\s*,?\s*(<br[^>]*>\s*)*</div>',
        # Common German/English signatures (nur einzelne Blöcke, nicht alles danach)
        *[rf'<div[^>]*>{re.escape(s)}.*?</div>' for s in SALUTATIONS],
        *[rf'<p[^>]*>{re.escape(s)}.*?</p>' for s in SALUTATIONS],
        # Double dash separator (nur einzelne Blöcke)
        r'<p[^>]*>--\s*</p>',
        r'--\s*<br[^>]*>',
        # Horizontal line signatures (nur <hr>, nicht alles danach)
        r'<hr[^>]*>',
    ]
    cleaned_html = html_content
    for pattern in signature_patterns:
        cleaned_html = re.sub(pattern, '', cleaned_html, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_html


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
