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
    else:
        raise BadRequest("Required parameter 'email_content' is missing.")
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
          # 4. Finale Signatur-Extraktion auf Markdown-Text mit verbesserter Erkennung
        removed_signature = None
        final_markdown = markdown_content
        
        if sender and markdown_content:
            # Säubere Markdown für bessere Talon-Erkennung 
            clean_text_for_talon = _clean_markdown_for_signature_extraction(markdown_content)
            
            # Versuche Talon-Extraktion auf dem gesäuberten Text
            cleaned_text, talon_signature = signature.extract(clean_text_for_talon, sender=sender)
            
            if talon_signature:
                # Wenn Talon eine Signatur gefunden hat, wende das auf das originale Markdown an
                final_markdown = _apply_signature_removal_to_markdown(markdown_content, clean_text_for_talon, cleaned_text)
                removed_signature = talon_signature
            else:
                # Fallback: eigene Markdown-Signatur-Patterns
                final_markdown, fallback_signature = _remove_markdown_signature_patterns(markdown_content, sender)
                if fallback_signature:
                    removed_signature = fallback_signature
        
        # Wenn durch Pattern-Matching eine Signatur entfernt wurde, aber keine finale Signatur gefunden wurde
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


# Enhanced salutation detection with better categorization
COMMON_SALUTATIONS = {
    # Common German salutations
    "Mit freundlichen Grüßen", "Freundliche Grüße", "Viele Grüße", "Beste Grüße", 
    "Herzliche Grüße", "Liebe Grüße", "Schöne Grüße", "Mit besten Grüßen", 
    "Hochachtungsvoll", "Alles Gute", "Bis bald",
    
    # Common English salutations  
    "Best regards", "Kind regards", "Warm regards", "Warmest regards", "Regards",
    "Sincerely", "Yours sincerely", "Yours truly", "Yours faithfully",
    "Best wishes", "With best wishes", "Thank you", "Thanks", "Cheers",
    "All the best", "Take care", "Looking forward", "Respectfully",
    
    # Multi-language combinations (German/English)
    "Schöne Grüße / Best regards", "Viele Grüße / Best regards", "Beste Grüße / Kind regards",
    "Mit freundlichen Grüßen / Best regards", "Freundliche Grüße / Best regards",
    "Best regards / Schöne Grüße", "Kind regards / Viele Grüße",
    
    # With punctuation variants
    "Best regards,", "Kind regards,", "Best wishes,", "Sincerely,", "Thanks,", "Cheers,",
    "Mit freundlichen Grüßen,", "Viele Grüße,", "Beste Grüße,",
    "Schöne Grüße / Best regards,", "Viele Grüße / Best regards,",
}

# Abbreviations and short forms that need very careful matching
SHORT_RISKY_SALUTATIONS = {
    "MfG", "VG", "Vg", "LG", "Lg", "Best", "Best,", "Thanks", "Gruß", "Gruss"
}

# Safe short salutations that are unlikely to appear in normal text
SAFE_SHORT_SALUTATIONS = {
    "Hochachtungsvoll", "Yours faithfully", "Yours sincerely", "Yours truly"
}

# Combine all salutations
ALL_SALUTATIONS = COMMON_SALUTATIONS | SHORT_RISKY_SALUTATIONS | SAFE_SHORT_SALUTATIONS

# Categorize for different matching strategies
LONG_SALUTATIONS = {s for s in ALL_SALUTATIONS if len(s.replace(',', '').replace('.', '').strip()) > 8}
MEDIUM_SALUTATIONS = {s for s in ALL_SALUTATIONS if 4 < len(s.replace(',', '').replace('.', '').strip()) <= 8}
SHORT_SALUTATIONS = SHORT_RISKY_SALUTATIONS


def strip_inline_tags(html):
    """Entfernt alle span, font, b, i, u, em, strong Tags (nur Tags, nicht Inhalt)."""
    # Entferne nur die Tags, nicht den Inhalt
    return re.sub(r'</?(span|font|b|i|u|em|strong)[^>]*>', '', html, flags=re.IGNORECASE)


def _remove_html_signature_patterns(html_content):
    """
    Remove common email signature patterns from HTML with improved precision
    - Enhanced salutation categorization for better matching
    - Much more restrictive table pattern matching
    - Better debugging and logging
    """
    original_content = html_content
    html_content = strip_inline_tags(html_content)
      # Build regex patterns for different salutation categories
    long_salutes_regex = "|".join([re.escape(s) for s in LONG_SALUTATIONS])
    medium_salutes_regex = "|".join([re.escape(s) for s in MEDIUM_SALUTATIONS]) 
    short_salutes_regex = "|".join([re.escape(s) for s in SHORT_SALUTATIONS])
    
    signature_patterns = [
        # 1. Explicit signature containers (highest confidence)
        r'<div[^>]*class=["\']?[^>]*signature[^>]*["\']?[^>]*>.*?</div>',
        r'<div[^>]*class=["\']?[^>]*sig[^>]*["\']?[^>]*>.*?</div>',
        r'<div[^>]*gmail_signature[^>]*>.*?</div>',
        r'<div[^>]*id=["\']?[^>]*signature[^>]*["\']?[^>]*>.*?</div>',

        # 2. CONSERVATIVE salutation patterns - only remove if followed by clear signature content
        rf'<div[^>]*>\s*({long_salutes_regex})\s*,?\s*</div>\s*<div[^>]*>[^<]*(?:[\w\.-]+@[\w\.-]+|Engineer|Director|Manager|CEO)[^<]*</div>',
        rf'<p[^>]*>\s*({long_salutes_regex})\s*,?\s*</p>\s*<(?:div|p)[^>]*>[^<]*(?:[\w\.-]+@[\w\.-]+|Engineer|Director|Manager|CEO)[^<]*</(?:div|p)>',
        
        rf'<div[^>]*>\s*({medium_salutes_regex})\s*,?\s*</div>\s*<div[^>]*>[^<]*(?:[\w\.-]+@[\w\.-]+|Engineer|Director|Manager|CEO)[^<]*</div>',
        rf'<p[^>]*>\s*({medium_salutes_regex})\s*,?\s*</p>\s*<(?:div|p)[^>]*>[^<]*(?:[\w\.-]+@[\w\.-]+|Engineer|Director|Manager|CEO)[^<]*</(?:div|p)>',

        # 3. Signature tables with salutation - very conservative
        rf'<div[^>]*>\s*({long_salutes_regex}|{medium_salutes_regex})\s*,?\s*</div>\s*<table[^>]*>.*?[\w\.-]+@[\w\.-]+.*?</table>',
        
        # 4. Stand-alone signature elements at end only
        rf'<table[^>]*>.*?(?:[\w\.-]+@[\w\.-]+).*?(?:Engineer|Director|Manager|CEO).*?</table>\s*$',
        
        # 5. Explicit signature separators  
        r'<hr[^>]*/?>\s*',        r'<p[^>]*>\s*--\s*</p>',
        r'--\s*<br[^>]*>',
    ]
    
    cleaned_html = html_content
    removed_parts = []
    pattern_hits = {}
    
    for i, pattern in enumerate(signature_patterns):
        matches = list(re.finditer(pattern, cleaned_html, flags=re.DOTALL | re.IGNORECASE))
        if matches:
            pattern_name = [
                "Explicit signature containers", "Explicit signature containers", "Explicit signature containers", "Explicit signature containers",
                "Conservative long salutations", "Conservative long salutations",
                "Conservative medium salutations", "Conservative medium salutations", 
                "Signature tables with salutation",
                "Signature tables at end",
                "Signature separators", "Signature separators", "Signature separators"
            ][i]
            pattern_hits[pattern_name] = len(matches)
            
            for match in matches:
                matched_text = match.group(0)
                if len(matched_text) > 100:
                    removed_parts.append(matched_text[:100] + "...")
                else:
                    removed_parts.append(matched_text)
        
        cleaned_html = re.sub(pattern, '', cleaned_html, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up empty blocks
    cleaned_html = re.sub(r'<(div|p|span)[^>]*>\s*</\\1>', '', cleaned_html, flags=re.IGNORECASE)
    
    # Enhanced logging for debugging
    if cleaned_html != original_content:
        total_removed = len(original_content) - len(cleaned_html)
        log.debug(f"Signature removal: {total_removed} characters removed ({len(removed_parts)} parts)")
        for pattern_name, count in pattern_hits.items():
            log.debug(f"  {pattern_name}: {count} matches")
        # Show first removed part for debugging
        if removed_parts:
            log.debug(f"First removed: {removed_parts[0]}")
    
    return cleaned_html


def _clean_markdown_for_signature_extraction(markdown_content):
    """
    Säubert Markdown-Text für bessere Talon Signatur-Erkennung
    Entfernt Links, komplexe Formatierungen, aber behält die Textstruktur
    """
    # Entferne Markdown-Links aber behalte den Text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', markdown_content)
    
    # Entferne andere Markdown-Formatierungen
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
    text = re.sub(r'`([^`]+)`', r'\1', text)        # Code
    text = re.sub(r'#+\s*', '', text)               # Headers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # List items
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Numbered lists
    text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)      # Blockquotes
    text = re.sub(r'^\s*\|\s*', '', text, flags=re.MULTILINE)     # Tables
    text = re.sub(r'\s*\|\s*$', '', text, flags=re.MULTILINE)     # Tables
    text = re.sub(r'^-+$', '', text, flags=re.MULTILINE)          # Table separators
    
    # Normalisiere Whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = text.strip()
    
    return text


def _apply_signature_removal_to_markdown(original_markdown, clean_text, cleaned_text):
    """
    Wendet die Signatur-Entfernung vom gesäuberten Text auf das originale Markdown an
    Intelligente Mapping-Strategie: Sucht nach dem Punkt im Original-Markdown,
    der dem Signatur-Start im gesäuberten Text entspricht
    """
    if len(cleaned_text) >= len(clean_text):
        return original_markdown
    
    # Berechne wie viel vom gesäuberten Text entfernt wurde
    removed_chars = len(clean_text) - len(cleaned_text)
    
    if removed_chars < 10:  # Zu wenig entfernt, wahrscheinlich kein echte Signatur
        return original_markdown
    
    # Finde den Schnittpunkt im gesäuberten Text
    if len(cleaned_text) > 50:
        # Nimm die letzten 50 Zeichen des gesäuberten Texts als Referenz
        reference_text = cleaned_text[-50:].strip()
    else:
        # Bei kurzem Text, nimm alles
        reference_text = cleaned_text.strip()
    
    # Suche diese Referenz im Original-Markdown
    if reference_text:
        # Normalisiere beide Texte für besseren Vergleich
        orig_normalized = re.sub(r'\s+', ' ', original_markdown.strip())
        ref_normalized = re.sub(r'\s+', ' ', reference_text)
        
        # Suche die Position im normalisierten Original-Text
        pos = orig_normalized.find(ref_normalized)
        if pos >= 0:
            # Berechne ungefähre Position im ursprünglichen Text
            # Berücksichtige Whitespace-Unterschiede
            lines = original_markdown.split('\n')
            char_count = 0
            
            for i, line in enumerate(lines):
                char_count += len(re.sub(r'\s+', ' ', line.strip()))
                if char_count >= pos + len(ref_normalized):
                    # Finde einen guten Schnittpunkt (Leerzeile oder Ende eines Absatzes)
                    for j in range(i, max(0, i-3), -1):
                        if not lines[j].strip():  # Leerzeile gefunden
                            return '\n'.join(lines[:j]).strip()
                    # Fallback: Schneide nach dem aktuellen Absatz
                    return '\n'.join(lines[:i+1]).strip()
    
    # Fallback: Verwende Ratio-basierte Methode
    removed_ratio = removed_chars / len(clean_text)
    if removed_ratio > 0.15:  # Nur wenn signifikant was entfernt wurde
        estimated_cut_point = int(len(original_markdown) * (1 - removed_ratio))
        lines = original_markdown[:estimated_cut_point].split('\n')
        
        # Suche rückwärts nach einer guten Schnittstelle
        for i in range(len(lines) - 1, max(0, len(lines) - 10), -1):
            line = lines[i].strip()
            if not line or line in ['---', '* * *', '___']:  # Leere Zeile oder Separator
                return '\n'.join(lines[:i]).strip()
        
        return '\n'.join(lines).strip()
    
    return original_markdown


def _remove_markdown_signature_patterns(markdown_content, sender=None):
    """
    Fallback-Signatur-Entfernung für Markdown mit Multi-Sprach-Unterstützung
    """
    lines = markdown_content.split('\n')
    signature_start = None
    
    # Multi-Sprach-Salutationen erweitert
    salutations = [
        "Schöne Grüße / Best regards", "Best regards / Schöne Grüße",
        "Mit freundlichen Grüßen / Best regards", "Best regards / Mit freundlichen Grüßen",
        "Viele Grüße / Best regards", "Best regards / Viele Grüße",
        "Freundliche Grüße / Kind regards", "Kind regards / Freundliche Grüße",
        "Mit freundlichen Grüßen", "Freundliche Grüße", "Viele Grüße", "Beste Grüße",
        "Herzliche Grüße", "Liebe Grüße", "Schöne Grüße", "Mit besten Grüßen",
        "Best regards", "Kind regards", "Warm regards", "Sincerely", "Regards",
        "Best wishes", "Thank you", "Thanks", "Cheers", "All the best"
    ]
    
    # Suche nach Signatur-Start
    for i, line in enumerate(lines):
        line_clean = line.strip()
        
        # Direkte Salutation-Erkennung
        for salutation in salutations:
            if salutation.lower() in line_clean.lower():
                signature_start = i
                break
        
        if signature_start is not None:
            break
        
        # Auch nach typischen Signatur-Indikatoren suchen
        if any(indicator in line_clean for indicator in ['---', '* * *', '__']):
            signature_start = i
            break
            
        # Email-Adresse oder typische Signatur-Elemente
        if re.search(r'[\w\.-]+@[\w\.-]+', line_clean) and i > len(lines) // 2:
            signature_start = max(0, i - 2)  # Etwas früher anfangen
            break
    
    if signature_start is not None:
        content_part = '\n'.join(lines[:signature_start]).strip()
        signature_part = '\n'.join(lines[signature_start:]).strip()
        return content_part, signature_part
    
    return markdown_content, None


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5505)
