from talon import signature, quotations
from flask import Flask, request, jsonify, json
from werkzeug.exceptions import HTTPException, BadRequest
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as md, BACKSLASH, ATX
import re
import talon
import logging
import requests
import os
import hashlib
import openai
from urllib.parse import urljoin, urlparse
from pathlib import Path
import tempfile
import base64
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import time
import sys
import json

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

DEFAULT_AI_MODEL = "gpt-4.1-mini"

known_english_salutations = [
    "best regards", "kind regards", "warm regards", "regards", "with kind regards", "with best regards",
    "thanks", "thank you", "many thanks", "thanks again", "thanks and regards",
    "cheers", "sincerely", "sincerely yours", "yours truly", "yours faithfully",
    "with appreciation", "with gratitude", "respectfully", "respectfully yours",
    "with warmest regards", "warmest regards"
]

known_salutations = {
    "german": [
        "mit freundlichen grüßen", "freundliche grüße", "viele grüße", "beste grüße", "herzliche grüße",
        "liebe grüße", "schöne grüße", "hochachtungsvoll", "danke", "dankeschön", "danke schön",
        "danke und gruß", "danke und alles gute", "mfg", "mit besten grüßen", "gruß", "grüße"
    ],
    "danish": [
        "med venlig hilsen", "venlig hilsen", "mvh", "bedste hilsner", "tak", "på forhånd tak",
        "tak og venlig hilsen", "med tak", "de bedste hilsner"
    ],
    "norwegian": [
        "med vennlig hilsen", "vennlig hilsen", "takk", "på forhånd takk", "beste hilsener"
    ],
    "swedish": [
        "med vänliga hälsningar", "vänliga hälsningar", "bästa hälsningar", "tack", "tack så mycket"
    ],
    "finnish": [
        "ystävällisin terveisin", "parhain terveisin", "kiitos", "kiitoksia"
    ],
    "french": [
        "cordialement", "bien à vous", "salutations distinguées", "meilleures salutations",
        "avec mes salutations", "sincères salutations", "merci", "bien cordialement"
    ],
    "spanish": [
        "saludos cordiales", "atentamente", "cordialmente", "un saludo", "muchas gracias", "gracias"
    ],
    "portuguese": [
        "cumprimentos", "atenciosamente", "com os melhores cumprimentos", "obrigado", "obrigada"
    ],
    "italian": [
        "cordiali saluti", "distinti saluti", "grazie", "saluti", "un caro saluto", "con stima"
    ],
    "dutch": [  # Benelux
        "met vriendelijke groet", "vriendelijke groeten", "groeten", "bedankt", "hartelijke groeten"
    ],
    "polish": [
        "z poważaniem", "pozdrawiam", "serdeczne pozdrowienia", "dziękuję", "z wyrazami szacunku"
    ],
    "turkish": [
        "saygılarımla", "iyi çalışmalar", "teşekkür ederim", "selamlar", "iyi günler"
    ],
    "japanese": [
        "よろしくお願いします", "敬具", "よろしくお願いいたします", "ありがとうございます"
    ],
    "korean": [
        "감사합니다", "안부 인사드립니다", "고맙습니다", "진심으로 감사합니다"
    ],
    "chinese": [
        "此致敬礼", "谢谢", "祝好", "敬上", "感谢您"
    ],
    "english": [  # optional fallback
        "best regards", "kind regards", "warm regards", "thanks", "thank you", "cheers",
        "sincerely", "yours truly", "with appreciation", "regards"
    ],
    "brazilian_portuguese": [
        "atenciosamente", "obrigado", "obrigada", "com os melhores cumprimentos"
    ],
    "us_english": [  # optional for nuance
        "best regards", "sincerely", "yours truly", "respectfully", "with gratitude", "thanks"
    ]
}
# Gather all known salutations (lowercase) from all languages
all_salutations = set(s.lower() for s in known_english_salutations)
for lang_list in known_salutations.values():
    all_salutations.update(s.lower() for s in lang_list)

# Separator regex for bilingual salutations: slash (/), pipe (|), comma (,), semicolon (;)
# Allows optional whitespace around the separator.
separator_re = r'\s*[/|,;]\s*'

# Pattern for a known English salutation (case-insensitive, with word boundaries)
known = r'(' + '|'.join(re.escape(s) for s in known_english_salutations) + r')'

# Pattern for the "other side" of a bilingual salutation (any phrase except separator chars, 2–50 chars)
any_phrase = r'[^/|,;]{2,50}'

# Main pattern:
# - Matches lines that are either:
#   - a known English salutation only, e.g., "Best regards"
#   - a bilingual salutation: [English separator other], e.g., "Best regards | Mit freundlichen Grüßen"
#   - a bilingual salutation: [other separator English], e.g., "Mit freundlichen Grüßen / Best regards"
# - Allows optional whitespace and punctuation at the end
#
# Explanation:
#   ^\s*              : line start, optional whitespace
#   (                 : begin main group
#      (?:...|...)    : either [English separator other] or [other separator English]
#      | known        : OR just English only
#   )
#   \s*[,:\s]*$       : optional trailing comma, colon, whitespace, line end
#
known_en_pattern = re.compile(
    rf'(?i)^\s*('
    rf'(?:{known}{separator_re}{any_phrase})'      # English + separator + other language
    rf'|(?:{any_phrase}{separator_re}{known})'     # Other language + separator + English
    rf'|{known}'                                   # English only
    rf')\s*[,:\s]*$'
)

talon.init()

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

app = Flask(__name__)

# Configure logging to suppress health check requests
class HealthCheckFilter(logging.Filter):
    """Filter to suppress health check log entries."""
    def filter(self, record):
        return '/health' not in record.getMessage()

# Apply filter to werkzeug logger to suppress health check logs
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.addFilter(HealthCheckFilter())

# Global OpenAI client cache to avoid recreating clients repeatedly
_openai_clients = {}

def sanitize_url_for_logging(url):
    """
    Sanitizes URLs for logging by removing sensitive parameters like access_token.
    """
    if not url:
        return url
    
    try:
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        
        # Remove sensitive parameters
        sensitive_params = ['access_token', 'token', 'api_key', 'secret', 'password', 'auth']
        for param in sensitive_params:
            if param in query_params:
                query_params[param] = ['***REDACTED***']
        
        # Reconstruct URL with sanitized query
        sanitized_query = urlencode(query_params, doseq=True)
        sanitized_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            sanitized_query,
            parsed.fragment
        ))
        
        return sanitized_url
    except Exception:
        # Fallback: simple regex replacement for common patterns
        import re
        sanitized = re.sub(r'([?&])(access_token|token|api_key|secret|password|auth)=([^&]*)', r'\1\2=***REDACTED***', url)
        return sanitized

def get_openai_client(api_key):
    """
    Returns a cached OpenAI client for the given API key.
    Creates a new client only if not already cached.
    """
    if api_key not in _openai_clients:
        try:
            _openai_clients[api_key] = openai.OpenAI(api_key=api_key)
            log.debug(f"Created new OpenAI client (cache size: {len(_openai_clients)})")
        except Exception as e:
            log.error(f"Failed to create OpenAI client: {str(e)}")
            return None
    return _openai_clients[api_key]


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
    """Health check endpoint for docker-compose and monitoring."""
    try:
        # Basic health status
        health_status = {
            "status": "healthy",
            "service": "talon-web",
            "version": "1.0.0",
            "timestamp": None
        }
          # Add timestamp
        import datetime
        health_status["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Test basic functionality - try to import talon modules
        try:
            from talon import signature, quotations
            health_status["talon_modules"] = "ok"
        except Exception as e:
            health_status["talon_modules"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        # Return appropriate HTTP status code
        if health_status["status"] == "healthy":
            return jsonify(health_status), 200
        else:
            return jsonify(health_status), 503
            
    except Exception as e:
        log.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": "An internal error occurred. Please contact support.",
            "service": "talon-web"
        }), 503


@app.route('/', methods=['GET'])
def root():
    """Root endpoint providing basic service information."""
    return jsonify({
        "service": "talon-web",
        "description": "Talon Email Processing API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "signature_extraction": "/talon/signature",
            "quotations_text": "/talon/quotations/text", 
            "quotations_html": "/talon/quotations/html",
            "html_to_markdown": "/talon/html-to-markdown"
        },
        "status": "running"
    })


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
        openai_api_key = data.get('openai_api_key')        
        base_url = data.get('base_url')
        auth_query_params = data.get('auth_query_params', '')
        image_path = data.get('image_path', './images/')
        image_prefix = data.get('image_prefix', None)
        ai_model = data.get('ai_model', DEFAULT_AI_MODEL)
        ai_prompt = data.get('ai_prompt', '')
        ai_model = data.get('ai_model', 'gpt-4.1-mini')
        ai_signature_extraction = data.get('ai_signature_extraction', False)
    else:
        html_content = request.form.get('html')
        sender = request.form.get('email_sender')
        openai_api_key = request.form.get('openai_api_key')
        base_url = request.form.get('base_url')
        auth_query_params = request.form.get('auth_query_params', '')
        image_path = request.form.get('image_path', './images/')
        image_prefix = request.form.get('image_prefix', None)
        ai_model = request.form.get('ai_model', DEFAULT_AI_MODEL)
        ai_prompt = request.form.get('ai_prompt', '')
        ai_model = request.form.get('ai_model', 'gpt-4.1-mini')
        ai_signature_extraction = request.form.get('ai_signature_extraction', 'false').lower() == 'true'
        
    if not html_content:
        raise BadRequest("Required parameter 'html' is missing.")
    
    auth_config = [];
    if isinstance(auth_query_params, str):
        # Prüfe ob es JSON ist
        try:
            auth_config = json.loads(auth_query_params)
        except (json.JSONDecodeError, TypeError):
            # Fallback: Verwende den String als globale Auth-Parameter
            log.warning("auth_query_params is not valid JSON, using as global parameter")
    elif isinstance(auth_query_params, dict):
        auth_config = auth_query_params
    elif auth_query_params is not None:
        log.warning("auth_query_params is not a valid type, expected dict or JSON string")

    try:
        # Zeitmessung für die einzelnen Schritte
        timings = {}
        t0 = time.perf_counter()

        # 1. Vorverarbeitung: HTML bereinigen
        soup = BeautifulSoup(html_content, "html.parser")
        # Entferne störende Tags
        for tag in soup(["style", "script", "header"]):
            tag.decompose()
        timings['html_cleanup'] = time.perf_counter() - t0
        t1 = time.perf_counter()

        removed_trackers = remove_tracking_pixels(soup)
        timings['tracker_removal'] = time.perf_counter() - t1

        if removed_trackers:
            log.info(f"Removed {len(removed_trackers)} tracking pixels: {removed_trackers}")

        # 2. Signatur aus HTML entfernen
        sig_html = None
        clean_html, sig_html = signature.extract(str(soup), sender=sender or "")

        timings['signature_extraction_html'] = time.perf_counter() - t1
        t2 = time.perf_counter()        # 3. HTML zu Markdown konvertieren
        markdown = md(
            clean_html,
            heading_style=ATX
        )
        timings['html_to_markdown'] = time.perf_counter() - t2
        t3 = time.perf_counter()

        # 5. Zitat entfernen via Talon
        markdown = quotations.extract_from_plain(markdown)
        sig = None
        markdown, sig = signature.extract(markdown, sender=sender or "")
        timings['quotation_and_signature_extraction'] = time.perf_counter() - t3
        t6 = time.perf_counter()

        if openai_api_key and ai_signature_extraction:
            # 7. AI-gestützte Grußformel-Erkennung und -Extraktion
            final_markdown = extract_content_until_salutation_with_ai(markdown, openai_api_key, ai_model)
            log.info("AI signature extraction completed.")
            timings['ai_signature_extraction'] = time.perf_counter() - t6
            t7 = time.perf_counter()
        else:
            # 7. Grußformel + Name aus Markdown beibehalten, danach abschneiden
            final_markdown = extract_content_until_salutation(markdown)
            timings['ai_signature_extraction'] = 0.0
            t7 = time.perf_counter()

        # 8. AI-Bildverarbeitung - NUR wenn OpenAI Key vorhanden und NACH der Signaturentfernung
        image_info = {}
        if openai_api_key and final_markdown:
            image_info = process_images_parallel_with_ai(
                final_markdown, openai_api_key, base_url, auth_config, image_path, image_prefix, ai_prompt, ai_model
            )
            timings['ai_image_processing'] = time.perf_counter() - t7
            t8 = time.perf_counter()
            if image_info:
                final_markdown = replace_images_with_ai_descriptions(final_markdown, image_info)
                timings['ai_image_replacement'] = time.perf_counter() - t8
            else:
                timings['ai_image_replacement'] = 0.0
        else:
            timings['ai_image_processing'] = 0.0
            timings['ai_image_replacement'] = 0.0

        # 9. Signatur-Behandlung für Response
        sig_markdown = None
        if final_markdown != markdown:
            sig_markdown = re.sub(rf'^{re.escape(final_markdown)}\s*', '', markdown, flags=re.MULTILINE).strip()
            if not sig_markdown:
                sig_markdown = None

        response_data = {
            'markdown': final_markdown,
            'email_sender': sender,
        }

        timings['total_processing_time'] = time.perf_counter() - t0

        response_data['processing_summary'] = {
            'content' : {
                'intermediate_markdown': markdown,
                'original_html': html_content,
                'removed_markdown_signature': str(sig_markdown) if sig_markdown else None,
                'removed_html_signature': str(sig_html) if sig_html else None,
                'removed_plain_signature': str(sig) if sig else None
            },
            'timing': timings
        }
        
        # Füge Bildinformationen hinzu wenn vorhanden
        if image_info:
            response_data['processed_images'] = {}
            response_data['processing_summary']['ai'] = {
                'total_images': len(image_info),
                'ai_model_used': ai_model,
                'custom_prompt_used': bool(ai_prompt.strip()),
                'batch_processing': len(image_info) > 1
            }
            
            for src, info in image_info.items():
                response_data['processed_images'][src] = {
                    'filename': info['filename'],
                    'ai_description': info['ai_description'],
                    'alt_text': info['alt_text'],
                    'local_path': info['local_path'],
                    'base64_data': info.get('base64_data', ''),  # Base64 kodierte Bilddaten
                    'content_type': info.get('content_type', 'image/jpeg'),
                    'original_src': info.get('original_src', src),
                    'full_url': info.get('full_url', src)
                }

        log.info(f"HTML to Markdown processing completed in {timings['total_processing_time']:.4f}s")
        return jsonify(response_data)

    except Exception as e:
        log.error(f"Error processing HTML to Markdown: {str(e)}")

def extract_content_until_salutation_with_ai(markdown: str, openai_api_key: str, ai_model=DEFAULT_AI_MODEL) -> str:
    """
    Nutzt AI, um
    - die letzte Zeile der Grußformel im Markdown zu finden
    - optional eine vereinfachte Signatur/Schlusszeile von der KI generieren zu lassen
    Schneidet den Markdown-Text bis dorthin zu und hängt die ggf. neue Schlusszeile an.

    Returns:
        Gekürzter Markdown-Text mit neuer Schlusszeile (wenn von KI geliefert), 
        sonst Originaltext bei Fehlern.
    """
    client = get_openai_client(openai_api_key)
    
    # Prompt zur Zeilenerkennung und optionale vereinfachte Signatur
    prompt = (
        "You receive a Markdown email text.\n"
        "Your task is to remove the full original signature and replace it by a short, clean, professional signature line with name, position and one contact detail.\n"
        "Output the entire cleaned Markdown text including greeting, body, and new simplified signature, preserving original formatting and line breaks where possible.\n"
        "Do not add explanations, output only the cleaned Markdown.\n\n"
        f"Email Markdown:\n{markdown}"
    )

    input_token_estimate = int(len(markdown) / 3.5)  # konservativ
    max_tokens = min(1500, max(300, int(input_token_estimate * 1.2)))  # Reserve für Output


    try:
        response = client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": "You are an assistant specialized in email content processing."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.1
        )
        cleaned_markdown = response.choices[0].message.content.strip()
        return cleaned_markdown

    except Exception as e:
        log.error(f"Error during AI processing: {e}")
        return markdown
    
def remove_tracking_pixels(soup: BeautifulSoup) -> list[str]:
    """
    Removes likely tracking pixels (1x1 images, invisible or suspicious sources) from the HTML.
    
    Args:
        soup (BeautifulSoup): Parsed HTML document
    
    Returns:
        List[str]: List of removed image `src` values for logging/debugging
    """
    removed_sources = []

    for img in soup.find_all("img"):
        if not isinstance(img, Tag):
            continue

        src = img.get("src", "").lower()
        width = img.get("width")
        height = img.get("height")
        style = img.get("style", "").lower()
        alt = img.get("alt", "").strip()

        suspicious_domain = any(domain in src for domain in [
            "hubspotlinks.com", "hs-analytics.net", "mailchimp.com", "clickdimensions.com",
            "mandrillapp.com", "sendgrid.net", "emsecure.net"
        ])

        tiny_pixel = (
            (width == "1" and height == "1")
            or ("width: 1px" in style and "height: 1px" in style)
            or "1x1" in src
        )

        hidden_style = (
            "display: none" in style
            or "visibility: hidden" in style
            or "opacity: 0" in style
        )

        if suspicious_domain or tiny_pixel or hidden_style:
            removed_sources.append(src)
            img.decompose()

    return removed_sources

def remove_social_links_from_line(line: str) -> str:
    """
    Entfernt alle ![](...) und [](...) Links zu bekannten Social-Media-Plattformen
    (z. B. Twitter, LinkedIn, YouTube, Blog etc.), unabhängig davon ob sie am
    Anfang, in der Mitte oder am Ende der Zeile stehen.
    """
    social_re = re.compile(
        r'(!?\[\])\(\s*(https?:\/\/)?(www\.)?(x\.com|twitter\.com|linkedin\.com|youtube\.com|youtu\.be|'
        r'blog\.[^)]*|[^)]*utm_.*?)\s*\)',
        re.IGNORECASE
    )
    return social_re.sub('', line).strip()

def is_salutation_line(line: str) -> bool:
    line_clean = line.strip().lower().strip("!,: ")

    if line_clean in all_salutations:
        return True
    if known_en_pattern.match(line.strip()):
        return True
    return False

def extract_content_until_salutation(markdown):
    """
    Extrahiert Inhalt bis zur Grußformel + Name und ggf. nachfolgende Zeilen,
    solange kein Trennzeichen wie ___ oder --- oder | erscheint.
    Zusätzlich werden typische Social-Media-/Marketing-Links entfernt.
    """
    lines = markdown.strip().splitlines()
    result = []

    # Trennzeichen (---, ___, |, etc.)
    separator_re = re.compile(r'^[-_|\s]{3,}$')
    image_re = re.compile(r'!\[.*?\]\(.*?\)')  # Markdown-Bild-Zeile

    found_salute = False
    i = 0
    j = 0
    while (i + j) < len(lines) and j < 6:
        line = remove_social_links_from_line(lines[i + j])
        line_stripped = line.strip()

        if not found_salute and is_salutation_line(line_stripped):
            found_salute = True
            j += 1
        elif found_salute:
            if separator_re.match(line_stripped) or image_re.search(line_stripped):
                break
            else:
                j += 1
        else:
            i += 1

        result.append(line)

    return "\n".join(result).strip()



def get_file_extension_from_url(url):
    """Ermittelt die Dateierweiterung aus einer URL."""
    parsed = urlparse(url)
    path = parsed.path.lower()
    
    # Bekannte Bildformate
    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']:
        if ext in path:
            return ext
            
    # Default zu .jpg
    return '.jpg'


def get_content_type_from_extension(file_extension):
    """Ermittelt den Content-Type aus einer Dateierweiterung."""
    extension_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.svg': 'image/svg+xml'
    }
    return extension_map.get(file_extension.lower(), 'image/jpeg')


def replace_images_with_ai_descriptions(markdown, image_info):
    """Ersetzt Bilder in Markdown mit erweiterten Beschreibungen."""
    
    # Finde alle Markdown-Bilder ![alt](src)
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    def replace_image(match):
        alt_text = match.group(1)
        src = match.group(2)
        
        # Prüfe ob wir Informationen für dieses Bild haben
        if src in image_info:
            info = image_info[src]
            
            # Verwende Alt-Text oder generiere einen aus der AI-Beschreibung
            if not alt_text.strip():
                # Nehme die ersten Worte der AI-Beschreibung als Alt-Text
                ai_words = info['ai_description'].split()[:6]
                alt_text = ' '.join(ai_words)
                if len(info['ai_description'].split()) > 6:
                    alt_text += '...'
            
            # Erstelle erweiterte Markdown-Syntax
            enhanced_markdown = f"""![{alt_text}]({info['local_path']})

> **Image-description (AI):**
> {info['ai_description']}"""
            
            return enhanced_markdown
        else:
            # Fallback: Originales Markdown-Bild
            return match.group(0)
    
    # Ersetze alle Bilder
    enhanced_markdown = re.sub(image_pattern, replace_image, markdown)
    
    return enhanced_markdown

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


def download_image(url, local_path):
    """Lädt ein Bild von einer URL herunter und speichert es lokal (optimiert ohne Cache)."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
          # get the current time to measure download time
        current_time = time.time()

        # Reduziertes Timeout für bessere Performance
        response = requests.get(url, headers=headers, timeout=8, stream=True)
        response.raise_for_status()
        
        # Prüfe Content-Type
        content_type = response.headers.get('content-type', '').lower()        
        if not content_type.startswith('image/'):
            log.warning(f"URL does not return an image: {sanitize_url_for_logging(url)} (Content-Type: {content_type})")
            return False
              # Speichere das Bild
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        log.info(f"Image downloaded: {sanitize_url_for_logging(url)} -> {local_path} (Time: {time.time() - current_time:.2f}s)")
        return True
        
    except Exception as e:
        log.error(f"Error downloading image {sanitize_url_for_logging(url)}: {str(e)}")
        return False

def process_single_image_with_ai(task_data, openai_api_key, markdown_context, ai_prompt, ai_model):
    """
    Processes a single image: downloads it and generates AI description.
    This function is designed to be run in parallel for multiple images.
    
    Args:
        task_data: Dictionary with image task information
        openai_api_key: OpenAI API key
        markdown_context: The markdown context for AI description
        ai_prompt: Custom AI prompt
        ai_model: AI model to use
    
    Returns:
        Dictionary with processed image information or None if failed    """
    temp_file = None
    client = None
    
    try:
        # Initialize OpenAI client
        client = get_openai_client(openai_api_key)
        if not client:
            log.error("Failed to get OpenAI client")
            return None
          # Download the image
        success = download_image(task_data['url'], task_data['path'])
        if not success:
            log.error(f"Failed to download image: {sanitize_url_for_logging(task_data['url'])}")
            return None
            
        temp_file = task_data['path']
        
        # Generate AI description using the marker approach
        ai_description = generate_ai_description_with_marker(
            client, task_data['path'], task_data['alt_text'], 
            markdown_context, ai_prompt, ai_model
        )
        
        # Encode image as base64
        with open(task_data['path'], 'rb') as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')
            
        # Return complete image information
        return {
            'src': task_data['src'],
            'original_src': task_data['src'],
            'full_url': task_data['url'],
            'local_path': task_data['local_path'],
            'filename': task_data['filename'],
            'alt_text': task_data['alt_text'],
            'ai_description': ai_description,
            'base64_data': base64_data,
            'content_type': get_content_type_from_extension(task_data['file_extension']),
            'success': True
        }
        
    except Exception as e:
        log.error(f"Error processing image {task_data.get('src', 'unknown')}: {str(e)}")
        return None
        
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                log.warning(f"Failed to delete temp file {temp_file}: {str(e)}")
def generate_ai_description_with_marker(client, image_path, alt_text, markdown_context, custom_prompt="", model=DEFAULT_AI_MODEL):
    """
    Generates AI description for a single image using the ![CURRENT_IMAGE]() marker approach.
    Optimized for speed - no caching since we process different images each time.
    """
    
    try:
        # Use custom prompt or default
        if not custom_prompt.strip():
            custom_prompt = """You will receive email content in Markdown format where ![CURRENT_IMAGE]() represents the image you need to describe.

Generate an accurate, concise description for this image considering its context in the email.

Provide a detailed description (1-2 sentences) emphasizing relevant details for the email context.

Reply with just the description text, no additional formatting."""

        # Create the context with the marker
        enhanced_context = markdown_context.replace('![CURRENT_IMAGE]()', f'![CURRENT_IMAGE]() - THIS IS THE IMAGE YOU SHOULD DESCRIBE')
        
        # Read and encode the image
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
        content_type = get_content_type_from_extension(os.path.splitext(image_path)[1])
        
        # Create the prompt with context
        full_prompt = f"""{custom_prompt}

Email context:
{enhanced_context}

Original alt text (if available): "{alt_text}"

Please provide a description for the image marked as ![CURRENT_IMAGE]()."""

        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{image_data}",
                                "detail": "low"  # Cost and speed optimization
                            }
                        }
                    ]
                }
            ],
            max_tokens=200,  # Reduced for speed
            temperature=0.1
        )
        
        description = response.choices[0].message.content.strip()
        log.debug(f"AI description generated: {description[:50]}...")
        
        return description
        
    except Exception as e:
        log.error(f"Error generating AI description with marker: {str(e)}")
        return "AI description could not be generated."
    
def process_images_parallel_with_ai(markdown, openai_api_key, base_url, auth_query_params, image_path, image_prefix, ai_prompt="", ai_model=DEFAULT_AI_MODEL, max_workers=6):
    """
    Processes all images from markdown in parallel - each image is downloaded and AI-described individually.
    Uses the ![CURRENT_IMAGE]() marker approach for contextual AI descriptions.
    
    Optimizations:
    - Increased max_workers to 6 for better parallelization
    - Reduced timeouts from 15s to 8s
    - No caching (since different images each time)
    - Faster AI model settings
    
    Args:
        markdown: The final markdown content (without signature)
        openai_api_key: OpenAI API key
        base_url: Base URL for relative image paths
        auth_query_params: Domain-specific authentication parameters. Can be:
                            '{"my.service-api.net": "token=abc123", "other.com": "key=xyz"}'
        image_path: Path prefix for local images
        image_prefix: Prefix for filenames
        ai_prompt: Custom AI prompt
        ai_model: AI model (default: gpt-4.1-mini)
        max_workers: Maximum number of parallel workers (default: 6)
    
    Returns:
        dict: Image information with AI descriptions
    """
    # Find all markdown images with regex: ![alt](src)
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    image_matches = re.findall(image_pattern, markdown)
    
    if not image_matches:
        return {}
    
    log.info(f"Found {len(image_matches)} images for parallel processing")
    
    # Create temporary directory for downloads
    temp_dir = tempfile.mkdtemp(prefix="talon_parallel_images_")
    
    try:
        # Prepare tasks for parallel processing
        tasks = []
        for idx, (alt_text, src) in enumerate(image_matches, 1):            
            try:
                # Create absolute URL
                if base_url and not src.startswith(('http://', 'https://', 'data:')):
                    full_url = urljoin(base_url, src)
                else:
                    full_url = src
                
                # Apply domain-specific auth parameters for ALL URLs (not just relative ones)
                if auth_query_params and not src.startswith('data:'):
                    domain_auth_params = get_auth_params_for_domain(auth_query_params, full_url)
                    if domain_auth_params:
                        parsed_url = urlparse(full_url)
                        query = parsed_url.query
                        if query:
                            full_url += f"&{domain_auth_params}"
                        else:
                            full_url += f"?{domain_auth_params}"
                    
                # Skip data URLs (base64 embedded images)
                if src.startswith('data:'):
                    log.info(f"Skipping data URL image: {src[:50]}...")
                    continue
                    
                # Generiere temporären Bildnamen
                url_hash = hashlib.md5(full_url.encode()).hexdigest()[:8]
                file_extension = get_file_extension_from_url(full_url)
                filename = f"{image_prefix}{url_hash}{file_extension}" if image_prefix else f"image_{idx}_{url_hash}{file_extension}"
                temp_path = os.path.join(temp_dir, filename)
                local_path = os.path.join(image_path, filename).replace('\\', '/')
                
                # Erstelle Kontext mit ![CURRENT_IMAGE]() Marker für dieses spezielle Bild
                image_context = markdown.replace(f'![{alt_text}]({src})', '![CURRENT_IMAGE]()', 1)
                
                # Erstelle Task-Daten
                task_data = {
                    'url': full_url,
                    'path': temp_path,
                    'src': src,
                    'alt_text': alt_text,
                    'filename': filename,
                    'local_path': local_path,
                    'file_extension': file_extension,
                    'image_context': image_context
                }
                
                tasks.append(task_data)
                    
            except Exception as e:
                log.error(f"Error preparing parallel task for image {src}: {str(e)}")
        
        # Verarbeite alle Bilder parallel
        image_info = {}
        if tasks:
            log.info(f"Starting parallel processing of {len(tasks)} images with {max_workers} workers...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Alle Tasks einreichen
                future_to_task = {
                    executor.submit(
                        process_single_image_with_ai, 
                        task, 
                        openai_api_key, 
                        task['image_context'], 
                        ai_prompt, 
                        ai_model
                    ): task for task in tasks
                }
                  # Ergebnisse sammeln, sobald sie abgeschlossen sind
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        if result and result.get('success'):
                            image_info[result['src']] = result
                            log.info(f"Successfully processed image: {result['filename']}")
                        else:
                            log.warning(f"Failed to process image: {task['src']}")
                    except Exception as e:
                        log.error(f"Error in parallel task for {task['src']}: {str(e)}")
            
            log.info(f"Parallel processing completed: {len(image_info)}/{len(tasks)} images successful")
            return image_info
        else:
            log.info("No images found for parallel processing.")
            return {}
    finally:
        # Temporäres Verzeichnis aufräumen
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            log.warning(f"Failed to delete temp directory {temp_dir}: {str(e)}")
            
def get_auth_params_for_domain(auth_config, url):
    """
    Ermittelt die passenden Auth-Parameter für eine URL basierend auf ihrer Domain.
    
    Args:
        auth_config: String (backward compatibility) oder Dict mit Domain-Auth-Mapping
        url: Die URL für die Auth-Parameter gesucht werden
    
    Returns:
        String mit Auth-Parametern oder leerer String
    
    Expected auth_config format (JSON):
    {
        "my.service-api.net": "token=abc123&user=john",
        "other-domain.com": "apikey=xyz789",
        "default": "common_param=value"  # optional fallback
    }
    """
    if not auth_config:
        return ""
    
    # Backward compatibility: Wenn auth_config ein String ist
    if isinstance(auth_config, dict):
        auth_dict = auth_config
    else:
        log.warning(f"Invalid auth_config type: {type(auth_config)}")
        return ""
    
    # Extrahiere Domain aus URL
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Suche exakte Domain-Übereinstimmung
        if domain in auth_dict:
            log.debug(f"Found auth params for domain {domain}")
            return auth_dict[domain]
        
        # Suche nach Subdomain-Übereinstimmung (z.B. api.domain.com -> domain.com)
        for auth_domain in auth_dict.keys():
            if auth_domain != "default" and domain.endswith('.' + auth_domain):
                log.debug(f"Found auth params for parent domain {auth_domain} (url domain: {domain})")
                return auth_dict[auth_domain]
        
        # Fallback auf default
        if "default" in auth_dict:
            log.debug(f"Using default auth params for domain {domain}")
            return auth_dict["default"]
            
        log.debug(f"No auth params found for domain {domain}")
        return ""
        
    except Exception as e:
        log.error(f"Error parsing URL {url} for auth params: {str(e)}")
        return ""

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Talon Web Bootstrap Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5505, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    log.info(f"Starting Talon Web Bootstrap Server on {args.host}:{args.port}")
    log.info(f"Debug mode: {args.debug}")
    
    # For production use, consider using a proper WSGI server like gunicorn
    if os.environ.get('PRODUCTION', 'false').lower() == 'true':
        log.info("Production mode detected - consider using gunicorn")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,  # Disable reloader to avoid duplicate initialization
        threaded=True  # Enable threading for better performance
    )