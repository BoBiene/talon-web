from talon import signature, quotations
from talon import signature, quotations
from flask import Flask, request, jsonify, json
from werkzeug.exceptions import HTTPException, BadRequest
from bs4 import BeautifulSoup
import html2text
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
        openai_api_key = data.get('openai_api_key')        
        base_url = data.get('base_url')
        auth_query_params = data.get('auth_query_params', '')
        image_path = data.get('image_path', './images/')
        image_prefix = data.get('image_prefix', '')
        ai_prompt = data.get('ai_prompt', '')
        ai_model = data.get('ai_model', 'gpt-4.1-mini')
    else:
        html_content = request.form.get('html')
        sender = request.form.get('email_sender')
        openai_api_key = request.form.get('openai_api_key')
        base_url = request.form.get('base_url')
        auth_query_params = request.form.get('auth_query_params', '')
        image_path = request.form.get('image_path', './images/')
        image_prefix = request.form.get('image_prefix', '')
        ai_prompt = request.form.get('ai_prompt', '')
        ai_model = request.form.get('ai_model', 'gpt-4.1-mini')
        
    if not html_content:
        raise BadRequest("Required parameter 'html' is missing.")

    try:
        # 1. Vorverarbeitung: HTML bereinigen
        soup = BeautifulSoup(html_content, "html.parser")
        # Entferne störende Tags
        for tag in soup(["style", "script", "header"]):
            tag.decompose()
        
        # 2. Signatur aus HTML entfernen
        sig_html = None
        clean_html, sig_html = signature.extract(str(soup), sender=sender or "")

        # 3. HTML zu Markdown konvertieren
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_emphasis = False
        h.body_width = 0
        h.unicode_snob = True
        markdown = h.handle(clean_html)

        # 4. Markdown-Postprocessing: Whitespace-Optimierung
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        markdown = '\n'.join([l.rstrip() for l in markdown.splitlines()])
        markdown = markdown.strip()

        # 5. Zitat entfernen via Talon
        markdown = quotations.extract_from_plain(markdown)
        sig = None
        markdown, sig = signature.extract(markdown, sender=sender or "")

        # 6. Markdown-hardbreaks entfernen
        markdown = re.sub(r'[ \t]+\n', '\n', markdown)
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)

        if openai_api_key:
            # 7. AI-gestützte Grußformel-Erkennung und -Extraktion
            final_markdown = extract_content_until_salutation_with_ai(markdown, openai_api_key, ai_model)
        else:
            # 7. Grußformel + Name aus Markdown beibehalten, danach abschneiden
            final_markdown = extract_content_until_salutation(markdown)
        
        # 8. AI-Bildverarbeitung - NUR wenn OpenAI Key vorhanden und NACH der Signaturentfernung
        image_info = {}
        if openai_api_key and final_markdown:
            image_info = process_images_from_markdown(
                final_markdown, openai_api_key, base_url, auth_query_params, image_path, image_prefix, ai_prompt, ai_model
            )
            
            # Bilder in Markdown mit AI-Beschreibungen ersetzen
            if image_info:
                final_markdown = replace_images_with_ai_descriptions(final_markdown, image_info)

        # 9. Signatur-Behandlung für Response
        sig_markdown = None
        if final_markdown != markdown:
            sig_markdown = re.sub(rf'^{re.escape(final_markdown)}\s*', '', markdown, flags=re.MULTILINE).strip()
            if not sig_markdown:
                sig_markdown = None

        response_data = {
            'original_html': html_content,
            'markdown': final_markdown,
            'intermediate_markdown': markdown,
            'email_sender': sender,
            'removed_markdown_signature': str(sig_markdown) if sig_markdown else None,
            'removed_html_signature': str(sig_html) if sig_html else None,
            'removed_plain_signature': str(sig) if sig else None
        }
        
        # Füge Bildinformationen hinzu wenn vorhanden
        if image_info:
            response_data['processed_images'] = {}
            response_data['ai_processing_summary'] = {
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

        return jsonify(response_data)

    except Exception as e:
        log.error(f"Error processing HTML to Markdown: {str(e)}")
        raise BadRequest(f"Error processing HTML: {str(e)}")

def extract_content_until_salutation_with_ai(markdown: str, openai_api_key: str, ai_model="gpt-4.1-mini") -> str:
    """
    Nutzt AI nur zur Erkennung der Zeilennummer (oder Position) der Grußformel,
    schneidet dann den Markdown-Text bis dorthin zu, um Markdown intakt zu halten.
    """

    try:
        client = openai.OpenAI(api_key=openai_api_key)
    except Exception as e:
        log.error(f"Failed to initialize OpenAI client: {str(e)}")
        return markdown  # Fallback auf Original-Markdown bei Fehler

    prompt = (
        "Analyze the following Markdown text and give me the line number of the last "
        "line that belongs to the salutation and the sender's name. "
        "Respond with only the number, without any further explanations.\n\n"
        f"Markdown text:\n{markdown}"
    )

    try:
        response = client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for extracting email content."},
                {"role": "user", "content": f"{prompt}\n\nMarkdown-Text:\n{markdown}"}
            ],
            max_tokens=1000,
            temperature=0.1
        )

        line_number_str =  response.choices[0].message.content.strip()
        line_number = int(re.findall(r'\d+', line_number_str)[0])
        log.info(f"AI detected salutation ending at line: {line_number}")

        # Markdown bis zur Zeile abschneiden (1-basiert)
        markdown_lines = markdown.splitlines()
        extracted = "\n".join(markdown_lines[:line_number])

        return extracted

    except Exception as e:
        log.error(f"Error during AI processing or parsing: {e}")
        # Fallback: komplette Markdown zurückgeben
        return markdown

def extract_content_until_salutation(markdown):
    """
    Extrahiert Inhalt bis zur Grußformel + Name und schneidet danach ab.
    Dies ist die elegante Version der Grußformel-Logik.
    """
    lines = markdown.strip().splitlines()
    result = []
    
    # Verbesserte Regex-Patterns für deutsche und englische Grußformeln
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

    return "\n".join(result).strip()


def process_images_from_markdown(markdown, openai_api_key, base_url, auth_query_params, image_path, image_prefix, ai_prompt="", ai_model="gpt-4.1-mini"):
    """
    Elegante Lösung: Findet Bilder im finalen Markdown und verarbeitet sie mit AI.
    
    Args:
        markdown: Das finale Markdown (ohne Signatur)
        openai_api_key: OpenAI API Schlüssel
        base_url: Basis-URL für relative Bildpfade
        auth_query_params: Authentifizierungs-Query-Parameter für Bilder (optional)
        image_path: Pfad-Präfix für lokale Bilder
        image_prefix: Präfix für Dateinamen
        ai_prompt: Benutzerdefinierter AI-Prompt
        ai_model: AI-Modell (default: gpt-4.1-mini)
    
    Returns:
        dict: Bildinformationen mit AI-Beschreibungen
    """
    import re
    import tempfile
    import os
    import hashlib
    import base64
    import shutil
    
    # Finde alle Markdown-Bilder mit Regex: ![alt](src)
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    image_matches = re.findall(image_pattern, markdown)
    
    if not image_matches:
        return {}
    
    log.info(f"Found {len(image_matches)} images in markdown")
    
    # OpenAI Client initialisieren
    try:
        import openai
        client = openai.OpenAI(api_key=openai_api_key)
    except Exception as e:
        log.error(f"Failed to initialize OpenAI client: {str(e)}")
        return {}
      # Temporäres Verzeichnis erstellen
    temp_dir = tempfile.mkdtemp(prefix="talon_markdown_images_")
    temp_files = []
    images_to_process = []
    
    try:
        # Sammle alle Bilder zum parallelen Download
        download_tasks = []
        for idx, (alt_text, src) in enumerate(image_matches, 1):
            try:
                # Absolute URL erstellen
                if base_url and not src.startswith(('http://', 'https://', 'data:')):
                    from urllib.parse import urljoin
                    full_url = urljoin(base_url, src)

                    if auth_query_params:
                        parsed_url = urlparse(full_url)
                        query = parsed_url.query
                        if query:
                            full_url += f"&{auth_query_params}"
                        else:
                            full_url += f"?{auth_query_params}"
                else:
                    full_url = src
                    
                # Skip data URLs (base64 embedded images)
                if src.startswith('data:'):
                    log.info(f"Skipping data URL image: {src[:50]}...")
                    continue
                    
                # Temporären Bildnamen generieren
                url_hash = hashlib.md5(full_url.encode()).hexdigest()[:8]
                file_extension = get_file_extension_from_url(full_url)
                filename = f"{image_prefix}{url_hash}{file_extension}" if image_prefix else f"image_{idx}_{url_hash}{file_extension}"
                temp_path = os.path.join(temp_dir, filename)
                local_path = os.path.join(image_path, filename).replace('\\', '/')
                
                # Sammle Download-Task
                download_tasks.append({
                    'url': full_url,
                    'path': temp_path,
                    'src': src,
                    'alt_text': alt_text,
                    'filename': filename,
                    'local_path': local_path,
                    'file_extension': file_extension
                })
                    
            except Exception as e:
                log.error(f"Error preparing image {src}: {str(e)}")
        
        # Paralleler Download aller Bilder
        if download_tasks:
            log.info(f"Starting parallel download of {len(download_tasks)} images...")
            download_results = download_images_parallel(download_tasks, max_workers=4)
            
            # Verarbeite erfolgreiche Downloads
            for result in download_results:
                if result['success']:
                    temp_files.append(result['path'])
                    images_to_process.append({
                        'src': result['src'],
                        'alt_text': result['alt_text'],
                        'full_url': result['url'],
                        'filename': result['filename'],
                        'temp_path': result['path'],
                        'local_path': result['local_path'],
                        'file_extension': result['file_extension']
                    })
        
        # AI-Beschreibungen generieren
        image_info = {}
        if images_to_process:
            ai_descriptions = generate_ai_descriptions_batch(
                client, images_to_process, markdown, ai_prompt, ai_model
            )
            
            # Ergebnisse kombinieren
            for i, img_data in enumerate(images_to_process):
                ai_desc = ai_descriptions.get(str(i), {})
                
                # Bild als Base64 kodieren
                with open(img_data['temp_path'], 'rb') as img_file:
                    base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                  # Informationen speichern
                image_info[img_data['src']] = {
                    'original_src': img_data['src'],
                    'full_url': img_data['full_url'],
                    'local_path': img_data['local_path'],
                    'filename': img_data['filename'],
                    'alt_text': ai_desc.get('alt_text', img_data['alt_text']),
                    'ai_description': ai_desc.get('description', 'Bildbeschreibung konnte nicht generiert werden.'),
                    'base64_data': base64_data,
                    'content_type': get_content_type_from_extension(img_data['file_extension'])
                }
                
    finally:
        # Aufräumen
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                log.warning(f"Failed to delete temp file {temp_file}: {str(e)}")
        
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            log.warning(f"Failed to delete temp directory {temp_dir}: {str(e)}")
            
    return image_info


def generate_ai_descriptions_batch(client, images_data, markdown_context, custom_prompt="", model="gpt-4.1-mini"):
    """
    Generiert AI-Beschreibungen für mehrere Bilder in einem Batch-Call.
    """
    if not images_data:
        return {}
    
    # Default-Prompt falls keiner angegeben
    if not custom_prompt:
        custom_prompt = """You will receive email content in Markdown format and must generate accurate English descriptions for the identified images.

For each image, provide:
1. **Alt text**: A short, concise description (maximum 10 words)
2. **Description**: A detailed description (1–2 sentences) emphasizing relevant details for the email context

Reply in JSON format:
{
  "0": {"alt_text": "...", "description": "..."},
  "1": {"alt_text": "...", "description": "..."}
}
"""
    
    # Erstelle Markdown-Kontext mit Bild-Markern
    enhanced_context = markdown_context + "\n\n" if markdown_context else ""
    
    # Füge Bild-Marker hinzu
    for i, img_data in enumerate(images_data):
        enhanced_context += f"\n\n<<BILD {i}: {img_data['alt_text']} - DU SOLLST FÜR DIESES BILD EINE BESCHREIBUNG ERSTELLEN>>\n"
    
    try:
        # Erstelle Content-Array mit Text und allen Bildern
        content = [
            {"type": "text", "text": f"{custom_prompt}\n\nE-Mail-Kontext:\n{enhanced_context}"}
        ]
        
        # Füge alle Bilder hinzu
        for i, img_data in enumerate(images_data):
            with open(img_data['temp_path'], 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            content_type = get_content_type_from_extension(img_data['file_extension'])
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{content_type};base64,{image_data}",
                    "detail": "low"  # Kosteneinsparung
                }
            })
        
        # API-Call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=500 * len(images_data),  # Mehr Tokens für mehrere Bilder
            response_format={"type": "json_object"}  # JSON-Response erzwingen
        )
        
        # Parse JSON-Response
        import json
        result_text = response.choices[0].message.content.strip()
        ai_results = json.loads(result_text)
        
        log.info(f"Generated AI descriptions for {len(images_data)} images")
        return ai_results
        
    except Exception as e:
        log.error(f"Error in batch AI description generation: {str(e)}")
        
        # Fallback: Einzelverarbeitung
        results = {}
        for i, img_data in enumerate(images_data):
            try:
                description = generate_ai_description_single(client, img_data['temp_path'], img_data['alt_text'], model)
                results[str(i)] = {
                    'alt_text': img_data['alt_text'] or f"Bild {i+1}",
                    'description': description
                }
            except Exception as e2:
                log.error(f"Error in fallback description for image {i}: {str(e2)}")
                results[str(i)] = {
                    'alt_text': img_data['alt_text'] or f"Bild {i+1}",
                    'description': "Bildbeschreibung konnte nicht generiert werden."
                }
        
        return results


def generate_ai_description_single(client, image_path, alt_text, model="gpt-4.1-mini"):
    """Fallback: Generiert eine AI-Beschreibung für ein einzelnes Bild."""
    try:
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
        prompt = f"""Beschreibe dieses Bild in 1-2 präzisen deutschen Sätzen. 
        Konzentriere dich auf die wichtigsten visuellen Elemente und den Kontext.
        Falls verfügbar, berücksichtige den Alt-Text: "{alt_text}"
        
        Antworte nur mit der Beschreibung, ohne zusätzliche Formatierung."""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        log.error(f"Error generating single AI description: {str(e)}")
        return "Bildbeschreibung konnte nicht generiert werden."


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


def process_images_for_ai_description(soup, openai_api_key, base_url, image_path, image_prefix, markdown_context="", ai_prompt="", ai_model="gpt-4.1-mini"):
    """
    Verarbeitet alle Bilder im HTML mit Batching für AI-Beschreibungen:
    1. Lädt Bilder temporär herunter
    2. Generiert AI-Beschreibungen mit OpenAI (Batching)
    3. Kodiert Bilder als Base64 für Response
    4. Löscht temporäre Dateien
    
    Args:
        soup: BeautifulSoup object der bereinigten HTML
        openai_api_key: OpenAI API Schlüssel
        base_url: Basis-URL für relative Bildpfade
        image_path: Pfad-Präfix für Bilder
        image_prefix: Präfix für Dateinamen
        markdown_context: Markdown-Kontext des E-Mail-Inhalts
        ai_prompt: Benutzerdefinierter AI-Prompt (optional)
        ai_model: AI-Modell (default: gpt-4.1-mini)
    """
    image_info = {}
    temp_files = []
    
    # Erstelle temporäres Bildverzeichnis
    temp_dir = tempfile.mkdtemp(prefix="talon_images_")
    
    # OpenAI Client initialisieren
    try:
        client = openai.OpenAI(api_key=openai_api_key)
    except Exception as e:
        log.error(f"Failed to initialize OpenAI client: {str(e)}")
        return image_info
      # Alle img-Tags finden
    img_tags = soup.find_all('img')
    
    if not img_tags:
        return image_info
    
    # Sammle alle Bilder zum parallelen Download
    download_tasks = []
    
    try:
        for idx, img_tag in enumerate(img_tags, 1):
            try:
                src = img_tag.get('src')
                alt_text = img_tag.get('alt', '')
                
                if not src:
                    continue
                    
                # Absolute URL erstellen
                if base_url and not src.startswith(('http://', 'https://', 'data:')):
                    full_url = urljoin(base_url, src)
                else:
                    full_url = src
                    
                # Skip data URLs (base64 embedded images) for now
                if src.startswith('data:'):
                    continue
                    
                # Temporären Bildnamen generieren
                url_hash = hashlib.md5(full_url.encode()).hexdigest()[:8]
                file_extension = get_file_extension_from_url(full_url)
                filename = f"{image_prefix}{url_hash}{file_extension}" if image_prefix else f"image_{idx}_{url_hash}{file_extension}"
                temp_path = os.path.join(temp_dir, filename)
                local_path = os.path.join(image_path, filename).replace('\\', '/')
                
                # Sammle Download-Task
                download_tasks.append({
                    'url': full_url,
                    'path': temp_path,
                    'src': src,
                    'alt_text': alt_text,
                    'filename': filename,
                    'local_path': local_path,
                    'file_extension': file_extension
                })
                    
            except Exception as e:
                log.error(f"Error preparing image {src}: {str(e)}")
        
        # Paralleler Download aller Bilder
        images_to_process = []
        if download_tasks:
            log.info(f"Starting parallel download of {len(download_tasks)} images...")
            download_results = download_images_parallel(download_tasks, max_workers=4)
            
            # Verarbeite erfolgreiche Downloads
            for result in download_results:
                if result['success']:
                    temp_files.append(result['path'])
                    images_to_process.append({
                        'src': result['src'],
                        'alt_text': result['alt_text'],
                        'full_url': result['url'],
                        'filename': result['filename'],
                        'temp_path': result['path'],
                        'local_path': result['local_path'],
                        'file_extension': result['file_extension']
                    })
        
        # Batch-Verarbeitung für AI-Beschreibungen
        if images_to_process:
            ai_descriptions = generate_ai_descriptions_batch(
                client, images_to_process, markdown_context, ai_prompt, ai_model
            )
            
            # Kombiniere Ergebnisse
            for i, img_data in enumerate(images_to_process):
                ai_desc = ai_descriptions.get(str(i), {})
                
                # Bild als Base64 kodieren
                with open(img_data['temp_path'], 'rb') as img_file:
                    base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                  # Informationen speichern
                image_info[img_data['src']] = {
                    'original_src': img_data['src'],
                    'full_url': img_data['full_url'],
                    'local_path': img_data['local_path'],
                    'filename': img_data['filename'],
                    'alt_text': ai_desc.get('alt_text', img_data['alt_text']),
                    'ai_description': ai_desc.get('description', 'Bildbeschreibung konnte nicht generiert werden.'),
                    'base64_data': base64_data,
                    'content_type': get_content_type_from_extension(img_data['file_extension'])
                }
                
    finally:
        # Temporäre Dateien löschen
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                log.warning(f"Failed to delete temp file {temp_file}: {str(e)}")
        
        # Temporäres Verzeichnis löschen
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            log.warning(f"Failed to delete temp directory {temp_dir}: {str(e)}")
            
    return image_info


def generate_ai_descriptions_batch(client, images_data, markdown_context, custom_prompt="", model="gpt-4.1-mini"):
    """
    Generiert AI-Beschreibungen für mehrere Bilder in einem Batch-Call.
    """
    if not images_data:
        return {}
    
    # Default-Prompt falls keiner angegeben
    if not custom_prompt:
        custom_prompt = """You will receive email content in Markdown format and must generate accurate German descriptions for the identified images.

For each image, provide:
1. **Alt text**: A short, concise description (maximum 10 words)
2. **Description**: A detailed description (1–2 sentences) emphasizing relevant details for the email context

Reply in JSON format:
{
  "0": {"alt_text": "...", "description": "..."},
  "1": {"alt_text": "...", "description": "..."}
}"""
    
    # Erstelle Markdown-Kontext mit Bild-Markern
    enhanced_context = markdown_context + "\n\n" if markdown_context else ""
    
    # Füge Bild-Marker hinzu
    for i, img_data in enumerate(images_data):
        enhanced_context += f"\n\n<<BILD {i}: {img_data['alt_text']} - DU SOLLST FÜR DIESES BILD EINE BESCHREIBUNG ERSTELLEN>>\n"
    
    try:
        # Erstelle Content-Array mit Text und allen Bildern
        content = [
            {"type": "text", "text": f"{custom_prompt}\n\nE-Mail-Kontext:\n{enhanced_context}"}
        ]
        
        # Füge alle Bilder hinzu
        for i, img_data in enumerate(images_data):
            with open(img_data['temp_path'], 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            content_type = get_content_type_from_extension(img_data['file_extension'])
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{content_type};base64,{image_data}",
                    "detail": "low"  # Kosteneinsparung
                }
            })
        
        # API-Call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=500 * len(images_data),  # Mehr Tokens für mehrere Bilder
            response_format={"type": "json_object"}  # JSON-Response erzwingen
        )
        
        # Parse JSON-Response
        import json
        result_text = response.choices[0].message.content.strip()
        ai_results = json.loads(result_text)
        
        log.info(f"Generated AI descriptions for {len(images_data)} images")
        return ai_results
        
    except Exception as e:
        log.error(f"Error in batch AI description generation: {str(e)}")
        
        # Fallback: Einzelverarbeitung
        results = {}
        for i, img_data in enumerate(images_data):
            try:
                description = generate_ai_description_single(client, img_data['temp_path'], img_data['alt_text'], model)
                results[str(i)] = {
                    'alt_text': img_data['alt_text'] or f"Bild {i+1}",
                    'description': description
                }
            except Exception as e2:
                log.error(f"Error in fallback description for image {i}: {str(e2)}")
                results[str(i)] = {
                    'alt_text': img_data['alt_text'] or f"Bild {i+1}",
                    'description': "Bildbeschreibung konnte nicht generiert werden."
                }
        
        return results


def generate_ai_description_single(client, image_path, alt_text, model="gpt-4.1-mini"):
    """Fallback: Generiert eine AI-Beschreibung für ein einzelnes Bild."""
    try:
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
        prompt = f"""Beschreibe dieses Bild in 1-2 präzisen deutschen Sätzen. 
        Konzentriere dich auf die wichtigsten visuellen Elemente und den Kontext.
        Falls verfügbar, berücksichtige den Alt-Text: "{alt_text}"
        
        Antworte nur mit der Beschreibung, ohne zusätzliche Formatierung."""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],        max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        log.error(f"Error generating single AI description: {str(e)}")
        return "Bildbeschreibung konnte nicht generiert werden."


def download_image(url, local_path):
    """Lädt ein Bild von einer URL herunter und speichert es lokal."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        
        # Prüfe Content-Type
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            log.warning(f"URL does not return an image: {url} (Content-Type: {content_type})")
            return False
            
        # Speichere das Bild
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return True
        
    except Exception as e:
        log.error(f"Error downloading image {url}: {str(e)}")
        return False


def download_image_parallel(image_data):
    """
    Wrapper-Funktion für parallelen Download eines einzelnen Bildes.
    Erwartet ein Dictionary mit 'url', 'path' und optional anderen Metadaten.
    """
    url = image_data['url']
    path = image_data['path']
    
    success = download_image(url, path)
    return {
        **image_data,
        'success': success
    }


def download_images_parallel(images_data, max_workers=4):
    """
    Lädt mehrere Bilder parallel herunter.
    
    Args:
        images_data: Liste von Dictionaries mit 'url' und 'path' Schlüsseln
        max_workers: Maximale Anzahl paralleler Downloads (default: 4)
    
    Returns:
        Liste von Dictionaries mit Ergebnissen (success=True/False)
    """
    if not images_data:
        return []
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Starte alle Download-Tasks
        future_to_data = {
            executor.submit(download_image_parallel, img_data): img_data 
            for img_data in images_data
        }
        
        # Sammle Ergebnisse sobald sie verfügbar sind
        for future in as_completed(future_to_data):
            try:
                result = future.result()
                results.append(result)
                
                if result['success']:
                    log.info(f"✓ Downloaded: {result['url']} -> {result['path']}")
                else:
                    log.warning(f"✗ Failed: {result['url']}")
                    
            except Exception as e:
                img_data = future_to_data[future]
                log.error(f"Exception downloading {img_data['url']}: {str(e)}")
                results.append({
                    **img_data,
                    'success': False,
                    'error': str(e)
                })
    
    successful_downloads = sum(1 for r in results if r['success'])
    log.info(f"Parallel download completed: {successful_downloads}/{len(results)} successful")
    
    return results


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

> **Bildbeschreibung (KI):**
> {info['ai_description']}"""
            
            return enhanced_markdown
        else:
            # Fallback: Originales Markdown-Bild
            return match.group(0)
    
    # Ersetze alle Bilder
    enhanced_markdown = re.sub(image_pattern, replace_image, markdown)
    
    return enhanced_markdown


def get_content_type_from_extension(extension):
    """Ermittelt den Content-Type basierend auf der Dateierweiterung."""
    content_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.svg': 'image/svg+xml'
    }
    return content_types.get(extension.lower(), 'image/jpeg')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5505)