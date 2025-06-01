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

# Global OpenAI client cache to avoid recreating clients repeatedly
_openai_clients = {}

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
        ai_signature_extraction = data.get('ai_signature_extraction', False)
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
        ai_signature_extraction = request.form.get('ai_signature_extraction', 'false').lower() == 'true'
        
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

        if openai_api_key and ai_signature_extraction:
            # 7. AI-gestützte Grußformel-Erkennung und -Extraktion
            final_markdown = extract_content_until_salutation_with_ai(markdown, openai_api_key, ai_model)
            log.info("AI signature extraction completed.")
        else:
            # 7. Grußformel + Name aus Markdown beibehalten, danach abschneiden
            final_markdown = extract_content_until_salutation(markdown)
          # 8. AI-Bildverarbeitung - NUR wenn OpenAI Key vorhanden und NACH der Signaturentfernung
        image_info = {}
        if openai_api_key and final_markdown:
            image_info = process_images_parallel_with_ai(
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

    try:
        response = client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": "You are an assistant specialized in email content processing."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.1
        )
        cleaned_markdown = response.choices[0].message.content.strip()
        return cleaned_markdown

    except Exception as e:
        log.error(f"Error during AI processing: {e}")
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
        
        # Reduziertes Timeout für bessere Performance
        response = requests.get(url, headers=headers, timeout=8, stream=True)
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
        
        log.debug(f"Image downloaded: {url}")
        return True
        
    except Exception as e:
        log.error(f"Error downloading image {url}: {str(e)}")
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
            log.error(f"Failed to download image: {task_data['url']}")
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


def generate_ai_description_with_marker(client, image_path, alt_text, markdown_context, custom_prompt="", model="gpt-4.1-mini"):
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


def process_images_parallel_with_ai(markdown, openai_api_key, base_url, auth_query_params, image_path, image_prefix, ai_prompt="", ai_model="gpt-4.1-mini", max_workers=6):
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
        auth_query_params: Authentication query parameters for images (optional)
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
            
    finally:
        # Temporäres Verzeichnis aufräumen
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            log.warning(f"Failed to delete temp directory {temp_dir}: {str(e)}")
            
    return image_info

def process_images_from_markdown(markdown, openai_api_key, base_url=None, auth_query_params=None, image_path="./images/", image_prefix="", ai_prompt="", ai_model="gpt-4.1-mini"):
    """
    Main wrapper function that calls the new parallel processing implementation.
    This maintains compatibility with existing code while using the new parallel approach.
    """
    return process_images_parallel_with_ai(
        markdown=markdown,
        openai_api_key=openai_api_key,
        base_url=base_url,
        auth_query_params=auth_query_params,
        image_path=image_path,
        image_prefix=image_prefix,
        ai_prompt=ai_prompt,
        ai_model=ai_model,
        max_workers=6  # Increased for better performance
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Talon Web Bootstrap Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5505, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    log.info(f"Starting Talon Web Bootstrap Server on {args.host}:{args.port}")
    log.info(f"Debug mode: {args.debug}")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False  # Disable reloader to avoid duplicate initialization
    )