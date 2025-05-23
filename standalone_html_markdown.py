#!/usr/bin/env python3
"""
Standalone HTML to Markdown converter with signature removal
This version works independently of Talon for testing purposes
"""

from flask import Flask, request, jsonify, json
from werkzeug.exceptions import HTTPException, BadRequest
import html2text
import re
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

app = Flask(__name__)

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    response = e.get_response()
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
    response = e.get_response()
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

@app.route('/html-to-markdown', methods=['POST'])
def html_to_markdown():
    """
    Converts HTML to Markdown with signature removal
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
        
        log.debug('HTML content received: ' + str(len(html_content)) + ' characters')
        
        # Remove signature patterns from HTML
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
            'cleaned_html': cleaned_html,
            'markdown': markdown_content,
            'success': True
        }
        
        return jsonify(response_data)
        
    except BadRequest:
        raise
    except Exception as e:
        log.error(f"Error in html_to_markdown: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

def _remove_html_signature_patterns(html_content):
    """
    Remove common email signature patterns from HTML
    """
    # Common signature separators and patterns
    signature_patterns = [
        # Horizontal line signatures - remove everything after HR
        r'<hr[^>]*>.*$',
        # Signature divs with class
        r'<div[^>]*class[^>]*signature[^>]*>.*?</div>',
        r'<div[^>]*class[^>]*sig[^>]*>.*?</div>',
        # Double dash separator - remove everything after --
        r'--\s*<br[^>]*>.*$',
        r'<p[^>]*>--\s*</p>.*$',
        # Common German signatures - remove everything after these phrases
        r'<div[^>]*>Mit freundlichen Grüßen.*$',
        r'<p[^>]*>Mit freundlichen Grüßen.*$',
        r'<div[^>]*>Freundliche Grüße.*$',
        r'<p[^>]*>Freundliche Grüße.*$',
        r'<div[^>]*>Viele Grüße.*$',
        r'<p[^>]*>Viele Grüße.*$',
        # Common English signatures - remove everything after these phrases
        r'<div[^>]*>Best regards.*$',
        r'<p[^>]*>Best regards.*$',
        r'<div[^>]*>Kind regards.*$',
        r'<p[^>]*>Kind regards.*$',
        r'<div[^>]*>Sincerely.*$',
        r'<p[^>]*>Sincerely.*$',
        r'<div[^>]*>Thanks.*$',
        r'<p[^>]*>Thanks.*$',
        # Gmail/Outlook signature blocks
        r'<div[^>]*gmail_signature[^>]*>.*?</div>',
        r'<div[^>]*id[^>]*signature[^>]*>.*?</div>',
    ]
    
    cleaned_html = html_content
    
    for pattern in signature_patterns:
        cleaned_html = re.sub(pattern, '', cleaned_html, flags=re.DOTALL | re.IGNORECASE)
    
    return cleaned_html

@app.route('/test', methods=['GET'])
def test():
    """Simple test endpoint"""
    return jsonify({'message': 'HTML to Markdown API is running!', 'success': True})

if __name__ == '__main__':
    print("Starting HTML to Markdown API...")
    print("Available endpoints:")
    print("  POST /html-to-markdown - Convert HTML to Markdown")
    print("  GET  /test - Test endpoint")
    app.run(debug=True, host='0.0.0.0', port=5001)
