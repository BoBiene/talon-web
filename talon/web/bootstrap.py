from talon import signature, quotations
from flask import Flask, request, jsonify, json
from werkzeug.exceptions import HTTPException, BadRequest
import talon
import logging
import html2text
import re

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
        
        # First extract text from HTML using Talon's existing functionality
        # This will remove quotations/replies
        clean_text = quotations.extract_from_html(html_content)
        log.debug('Text after quotation removal: ' + clean_text)
        
        # Then remove signature if sender is provided
        signature_text = None
        if sender:
            clean_text, signature_text = signature.extract(clean_text, sender=sender)
            log.debug('Text after signature removal: ' + clean_text)
            log.debug('Extracted signature: ' + str(signature_text))
        
        # Configure html2text for optimal markdown conversion
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_emphasis = False
        h.body_width = 0  # Don't wrap lines
        h.unicode_snob = True
        h.use_automatic_links = True
        h.protect_links = True
        
        # Convert the cleaned text back to HTML temporarily for proper markdown conversion
        # Since Talon returns plain text, we need to handle it appropriately
        if clean_text:
            # If we have plain text, convert it to simple HTML first
            simple_html = clean_text.replace('\n', '<br>\n')
            markdown_content = h.handle(simple_html)
        else:
            markdown_content = ""
        
        # Clean up extra whitespace and normalize line breaks
        markdown_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', markdown_content)
        markdown_content = markdown_content.strip()
        
        # Prepare response
        response_data = {
            'original_html': html_content,
            'markdown': markdown_content,
            'removed_signature': str(signature_text) if signature_text else None,
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


def _remove_html_signature_patterns(html_content):
    """
    Remove common email signature patterns from HTML
    """
    # Common signature separators and patterns
    signature_patterns = [
        # Horizontal line signatures
        r'<hr[^>]*>.*$',
        # Signature divs with class
        r'<div[^>]*class[^>]*signature[^>]*>.*?</div>',
        r'<div[^>]*class[^>]*sig[^>]*>.*?</div>',
        # Double dash separator
        r'--\s*<br[^>]*>.*$',
        r'<p[^>]*>--\s*</p>.*$',
        # Common German signatures
        r'<div[^>]*>Mit freundlichen Grüßen.*?</div>.*$',
        r'<p[^>]*>Mit freundlichen Grüßen.*$',
        r'<div[^>]*>Freundliche Grüße.*?</div>.*$',
        r'<p[^>]*>Freundliche Grüße.*$',
        r'<div[^>]*>Viele Grüße.*?</div>.*$',
        r'<p[^>]*>Viele Grüße.*$',
        # Common English signatures
        r'<div[^>]*>Best regards.*?</div>.*$',
        r'<p[^>]*>Best regards.*$',
        r'<div[^>]*>Kind regards.*?</div>.*$',
        r'<p[^>]*>Kind regards.*$',
        r'<div[^>]*>Sincerely.*?</div>.*$',
        r'<p[^>]*>Sincerely.*$',
        r'<div[^>]*>Thanks.*?</div>.*$',
        r'<p[^>]*>Thanks.*$',
        # Gmail/Outlook signature blocks
        r'<div[^>]*gmail_signature[^>]*>.*?</div>',
        r'<div[^>]*id[^>]*signature[^>]*>.*?</div>',
    ]
    
    cleaned_html = html_content
    
    for pattern in signature_patterns:
        cleaned_html = re.sub(pattern, '', cleaned_html, flags=re.DOTALL | re.IGNORECASE)
    
    return cleaned_html


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
