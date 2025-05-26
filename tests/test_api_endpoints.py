"""
Test API endpoints functionality.
"""
import json
import pytest
from .fixtures.html_samples import (
    PROFESSIONAL_SIGNATURES,
    SIMPLE_SIGNATURES,
    MIXED_CONTENT_SCENARIOS
)


class TestApiEndpoints:
    """Test API endpoint functionality."""

    def test_signature_extraction_endpoint(self, test_client):
        """Test signature extraction endpoint."""
        test_content = "Hello,\n\nThis is a test email.\n\nBest regards,\nJohn Doe\nSenior Developer"
        
        response = test_client.post('/talon/signature',
                                  data={'email_content': test_content,
                                        'email_sender': 'john@example.com'})
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'email_signature' in result

    def test_text_quotation_extraction_endpoint(self, test_client):
        """Test text quotation extraction endpoint."""
        test_content = "Reply text\n\nOn Date, Person wrote:\n> Original message"
        
        response = test_client.post('/talon/quotations/text',
                                  data={'email_content': test_content,
                                        'email_sender': 'test@example.com'})
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'email_reply' in result

    def test_html_quotation_extraction_endpoint(self, test_client):
        """Test HTML quotation extraction endpoint."""
        test_content = '<p>Reply text</p><blockquote>Original message</blockquote>'
        
        response = test_client.post('/talon/quotations/html',
                                  data={'email_content': test_content,
                                        'email_sender': 'test@example.com'})
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'email_reply' in result

    def test_html_to_markdown_endpoint(self, test_client):
        """Test HTML to markdown conversion endpoint."""
        test_content = '<h1>Title</h1><p>Paragraph with <strong>bold</strong> text.</p>'
        
        response = test_client.post('/talon/html-to-markdown',
                                  data={'html': test_content})
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'markdown' in result

    def test_api_error_handling(self, test_client):
        """Test API error handling."""
        # Test missing content
        response = test_client.post('/talon/signature', data={})
        assert response.status_code in [400, 422]  # Client error expected

    def test_api_content_type_handling(self, test_client):
        """Test API handles different content types."""
        test_content = "Hello,\n\nTest content.\n\nBest regards,\nTest User"
        
        # Test with form data
        response = test_client.post('/talon/signature',
                                  data={'email_content': test_content,
                                        'email_sender': 'test@example.com'})
        assert response.status_code == 200

    def test_api_unicode_support(self, test_client):
        """Test API Unicode support."""
        unicode_content = '''Hallo,

Bitte überprüfen Sie die Dokumente.

Mit freundlichen Grüßen
Müller
müller@firma.de'''
        
        response = test_client.post('/talon/quotations/text',
                                  data={'email_content': unicode_content,
                                        'email_sender': 'müller@firma.de'})
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'email_reply' in result
        assert 'überprüfen' in result['email_reply']

    def test_api_empty_content(self, test_client):
        """Test API with empty content."""
        response = test_client.post('/talon/quotations/text',
                                  data={'email_content': '',
                                        'email_sender': 'test@example.com'})
        
        assert response.status_code == 400  # Empty content should return 400

    def test_api_performance(self, test_client):
        """Test API performance with larger content."""
        large_content = "Test content " * 100  # Repeat content
        
        response = test_client.post('/talon/signature',
                                  data={'email_content': large_content,
                                        'email_sender': 'test@example.com'})
        
        assert response.status_code == 200
        # Performance test - should complete within reasonable time
