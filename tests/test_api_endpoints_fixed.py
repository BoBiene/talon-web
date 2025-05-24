# filepath: c:\GIT\talon-web\tests\test_api_endpoints_fixed.py
"""
Consolidated API endpoints tests
Tests all API endpoints with various content types and edge cases.
"""
import pytest
import json
from tests.fixtures.html_samples import (
    PROFESSIONAL_SIGNATURES,
    SIMPLE_SIGNATURES,
    MIXED_CONTENT_SCENARIOS
)


class TestAPIEndpoints:
    """Test all API endpoints with comprehensive scenarios."""

    def test_health_endpoint(self, test_client):
        """Test the health check endpoint."""
        response = test_client.get('/health')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        assert 'status' in result
        assert result['status'] == 'healthy'
        assert 'service' in result
        assert 'endpoints' in result

    def test_quotation_extraction_html_endpoint(self, test_client):
        """Test the /talon/quotations/html endpoint."""
        # Test with HTML content containing professional signature
        html_content = '''
            <div>Hello,</div>
            <div>Please review the attached documents.</div>
            <div>Best regards<br>John Doe<br>Senior Engineer<br>john@company.com</div>
            '''
        
        response = test_client.post('/talon/quotations/html', 
                                  data={'email_content': html_content,
                                        'email_sender': 'john@company.com'})
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        # Should contain the extracted content
        assert 'email_reply' in result
        assert 'Hello' in result['email_reply']
        assert 'review the attached' in result['email_reply']

    def test_quotation_extraction_text_endpoint(self, test_client):
        """Test the /talon/quotations/text endpoint."""
        text_content = '''Hello,

Please check the server logs.

Best regards
John Doe
Senior Engineer
john@company.com'''
        
        response = test_client.post('/talon/quotations/text',
                                  data={'email_content': text_content,
                                        'email_sender': 'john@company.com'})
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        # Should extract content properly
        assert 'email_reply' in result
        assert 'check the server logs' in result['email_reply']

    def test_html_markdown_endpoint(self, test_client):
        """Test the /talon/html-to-markdown endpoint."""
        test_data = {
            'html': '<div>Hello Team,</div><div>Please review.</div><div class="gmail_signature">John Doe<br>john@company.com</div>',
            'sender': 'john@company.com'
        }
        
        response = test_client.post('/talon/html-to-markdown',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        # Should contain markdown conversion
        assert 'markdown' in result
        assert 'Hello Team' in result['markdown']
        assert 'Please review' in result['markdown']

    def test_signature_extraction_endpoint(self, test_client):
        """Test the /talon/signature endpoint."""
        email_content = '''Main content

Best regards,
John Doe
john@company.com'''
        
        response = test_client.post('/talon/signature',
                                  data={'email_content': email_content,
                                        'email_sender': 'john@company.com'})
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'email_body' in result
        assert 'email_signature' in result

    def test_api_with_professional_signatures(self, test_client):
        """Test API endpoints with professional signature samples."""
        for test_case in PROFESSIONAL_SIGNATURES[:3]:  # Test first 3 cases
            html_content = f'<div>Important message content.</div>{test_case["html"]}'
            
            response = test_client.post('/talon/quotations/html',
                                      data={'email_content': html_content,
                                            'email_sender': 'test@example.com'})
            
            assert response.status_code == 200
            result = json.loads(response.data)
            
            # Main content should be preserved
            assert 'Important message content' in result['email_reply']

    def test_api_error_handling(self, test_client):
        """Test API error handling for missing parameters."""
        # Test missing email_content for signature endpoint
        response = test_client.post('/talon/signature',
                                  data={'email_sender': 'test@example.com'})
        
        assert response.status_code == 400

        # Test missing email_content for quotations/text endpoint
        response = test_client.post('/talon/quotations/text',
                                  data={'email_sender': 'test@example.com'})
        
        assert response.status_code == 400

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
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        assert 'email_reply' in result


class TestAPIPerformance:
    """Test API performance with larger content."""

    def test_large_html_content(self, test_client):
        """Test API with large HTML content."""
        large_content = '<div>Content line</div>' * 100 + '<div>Best regards<br>John</div>'
        
        response = test_client.post('/talon/quotations/html',
                                  data={'email_content': large_content,
                                        'email_sender': 'john@example.com'})
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'email_reply' in result

    def test_multiple_signatures(self, test_client):
        """Test content with multiple potential signatures."""
        content = '''Main content here.

Best regards,
First Person

---

Thanks,
Second Person
second@example.com'''
        
        response = test_client.post('/talon/quotations/text',
                                  data={'email_content': content,
                                        'email_sender': 'first@example.com'})
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'email_reply' in result


class TestAPIContentTypes:
    """Test API with different content types."""

    def test_form_data_submission(self, test_client):
        """Test form-data submission (standard for the API)."""
        response = test_client.post('/talon/signature',
                                  data={'email_content': 'Test content\n\nBest,\nJohn',
                                        'email_sender': 'john@example.com'})
        
        assert response.status_code == 200

    def test_json_submission_html_markdown(self, test_client):
        """Test JSON submission for HTML-to-markdown endpoint."""
        test_data = {
            'html': '<p>Test content</p>',
            'sender': 'test@example.com'
        }
        
        response = test_client.post('/talon/html-to-markdown',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        
        assert response.status_code == 200


class TestAPIEdgeCases:
    """Test API edge cases and boundary conditions."""

    def test_malformed_html(self, test_client):
        """Test API with malformed HTML."""
        malformed_html = '<div>Test <p>Unclosed tags <br> content</div>'
        
        response = test_client.post('/talon/quotations/html',
                                  data={'email_content': malformed_html,
                                        'email_sender': 'test@example.com'})
        
        # Should handle gracefully
        assert response.status_code == 200

    def test_very_long_lines(self, test_client):
        """Test API with very long lines."""
        long_line = 'A' * 1000
        content = f'Normal content\n\n{long_line}\n\nBest regards\nJohn'
        
        response = test_client.post('/talon/quotations/text',
                                  data={'email_content': content,
                                        'email_sender': 'john@example.com'})
        
        assert response.status_code == 200

    def test_special_characters(self, test_client):
        """Test API with special characters."""
        special_content = '''Content with special chars: !@#$%^&*()

Best regards,
User™
user@example.com'''
        
        response = test_client.post('/talon/signature',
                                  data={'email_content': special_content,
                                        'email_sender': 'user@example.com'})
        
        assert response.status_code == 200
