"""
Consolidated integration tests
Tests the complete workflow and integration between components.
"""
import pytest
import json
from talon import signature, quotations
from talon.quotations import extract_from_html
from tests.fixtures.html_samples import MIXED_CONTENT_SCENARIOS, PROFESSIONAL_SIGNATURES


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_full_email_processing_workflow(self, test_client):
        """Test complete email processing from API to output."""
        # Real-world email example
        email_content = '''
        <html>
        <body>
            <div>Hi Team,</div>
            <div><br></div>
            <div>I've completed the code review and have the following feedback:</div>
            <div><br></div>
            <ul>
                <li>The new authentication logic looks good</li>
                <li>Please add unit tests for the edge cases</li>
                <li>Consider adding error handling for network timeouts</li>
            </ul>
            <div><br></div>
            <div>Let me know if you have any questions.</div>
            <div><br></div>
            <hr>
            <div>Best regards</div>
            <div>John Doe<br>
            Senior Software Engineer<br>
            Engineering Team<br>
            john.doe@company.com<br>
            Mobile: +1 (555) 123-4567</div>
        </body>
        </html>
        '''
        
        # Test API processing
        test_data = {
            'body': email_content,
            'content_type': 'text/html'
        }
        
        response = test_client.post('/quotation_extraction',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        # Should preserve main email content
        assert 'code review' in result['reply']
        assert 'authentication logic' in result['reply']
        assert 'unit tests' in result['reply']
        assert 'error handling' in result['reply']
        assert 'questions' in result['reply']
        
        # Should remove professional signature
        assert 'john.doe@company.com' not in result['reply']
        assert 'Senior Software Engineer' not in result['reply']
        assert '+1 (555) 123-4567' not in result['reply']

    def test_html_to_markdown_integration(self, test_client):
        """Test HTML to Markdown conversion integration."""
        html_content = '''
        <div>
            <h2>Meeting Notes</h2>
            <p>Please review the <strong>action items</strong> from today's meeting:</p>
            <ol>
                <li>Update the <a href="https://docs.example.com">documentation</a></li>
                <li>Schedule follow-up meeting</li>
                <li>Send summary to stakeholders</li>
            </ol>
            <blockquote>
                <p>Note: All items should be completed by Friday.</p>
            </blockquote>
        </div>
        <div class="gmail_signature">
            <p>Best regards<br>
            Project Manager<br>
            pm@company.com</p>
        </div>
        '''
        
        # Test HTML to Markdown conversion
        test_data = {
            'html': html_content,
            'content_type': 'text/html'
        }
        
        response = test_client.post('/html_markdown',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        # Should convert to markdown and preserve content
        markdown = result['markdown']
        assert 'Meeting Notes' in markdown
        assert 'action items' in markdown
        assert 'documentation' in markdown
        assert 'follow-up meeting' in markdown
        
        # Professional signature should be removed
        assert 'pm@company.com' not in markdown

    def test_quotation_and_signature_integration(self):
        """Test integration between quotation and signature extraction."""
        # Email with both quotation and signature
        email_with_quote = '''
        <div>Thanks for the update!</div>
        <div><br></div>
        <div>I'll review the proposal and get back to you by tomorrow.</div>
        <div><br></div>
        <div class="gmail_quote">
            <div>On Mon, Jan 1, 2024 at 10:00 AM, sender@example.com wrote:</div>
            <blockquote>
                <div>Hi,</div>
                <div>Please find the attached proposal for your review.</div>
                <div>Let me know your thoughts.</div>
                <div><br></div>
                <div>Best regards<br>
                Original Sender<br>
                sender@example.com</div>
            </blockquote>
        </div>
        <div><br></div>
        <div>Best regards</div>
        <div>John Doe<br>
        Senior Manager<br>
        john@company.com</div>
        '''
        
        # Process through both quotation and signature extraction
        # First remove quotations
        without_quotes = quotations.extract_from_html(email_with_quote)
        
        # Then remove signatures
        final_result = signature.extract(without_quotes, "text/html")
        
        # Should preserve main response content
        assert 'Thanks for the update' in final_result
        assert 'review the proposal' in final_result
        assert 'get back to you' in final_result
        
        # Should remove quoted content
        assert 'attached proposal' not in final_result
        assert 'sender@example.com' not in final_result
        
        # Should remove signature
        assert 'john@company.com' not in final_result
        assert 'Senior Manager' not in final_result

    def test_complex_nested_content_processing(self):
        """Test processing of complex nested email structures."""
        complex_email = '''
        <div class="email-container">
            <div class="header">
                <h3>Project Status Update</h3>
            </div>
            <div class="content">
                <div class="section">
                    <h4>Completed Tasks</h4>
                    <table>
                        <tr>
                            <td>Database Migration</td>
                            <td>✓ Complete</td>
                        </tr>
                        <tr>
                            <td>API Testing</td>
                            <td>✓ Complete</td>
                        </tr>
                    </table>
                </div>
                <div class="section">
                    <h4>Next Steps</h4>
                    <ol>
                        <li>Deploy to staging environment</li>
                        <li>User acceptance testing</li>
                        <li>Production deployment</li>
                    </ol>
                </div>
                <div class="notes">
                    <p><em>Note: Please review the staging environment before UAT.</em></p>
                </div>
            </div>
            <div class="signature">
                <div>Best regards</div>
                <div>Development Team Lead<br>
                lead@company.com<br>
                Company Name</div>
            </div>
        </div>
        '''
        
        result = signature.extract(complex_email, "text/html")
        
        # Should preserve all main content
        assert 'Project Status Update' in result
        assert 'Completed Tasks' in result
        assert 'Database Migration' in result
        assert 'Next Steps' in result
        assert 'staging environment' in result
        assert 'UAT' in result
        
        # Should remove signature
        assert 'lead@company.com' not in result
        assert 'Development Team Lead' not in result


class TestComponentIntegration:
    """Test integration between different components."""

    def test_signature_extraction_with_html_parsing(self):
        """Test signature extraction integrates properly with HTML parsing."""
        for test_case in PROFESSIONAL_SIGNATURES:
            # Process through HTML parsing first
            parsed_html = extract_from_html(test_case['html'])
            
            # Then through signature extraction
            final_result = signature.extract(parsed_html, "text/html")
            
            # Professional elements should be removed
            for element in test_case['should_remove']:
                assert element not in final_result, f"Element '{element}' should be removed after full processing"

    def test_api_integration_with_mixed_scenarios(self, test_client):
        """Test API integration with mixed content scenarios."""
        for test_case in MIXED_CONTENT_SCENARIOS:
            test_data = {
                'body': test_case['html'],
                'content_type': 'text/html'
            }
            
            response = test_client.post('/quotation_extraction',
                                      data=json.dumps(test_data),
                                      content_type='application/json')
            
            assert response.status_code == 200
            result = json.loads(response.data)
            
            # Main content should be preserved
            for element in test_case['should_preserve']:
                if element not in test_case.get('should_remove', []):
                    assert element in result['reply'], f"Main content '{element}' should be preserved in API result"

    def test_error_recovery_integration(self, test_client):
        """Test error recovery across integrated components."""
        # Test with various problematic inputs
        problematic_inputs = [
            '',  # Empty content
            '<div>',  # Malformed HTML
            'Plain text without HTML',  # Plain text
            '<script>alert("test")</script>',  # Malicious content
            '☃' * 1000,  # Large Unicode content
        ]
        
        for content in problematic_inputs:
            test_data = {
                'body': content,
                'content_type': 'text/html'
            }
            
            response = test_client.post('/quotation_extraction',
                                      data=json.dumps(test_data),
                                      content_type='application/json')
            
            # Should handle gracefully without crashing
            assert response.status_code in [200, 400]
            
            if response.status_code == 200:
                result = json.loads(response.data)
                assert 'reply' in result

    def test_performance_integration(self, test_client):
        """Test performance of integrated components."""
        import time
        
        # Create moderately complex content
        complex_content = '''
        <div>Performance test email content.</div>
        ''' + '<p>Content paragraph. </p>' * 100 + '''
        <div class="gmail_signature">
            <table>
                <tr>
                    <td>
                        <p>John Doe<br>
                        Senior Performance Engineer<br>
                        Performance Team<br>
                        john.doe@performance.com<br>
                        Mobile: +1 (555) 123-4567</p>
                    </td>
                </tr>
            </table>
        </div>
        '''
        
        test_data = {
            'body': complex_content,
            'content_type': 'text/html'
        }
        
        start_time = time.time()
        response = test_client.post('/quotation_extraction',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Should complete within reasonable time
        processing_time = end_time - start_time
        assert processing_time < 2.0, f"Integration processing too slow: {processing_time:.2f}s"
        
        result = json.loads(response.data)
        assert 'Performance test' in result['reply']
        assert 'john.doe@performance.com' not in result['reply']


class TestDataFlowIntegration:
    """Test data flow between different processing stages."""

    def test_html_input_to_text_output_flow(self, test_client):
        """Test complete flow from HTML input to clean text output."""
        html_input = '''
        <html>
        <head><title>Email</title></head>
        <body>
            <div style="font-family: Arial;">
                <p>Dear <strong>Team</strong>,</p>
                <p>The <em>quarterly review</em> meeting is scheduled for:</p>
                <ul>
                    <li><b>Date:</b> Friday, Jan 5th</li>
                    <li><b>Time:</b> 2:00 PM - 3:30 PM</li>
                    <li><b>Location:</b> Conference Room A</li>
                </ul>
                <p>Please prepare your <a href="#reports">reports</a> beforehand.</p>
                <p>Looking forward to seeing everyone there!</p>
            </div>
            <div class="elementToProof">
                <p>Best regards<br>
                Meeting Coordinator<br>
                coordinator@company.com<br>
                Extension: 1234</p>
            </div>
        </body>
        </html>
        '''
        
        test_data = {
            'body': html_input,
            'content_type': 'text/html'
        }
        
        response = test_client.post('/quotation_extraction',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        # Should preserve main content structure
        assert 'Dear Team' in result['reply']
        assert 'quarterly review' in result['reply']
        assert 'Friday, Jan 5th' in result['reply']
        assert '2:00 PM - 3:30 PM' in result['reply']
        assert 'Conference Room A' in result['reply']
        assert 'reports' in result['reply']
        assert 'Looking forward' in result['reply']
        
        # Should remove signature
        assert 'coordinator@company.com' not in result['reply']
        assert 'Meeting Coordinator' not in result['reply']
        assert 'Extension: 1234' not in result['reply']

    def test_markdown_conversion_flow(self, test_client):
        """Test HTML to Markdown conversion data flow."""
        html_input = '''
        <div>
            <h1>Important Announcement</h1>
            <p>We're excited to announce the launch of our new <strong>product features</strong>:</p>
            <ol>
                <li><em>Advanced Analytics</em> dashboard</li>
                <li><strong>Real-time notifications</strong></li>
                <li>Enhanced <a href="https://docs.example.com">documentation</a></li>
            </ol>
            <blockquote>
                <p>"This will revolutionize our workflow" - Beta User</p>
            </blockquote>
        </div>
        <div class="gmail_signature">
            <p>Product Team<br>
            product@company.com</p>
        </div>
        '''
        
        test_data = {
            'html': html_input,
            'content_type': 'text/html'
        }
        
        response = test_client.post('/html_markdown',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        markdown = result['markdown']
        
        # Should convert to proper markdown format
        assert 'Important Announcement' in markdown
        assert 'product features' in markdown
        assert 'Advanced Analytics' in markdown
        assert 'Real-time notifications' in markdown
        assert 'documentation' in markdown
        assert 'revolutionize our workflow' in markdown
        
        # Signature should be removed
        assert 'product@company.com' not in markdown

    def test_error_propagation_flow(self, test_client):
        """Test how errors propagate through the processing flow."""
        # Test various error scenarios
        error_scenarios = [
            # Invalid JSON
            ('invalid json', 'application/json', 400),
            
            # Missing required fields
            ('{}', 'application/json', [200, 400]),
            
            # Very large content
            (json.dumps({'body': 'x' * 1000000, 'content_type': 'text/html'}), 'application/json', [200, 413]),
        ]
        
        for data, content_type, expected_codes in error_scenarios:
            response = test_client.post('/quotation_extraction',
                                      data=data,
                                      content_type=content_type)
            
            if isinstance(expected_codes, list):
                assert response.status_code in expected_codes
            else:
                assert response.status_code == expected_codes

    def test_consistency_across_endpoints(self, test_client):
        """Test consistency of processing across different API endpoints."""
        test_html = '''
        <div>Consistent test content.</div>
        <div class="gmail_signature">Test Signature<br>test@example.com</div>
        '''
        
        # Test quotation_extraction endpoint
        quotation_data = {
            'body': test_html,
            'content_type': 'text/html'
        }
        
        quotation_response = test_client.post('/quotation_extraction',
                                            data=json.dumps(quotation_data),
                                            content_type='application/json')
        
        # Test html_markdown endpoint
        markdown_data = {
            'html': test_html,
            'content_type': 'text/html'
        }
        
        markdown_response = test_client.post('/html_markdown',
                                           data=json.dumps(markdown_data),
                                           content_type='application/json')
        
        # Both should handle signature removal consistently
        if quotation_response.status_code == 200 and markdown_response.status_code == 200:
            quotation_result = json.loads(quotation_response.data)
            markdown_result = json.loads(markdown_response.data)
            
            # Both should preserve main content
            assert 'Consistent test content' in quotation_result['reply']
            assert 'Consistent test content' in markdown_result['markdown']
            
            # Both should remove signature
            assert 'test@example.com' not in quotation_result['reply']
            assert 'test@example.com' not in markdown_result['markdown']


class TestRegressionIntegration:
    """Test integration scenarios that have caused issues in the past."""

    def test_outlook_elementtoproof_regression(self, test_client):
        """Test regression for Outlook elementToProof class handling."""
        outlook_email = '''
        <div>Meeting notes from today's discussion:</div>
        <ul>
            <li>Reviewed Q4 goals</li>
            <li>Discussed budget allocation</li>
            <li>Planned next quarter initiatives</li>
        </ul>
        <div>Action items will be sent separately.</div>
        <div class="elementToProof">
            <p style="margin:0">
                <span>Best regards<br>
                John Smith<br>
                Senior Project Manager<br>
                john.smith@company.com<br>
                Direct: +1 (555) 123-4567</span>
            </p>
        </div>
        '''
        
        test_data = {
            'body': outlook_email,
            'content_type': 'text/html'
        }
        
        response = test_client.post('/quotation_extraction',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        # Should preserve meeting content
        assert 'Meeting notes' in result['reply']
        assert 'Q4 goals' in result['reply']
        assert 'budget allocation' in result['reply']
        assert 'Action items' in result['reply']
        
        # Should remove Outlook signature
        assert 'john.smith@company.com' not in result['reply']
        assert 'Senior Project Manager' not in result['reply']

    def test_german_signature_regression(self, test_client):
        """Test regression for German signature detection."""
        german_email = '''
        <div>Hallo Team,</div>
        <div><br></div>
        <div>die Präsentation für morgen ist fertig.</div>
        <div>Bitte überprüft die Folien vor dem Meeting.</div>
        <div><br></div>
        <div>Vielen Dank!</div>
        <div><br></div>
        <div>Mit freundlichen Grüßen</div>
        <div>Klaus Weber<br>
        Projektleiter<br>
        klaus.weber@firma.de<br>
        Tel: +49 30 12345678</div>
        '''
        
        test_data = {
            'body': german_email,
            'content_type': 'text/html'
        }
        
        response = test_client.post('/quotation_extraction',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        # Should preserve German email content
        assert 'Hallo Team' in result['reply']
        assert 'Präsentation' in result['reply']
        assert 'überprüft' in result['reply']
        assert 'Vielen Dank' in result['reply']
        
        # Should remove German signature
        assert 'klaus.weber@firma.de' not in result['reply']
        assert 'Projektleiter' not in result['reply']

    def test_mixed_content_regression(self, test_client):
        """Test regression for mixed content scenarios."""
        mixed_email = '''
        <div>Hi everyone,</div>
        <div><br></div>
        <div>The server maintenance is complete. Here are the details:</div>
        <div><br></div>
        <table>
            <tr><td>Server:</td><td>PROD-WEB-01</td></tr>
            <tr><td>Downtime:</td><td>2 hours</td></tr>
            <tr><td>Status:</td><td>✓ Operational</td></tr>
            <tr><td>Contact:</td><td>ops@company.com</td></tr>
        </table>
        <div><br></div>
        <div>All systems are now running normally.</div>
        <div><br></div>
        <div>Best regards</div>
        <div>Operations Team<br>
        ops-lead@company.com</div>
        '''
        
        test_data = {
            'body': mixed_email,
            'content_type': 'text/html'
        }
        
        response = test_client.post('/quotation_extraction',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        # Should preserve main content and table data
        assert 'server maintenance' in result['reply']
        assert 'PROD-WEB-01' in result['reply']
        assert '2 hours' in result['reply']
        assert 'Operational' in result['reply']
        assert 'ops@company.com' in result['reply']  # This is content, not signature
        assert 'running normally' in result['reply']
        
        # Should remove signature
        assert 'ops-lead@company.com' not in result['reply']
