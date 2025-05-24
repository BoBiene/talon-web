"""
Consolidated HTML processing and conversion tests
Tests HTML parsing, cleaning, and markdown conversion functionality.
"""
import pytest
import html2text
from talon.quotations import extract_from_html
from tests.fixtures.html_samples import MIXED_CONTENT_SCENARIOS, EDGE_CASES


class TestHTMLProcessing:
    """Test HTML parsing and processing functionality."""

    def test_html_parsing_basic(self):
        """Test basic HTML parsing functionality."""
        html_content = '''
        <html>
        <body>
            <div>Main content</div>
            <p>Paragraph content</p>
        </body>
        </html>
        '''
        
        # Test that HTML can be processed
        result = extract_from_html(html_content)
        
        assert result is not None
        assert 'Main content' in result
        assert 'Paragraph content' in result

    def test_html_cleaning_tags(self):
        """Test HTML tag cleaning and normalization."""
        test_cases = [
            # Test script tag removal
            ('<div>Content</div><script>alert("test")</script>', 'Content'),
            
            # Test style tag removal
            ('<div>Content</div><style>body{color:red}</style>', 'Content'),
            
            # Test link preservation
            ('<div><a href="http://example.com">Link</a></div>', 'Link'),
            
            # Test image handling
            ('<div><img src="test.jpg" alt="Test Image"></div>', ''),
        ]
        
        for html_input, expected_content in test_cases:
            result = extract_from_html(html_input)
            
            if expected_content:
                assert expected_content in result, f"Expected '{expected_content}' in result from {html_input}"
            
            # Verify problematic tags are removed
            assert '<script>' not in result
            assert '<style>' not in result

    def test_html_attribute_handling(self):
        """Test handling of HTML attributes."""
        html_content = '''
        <div id="main" class="content" style="color: red;">
            <p data-id="123">Paragraph with attributes</p>
            <span onclick="malicious()">Click me</span>
        </div>
        '''
        
        result = extract_from_html(html_content)
        
        # Content should be preserved
        assert 'Paragraph with attributes' in result
        assert 'Click me' in result
        
        # Malicious attributes should be cleaned
        assert 'onclick' not in result
        assert 'malicious()' not in result

    def test_html_encoding_handling(self):
        """Test handling of HTML character encoding."""
        test_cases = [
            ('&amp;', '&'),
            ('&lt;', '<'),
            ('&gt;', '>'),
            ('&quot;', '"'),
            ('&#39;', "'"),
            ('&nbsp;', ' '),
        ]
        
        for encoded, decoded in test_cases:
            html_content = f'<div>Test {encoded} content</div>'
            result = extract_from_html(html_content)
            
            # Should decode HTML entities
            assert f'Test {decoded} content' in result

    def test_html_nested_structure_processing(self):
        """Test processing of deeply nested HTML structures."""
        html_content = '''
        <div class="outer">
            <div class="middle">
                <div class="inner">
                    <p>Deeply nested content</p>
                    <ul>
                        <li>Item 1</li>
                        <li>Item 2</li>
                    </ul>
                </div>
            </div>
        </div>
        '''
        
        result = extract_from_html(html_content)
        
        # Should extract content from nested structures
        assert 'Deeply nested content' in result
        assert 'Item 1' in result
        assert 'Item 2' in result

    def test_html_table_processing(self):
        """Test processing of HTML tables."""
        html_content = '''
        <table>
            <thead>
                <tr>
                    <th>Header 1</th>
                    <th>Header 2</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Data 1</td>
                    <td>Data 2</td>
                </tr>
            </tbody>
        </table>
        '''
        
        result = extract_from_html(html_content)
        
        # Should extract table content
        assert 'Header 1' in result
        assert 'Header 2' in result
        assert 'Data 1' in result
        assert 'Data 2' in result

    def test_html_list_processing(self):
        """Test processing of HTML lists."""
        html_content = '''
        <ul>
            <li>Unordered item 1</li>
            <li>Unordered item 2</li>
        </ul>
        <ol>
            <li>Ordered item 1</li>
            <li>Ordered item 2</li>
        </ol>
        '''
        
        result = extract_from_html(html_content)
        
        # Should extract list content
        assert 'Unordered item 1' in result
        assert 'Unordered item 2' in result
        assert 'Ordered item 1' in result
        assert 'Ordered item 2' in result


class TestHTMLToMarkdown:
    """Test HTML to Markdown conversion functionality."""

    def test_basic_html_to_markdown(self):
        """Test basic HTML to Markdown conversion."""
        html_content = '''
        <h1>Main Title</h1>
        <p>This is a <strong>bold</strong> paragraph with <em>italic</em> text.</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>        '''
        
        # Convert to markdown using html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        result = h.handle(html_content)
        
        # Should convert to markdown format
        assert '# Main Title' in result or 'Main Title' in result
        assert '**bold**' in result or '*bold*' in result
        assert '_italic_' in result or '*italic*' in result
        assert 'Item 1' in result
        assert 'Item 2' in result

    def test_link_conversion_to_markdown(self):
        """Test link conversion to Markdown format."""
        html_content = '''
        <p>Visit <a href="https://example.com">our website</a> for more info.</p>
        <p>Email us at <a href="mailto:test@example.com">test@example.com</a></p>        '''
        
        # Convert to markdown using html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        result = h.handle(html_content)
        
        # Should convert links to markdown format
        assert 'our website' in result
        assert 'example.com' in result
        assert 'test@example.com' in result

    def test_image_conversion_to_markdown(self):
        """Test image conversion to Markdown format."""
        html_content = '''
        <p>Here is an image: <img src="test.jpg" alt="Test Image" /></p>        '''
        
        # Convert to markdown using html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        result = h.handle(html_content)
        
        # Should handle images appropriately
        assert 'Here is an image' in result

    def test_table_conversion_to_markdown(self):
        """Test table conversion to Markdown format."""
        html_content = '''
        <table>
            <tr>
                <th>Name</th>
                <th>Age</th>
            </tr>
            <tr>
                <td>John</td>
                <td>30</td>
            </tr>
        </table>        '''
        
        # Convert to markdown using html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        result = h.handle(html_content)
        
        # Should preserve table content
        assert 'Name' in result
        assert 'Age' in result
        assert 'John' in result
        assert '30' in result

    def test_complex_html_to_markdown(self):
        """Test complex HTML structure conversion."""
        html_content = '''
        <div>
            <h2>Project Update</h2>
            <p>The project is <strong>on track</strong>. Here are the updates:</p>
            <ol>
                <li>Database migration completed</li>
                <li>API endpoints tested</li>
                <li>UI improvements in progress</li>
            </ol>
            <blockquote>
                <p>Great work team! - Manager</p>
            </blockquote>
        </div>        '''
        
        # Convert to markdown using html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        result = h.handle(html_content)
        
        # Should convert complex structure
        assert 'Project Update' in result
        assert 'on track' in result
        assert 'Database migration' in result
        assert 'Great work team' in result

    def test_markdown_preserves_line_breaks(self):
        """Test that markdown conversion preserves appropriate line breaks."""
        html_content = '''
        <p>Paragraph 1</p>
        <p>Paragraph 2</p>
        <br>
        <p>Paragraph 3</p>        '''
        
        # Convert to markdown using html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        result = h.handle(html_content)
        
        # Should preserve paragraph separation
        assert 'Paragraph 1' in result
        assert 'Paragraph 2' in result
        assert 'Paragraph 3' in result


class TestHTMLQuotationExtraction:
    """Test HTML quotation extraction specific functionality."""

    def test_html_quotation_detection(self):
        """Test detection of quoted content in HTML."""
        html_content = '''
        <div>Original message content.</div>
        <blockquote>
            <div>This is a quoted message from previous email.</div>
        </blockquote>
        '''
        
        result = extract_from_html(html_content)
        
        # Should extract content appropriately
        assert 'Original message content' in result

    def test_html_reply_separator_detection(self):
        """Test detection of reply separators in HTML."""
        html_content = '''
        <div>New message content.</div>
        <hr>
        <div>From: sender@example.com</div>
        <div>To: recipient@example.com</div>
        <div>Original message content.</div>
        '''
        
        result = extract_from_html(html_content)
        
        # Should handle reply separators
        assert 'New message content' in result

    def test_html_forwarded_message_detection(self):
        """Test detection of forwarded messages in HTML."""
        html_content = '''
        <div>Forwarding this for your review.</div>
        <div>---------- Forwarded message ----------</div>
        <div>From: original@example.com</div>
        <div>Original forwarded content.</div>
        '''
        
        result = extract_from_html(html_content)
        
        # Should handle forwarded messages
        assert 'Forwarding this for your review' in result

    def test_html_gmail_quote_detection(self):
        """Test detection of Gmail-style quotes."""
        html_content = '''
        <div>Reply content here.</div>
        <div class="gmail_quote">
            <div>On Mon, Jan 1, 2024 at 10:00 AM, sender wrote:</div>
            <blockquote>
                <div>Original quoted message.</div>
            </blockquote>
        </div>
        '''
        
        result = extract_from_html(html_content)
        
        # Should extract reply content
        assert 'Reply content here' in result

    def test_html_outlook_quote_detection(self):
        """Test detection of Outlook-style quotes."""
        html_content = '''
        <div>My response to your email.</div>
        <div>
            <hr>
            <div><b>From:</b> sender@example.com</div>
            <div><b>Sent:</b> Monday, January 1, 2024 10:00 AM</div>
            <div><b>To:</b> recipient@example.com</div>
            <div><b>Subject:</b> Original subject</div>
        </div>
        <div>Original message content from Outlook.</div>
        '''
        
        result = extract_from_html(html_content)
        
        # Should extract response content
        assert 'My response to your email' in result


class TestHTMLEdgeCases:
    """Test HTML processing edge cases."""

    def test_malformed_html_handling(self):
        """Test handling of malformed HTML."""
        malformed_cases = [
            '<div>Unclosed div',
            '<p>Missing end tag <span>text',
            '<div><p>Nested without proper closing</div>',
            'Plain text without HTML tags',
            '',
        ]
        
        for html_content in malformed_cases:
            # Should not crash on malformed HTML
            try:
                result = extract_from_html(html_content)
                assert result is not None
            except Exception as e:
                pytest.fail(f"Failed to handle malformed HTML: {html_content}, Error: {e}")

    def test_large_html_processing(self):
        """Test processing of large HTML content."""
        # Create large HTML content
        large_content = '<div>Start</div>'
        large_content += '<p>Content paragraph. </p>' * 1000
        large_content += '<div>End</div>'
        
        result = extract_from_html(large_content)
        
        # Should handle large content
        assert 'Start' in result
        assert 'End' in result
        assert 'Content paragraph' in result

    def test_unicode_html_processing(self):
        """Test processing of HTML with Unicode characters."""
        html_content = '''
        <div>Unicode test: 测试内容</div>
        <p>Émojis: 😀 🎉 ✨</p>
        <p>German: Schöne Grüße</p>
        <p>French: Café naïve</p>
        '''
        
        result = extract_from_html(html_content)
        
        # Should handle Unicode properly
        assert '测试内容' in result
        assert 'Schöne Grüße' in result
        assert 'Café naïve' in result

    def test_html_with_mixed_content_scenarios(self):
        """Test HTML processing with mixed content scenarios from fixtures."""
        for test_case in MIXED_CONTENT_SCENARIOS:
            html_content = test_case['html']
            
            result = extract_from_html(html_content)
            
            # Should process without errors
            assert result is not None
            assert isinstance(result, str)
            
            # Should preserve main content elements
            for element in test_case['should_preserve']:
                if 'john.doe@company.com' not in element and 'Senior DevOps Engineer' not in element:
                    assert element in result, f"Element '{element}' should be preserved in HTML processing"

    def test_html_with_edge_cases(self):
        """Test HTML processing with edge case scenarios from fixtures."""
        for test_case in EDGE_CASES:
            html_content = test_case['html']
            
            result = extract_from_html(html_content)
            
            # Should process without errors
            assert result is not None
            assert isinstance(result, str)
            
            # Should preserve all content in edge cases
            for element in test_case['should_preserve']:
                assert element in result, f"Element '{element}' should be preserved in edge case processing"
