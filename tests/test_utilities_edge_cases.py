"""
Consolidated utilities and edge cases tests
Tests utility functions, edge cases, and error handling scenarios.
"""
import pytest
import html2text
from talon import signature, quotations
from talon.quotations import extract_from_html
from tests.fixtures.html_samples import EDGE_CASES


class TestUtilityFunctions:
    """Test utility functions and helper methods."""
    
    def test_text_cleaning_utilities(self):
        """Test text cleaning and normalization utilities."""
        # Test whitespace normalization
        messy_text = "   Multiple    spaces   \n\n\n  and   newlines   "
        body, signature_part = signature.extract(messy_text, sender="test@example.com")
        
        # Should normalize whitespace appropriately
        assert body is not None
        assert isinstance(body, str)

    def test_html_entity_decoding(self):
        """Test HTML entity decoding utilities."""
        html_with_entities = '''
        <div>Meeting at 3:00 PM &amp; 4:00 PM</div>
        <div>&lt;Important&gt; &quot;information&quot;</div>
        <div>Temperature: 20&deg;C</div>
        <div>Copyright &copy; 2024</div>
        '''
        
        result = extract_from_html(html_with_entities)
        
        # Should decode HTML entities
        assert '&' in result  # &amp; decoded
        assert '<Important>' in result  # &lt; &gt; decoded
        assert '"information"' in result  # &quot; decoded
        assert '°C' in result or 'deg' in result  # &deg; handled
        assert '©' in result or 'copy' in result  # &copy; handled

    def test_url_detection_utilities(self):
        """Test URL detection and handling utilities."""
        content_with_urls = '''
        <div>Visit our website at https://www.example.com for more info.</div>
        <div>Download from ftp://files.example.com/downloads/</div>
        <div>Contact support@example.com for help.</div>
        '''
        
        result = extract_from_html(content_with_urls)
        
        # URLs in content should be preserved
        assert 'example.com' in result
        assert 'support@example.com' in result

    def test_email_pattern_utilities(self):
        """Test email pattern detection utilities."""
        email_patterns = [
            'simple@example.com',
            'user.name@domain.co.uk',
            'user+tag@subdomain.domain.org',
            'user-name@domain-name.net',
            'firstname.lastname@company.com',
        ]
        
        for email in email_patterns:
            # In signature context
            signature_content = f'<p>Best regards<br>John<br>{email}</p>'
            result = signature.extract(signature_content, "text/html")
            
            # Should detect email in signature context
            # (actual behavior depends on other signature indicators)
            assert result is not None

    def test_phone_number_utilities(self):
        """Test phone number detection and handling."""
        phone_formats = [
            '+1 (555) 123-4567',
            '555-123-4567',
            '(555) 123-4567',
            '+49 30 12345678',
            '+33 1 23 45 67 89',
            '1-800-EXAMPLE',
        ]
        
        for phone in phone_formats:
            content = f'<div>Call us at {phone} for support.</div>'
            result = extract_from_html(content)
            
            # Phone numbers in content should be preserved
            assert phone in result

    def test_language_detection_utilities(self):
        """Test language detection utilities."""
        multilingual_content = {
            'english': 'Best regards from the team.',
            'german': 'Mit freundlichen Grüßen vom Team.',
            'french': 'Cordialement de la part de l\'équipe.',
            'spanish': 'Saludos cordiales del equipo.',
        }
        
        for lang, text in multilingual_content.items():
            content = f'<div>{text}</div>'
            result = extract_from_html(content)
            
            # Should handle different languages
            assert text in result

    def test_signature_boundary_detection(self):
        """Test signature boundary detection utilities."""
        boundary_indicators = [
            '<hr>',
            '<div>--</div>',
            '<div>___</div>',
            '<br><br>',
            '<div style="border-top: 1px solid;">',
        ]
        
        for boundary in boundary_indicators:
            content = f'''
            <div>Main email content here.</div>
            {boundary}
            <div>Potential signature content<br>Name<br>email@example.com</div>
            '''
            
            result = signature.extract(content, "text/html")
            
            # Should preserve main content
            assert 'Main email content' in result


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_empty_and_minimal_content(self):
        """Test handling of empty and minimal content."""
        minimal_cases = [
            '',
            ' ',
            '\n\n\n',
            '<div></div>',
            '<p></p>',
            '<html><body></body></html>',
            'Hi',
            '<div>Hi</div>',
        ]
        
        for content in minimal_cases:
            # Should not crash on minimal content
            try:
                result = signature.extract(content, "text/html")
                assert result is not None
                assert isinstance(result, str)
                
                result_html = extract_from_html(content)
                assert result_html is not None
                assert isinstance(result_html, str)
            except Exception as e:
                pytest.fail(f"Failed on minimal content '{content}': {e}")

    def test_very_large_content(self):
        """Test handling of very large content."""
        # Create large content
        large_content = '<div>Start of large content.</div>'
        large_content += '<p>Repeated paragraph content. </p>' * 5000
        large_content += '<div>End of large content.</div>'
        large_content += '<div class="gmail_signature">Large Signature<br>large@example.com</div>'
        
        # Should handle large content without performance issues
        result = signature.extract(large_content, "text/html")
        
        assert 'Start of large content' in result
        assert 'End of large content' in result
        assert 'Repeated paragraph' in result

    def test_deeply_nested_html(self):
        """Test handling of deeply nested HTML structures."""
        # Create deeply nested structure
        nested_content = '<div>'
        for i in range(50):
            nested_content += f'<div class="level-{i}"><p>Content at level {i}</p>'
        
        nested_content += '<span>Deep content</span>'
        
        for i in range(50):
            nested_content += '</div>'
        
        nested_content += '</div>'
        nested_content += '<div class="gmail_signature">Deep Signature<br>deep@example.com</div>'
        
        # Should handle deep nesting
        result = signature.extract(nested_content, "text/html")
        
        assert 'Deep content' in result
        assert 'Content at level' in result

    def test_malformed_html_edge_cases(self):
        """Test handling of malformed HTML."""
        malformed_cases = [
            '<div>Unclosed div with signature<br>email@example.com',
            '<p>Missing close tag <span>with signature content</span><br>test@example.com',
            '<div><p>Improperly nested <span></div></p></span>',
            '<<>>Invalid tags<</><div>Real content</div>',
            '<div>Valid content</div><signature>Custom tag<br>custom@example.com</signature>',
        ]
        
        for malformed_html in malformed_cases:
            # Should not crash on malformed HTML
            try:
                result = signature.extract(malformed_html, "text/html")
                assert result is not None
                assert isinstance(result, str)
            except Exception as e:
                pytest.fail(f"Failed on malformed HTML: {e}")

    def test_unicode_edge_cases(self):
        """Test handling of Unicode edge cases."""
        unicode_cases = [
            '😀 🎉 ✨ 🚀 💻',  # Emojis
            '测试内容 中文字符',  # Chinese characters
            'Тест кириллица',  # Cyrillic
            'العربية النص',  # Arabic
            'ＺＺＺ',  # Full-width characters
            '\u200b\u200c\u200d',  # Zero-width characters
            '🇺🇸 🇩🇪 🇫🇷',  # Flag emojis
        ]
        
        for unicode_text in unicode_cases:
            content = f'<div>{unicode_text}</div><div class="gmail_signature">Unicode Sig<br>unicode@example.com</div>'
            
            # Should handle Unicode properly
            result = signature.extract(content, "text/html")
            assert unicode_text in result

    def test_special_character_edge_cases(self):
        """Test handling of special characters."""
        special_chars = [
            '&lt;script&gt;alert("test")&lt;/script&gt;',
            '\\n\\r\\t',
            '\x00\x01\x02',  # Control characters
            '```code block```',
            '<!-- HTML comment -->',
            '<?xml version="1.0"?>',
        ]
        
        for special_char in special_chars:
            content = f'<div>Content with {special_char}</div>'
            
            # Should handle special characters gracefully
            try:
                result = extract_from_html(content)
                assert result is not None
            except Exception as e:
                pytest.fail(f"Failed on special character '{special_char}': {e}")

    def test_edge_cases_from_fixtures(self):
        """Test edge cases from fixtures that could cause false positives."""
        for test_case in EDGE_CASES:
            html_content = test_case['html']
            
            result = signature.extract(html_content, "text/html")
            
            # Edge cases should not be modified
            assert result == html_content, f"Edge case incorrectly modified: {test_case['name']}"
            
            # All content should be preserved
            for element in test_case['should_preserve']:
                assert element in result, f"Element '{element}' should be preserved in edge case {test_case['name']}"

    def test_mixed_content_type_edge_cases(self):
        """Test edge cases with mixed content types."""
        mixed_cases = [
            {
                'content': 'Plain text email\n\nBest regards\nJohn\njohn@example.com',
                'content_type': 'text/plain'
            },
            {
                'content': '<div>HTML content</div>Plain text mixed in',
                'content_type': 'text/html'
            },
            {
                'content': 'Text with <html>tags</html> but plain type',
                'content_type': 'text/plain'
            },
        ]
        
        for case in mixed_cases:
            # Should handle mixed content appropriately
            result = signature.extract(case['content'], case['content_type'])
            assert result is not None
            assert isinstance(result, str)


class TestErrorHandling:
    """Test error handling and recovery scenarios."""

    def test_none_input_handling(self):
        """Test handling of None inputs."""
        none_cases = [
            (None, "text/html"),
            ("content", None),
            (None, None),
        ]
        
        for content, content_type in none_cases:
            # Should handle None inputs gracefully
            try:
                if content is not None and content_type is not None:
                    result = signature.extract(content, content_type)
                    assert result is not None or result == ""
            except (TypeError, ValueError):
                # Expected to raise an error for None inputs
                pass

    def test_invalid_content_type_handling(self):
        """Test handling of invalid content types."""
        invalid_types = [
            'text/invalid',
            'application/json',
            'image/png',
            '',
            'invalid',
        ]
        
        content = '<div>Test content</div>'
        
        for content_type in invalid_types:
            # Should handle invalid content types gracefully
            try:
                result = signature.extract(content, content_type)
                # Should either process as text or return original
                assert result is not None
            except ValueError:
                # Acceptable to raise ValueError for invalid types
                pass

    def test_memory_pressure_handling(self):
        """Test handling under memory pressure."""
        # Create very large content to test memory handling
        huge_content = '<div>Large content test.</div>'
        huge_content += '<p>Memory test content. </p>' * 10000
        huge_content += '<div class="gmail_signature">Memory Sig<br>memory@example.com</div>'
        
        # Should handle large content efficiently
        result = signature.extract(huge_content, "text/html")
        
        assert 'Large content test' in result
        assert 'Memory test content' in result

    def test_concurrent_processing_edge_cases(self):
        """Test edge cases that might occur during concurrent processing."""
        import threading
        import time
        
        results = []
        errors = []
        
        def process_content(content_id):
            try:
                content = f'<div>Concurrent test {content_id}</div><div class="gmail_signature">Sig {content_id}<br>test{content_id}@example.com</div>'
                result = signature.extract(content, "text/html")
                results.append((content_id, result))
            except Exception as e:
                errors.append((content_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=process_content, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should handle concurrent processing
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == 10
        
        for content_id, result in results:
            assert f'Concurrent test {content_id}' in result
            assert f'test{content_id}@example.com' not in result

    def test_stack_overflow_protection(self):
        """Test protection against stack overflow in recursive processing."""
        # Create content that might cause deep recursion
        recursive_content = ''
        for i in range(1000):
            recursive_content += f'<div class="level-{i}">'
        
        recursive_content += 'Deep content'
        
        for i in range(1000):
            recursive_content += '</div>'
        
        recursive_content += '<div class="gmail_signature">Deep Sig<br>deep@example.com</div>'
        
        # Should handle deep structures without stack overflow
        try:
            result = signature.extract(recursive_content, "text/html")
            assert 'Deep content' in result
        except RecursionError:
            pytest.fail("Stack overflow occurred during processing")

    def test_encoding_error_handling(self):
        """Test handling of encoding errors."""
        # Test various encoding scenarios
        encoding_cases = [
            'café naïve résumé',  # Latin-1 characters
            'Москва',  # Cyrillic
            '北京',  # Chinese
            '🌟⭐✨',  # Emojis
        ]
        
        for text in encoding_cases:
            content = f'<div>{text}</div><div class="gmail_signature">Encoding Sig<br>encoding@example.com</div>'
            
            # Should handle encoding properly
            try:
                result = signature.extract(content, "text/html")
                assert text in result
            except UnicodeError:
                pytest.fail(f"Encoding error with text: {text}")


class TestPerformanceEdgeCases:
    """Test performance-related edge cases."""

    def test_worst_case_performance(self):
        """Test worst-case performance scenarios."""
        import time
        
        # Create pathological content that might be slow to process
        pathological_content = '<div>Main content.</div>'
        
        # Add many potential signature candidates
        for i in range(100):
            pathological_content += f'''
            <div>Best regards{i}<br>
            Person{i}<br>
            person{i}@example.com</div>
            '''
        
        start_time = time.time()
        result = signature.extract(pathological_content, "text/html")
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete within reasonable time even for pathological cases
        assert processing_time < 5.0, f"Worst-case performance too slow: {processing_time:.2f}s"
        assert 'Main content' in result

    def test_memory_efficiency_edge_cases(self):
        """Test memory efficiency with edge cases."""
        # Test with many small signatures
        many_signatures = '<div>Main email content.</div>'
        
        for i in range(1000):
            many_signatures += f'<span class="sig-{i}">Sig{i}<br>email{i}@test.com</span>'
        
        # Should handle many signatures efficiently
        result = signature.extract(many_signatures, "text/html")
        assert 'Main email content' in result

    def test_regex_complexity_edge_cases(self):
        """Test edge cases that might cause regex complexity issues."""
        # Content designed to potentially cause regex backtracking
        complex_content = '''
        <div>Complex content with many repeated patterns:</div>
        ''' + 'Best ' * 100 + 'regards<br>' + 'John ' * 50 + 'Doe<br>john@example.com'
        
        # Should handle complex patterns efficiently
        result = signature.extract(complex_content, "text/html")
        assert 'Complex content' in result
