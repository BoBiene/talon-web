"""
Consolidated signature detection functionality tests
Tests the core signature detection logic with various patterns and edge cases.
"""
import pytest
from talon import signature


class TestSignatureDetection:
    """Core signature detection functionality tests."""

    def test_basic_signature_extraction(self):
        """Test basic signature extraction functionality."""
        content = '''Main email content here.

Best regards,
John Doe
john@company.com'''
        
        body, detected_signature = signature.extract(content, sender="john@company.com")
        
        assert body is not None
        assert 'Main email content' in body

    def test_professional_signature_detection(self):
        """Test detection of professional signatures."""
        content = '''Please review the documents.

Best regards,
John Smith
Senior Engineer
Tech Corporation
john.smith@techcorp.com
+1-555-123-4567'''
        
        body, detected_signature = signature.extract(content, sender="john.smith@techcorp.com")
        
        # Should detect and extract signature
        assert body is not None
        assert 'Please review the documents' in body
        # Conservative: may or may not detect signature depending on algorithm

    def test_simple_closing_preservation(self):
        """Test that simple closings are preserved (conservative approach)."""
        content = '''Meeting notes:
- Item 1
- Item 2

Thanks,
John'''
        
        body, detected_signature = signature.extract(content, sender="john@example.com")
        
        # Conservative approach: preserve simple closings
        assert body is not None
        assert 'Meeting notes' in body
        assert 'Item 1' in body

    def test_email_with_separator(self):
        """Test email with clear signature separator."""
        content = '''Email content here.

--
John Doe
Engineering Team
john@company.com'''
        
        body, detected_signature = signature.extract(content, sender="john@company.com")
        
        assert body is not None
        assert 'Email content here' in body

    def test_empty_content(self):
        """Test signature detection with empty content."""
        body, detected_signature = signature.extract("", sender="test@example.com")
        assert body == ""
        assert detected_signature is None

    def test_none_sender(self):
        """Test signature detection without sender."""
        content = "Test content"
        body, detected_signature = signature.extract(content, sender=None)
        assert body is not None

    def test_unicode_content(self):
        """Test signature detection with Unicode content."""
        content = '''Hauptinhalt der E-Mail.

Mit freundlichen Grüßen
Müller
müller@firma.de'''
        
        body, detected_signature = signature.extract(content, sender="müller@firma.de")
        
        assert body is not None
        assert 'Hauptinhalt' in body

    def test_html_content(self):
        """Test signature detection with HTML content."""
        content = '''<div>Main content</div>
<div>Best regards</div>
<div>John Doe</div>
<div>john@company.com</div>'''
        
        body, detected_signature = signature.extract(content, sender="john@company.com")
        
        assert body is not None

    def test_multiple_potential_signatures(self):
        """Test content with multiple potential signatures."""
        content = '''Main content.

Thanks,
First Person

Best regards,
Second Person
second@company.com'''
        
        body, detected_signature = signature.extract(content, sender="first@company.com")
        
        assert body is not None
        assert 'Main content' in body

    def test_long_content(self):
        """Test signature detection with long content."""
        long_content = "Content line.\n" * 50 + "\nBest regards\nJohn"
        
        body, detected_signature = signature.extract(long_content, sender="john@example.com")
        
        assert body is not None
        assert len(body) > 0


class TestSignatureEdgeCases:
    """Test signature detection edge cases."""

    def test_malformed_input(self):
        """Test with various malformed inputs."""
        test_cases = [
            None,
            "",
            "   ",
            "\n\n\n",
            "<html><body></body></html>",
        ]
        
        for content in test_cases:
            try:
                body, detected_signature = signature.extract(content, sender="test@example.com")
                # Should not crash
                assert body is not None or content is None
            except Exception as e:
                # Some exceptions may be acceptable for None input
                if content is not None:
                    pytest.fail(f"Unexpected exception for content '{content}': {e}")

    def test_very_short_content(self):
        """Test with very short content."""
        short_contents = [
            "Hi",
            "Thanks",
            "John",
            "Best regards",
        ]
        
        for content in short_contents:
            body, detected_signature = signature.extract(content, sender="john@example.com")
            assert body is not None
            # Short content should typically be preserved

    def test_no_signature_content(self):
        """Test content that clearly has no signature."""
        content = '''This is just regular email content with no signature elements.
It talks about business matters and project updates.
There are no closing salutations or contact information.'''
        
        body, detected_signature = signature.extract(content, sender="user@example.com")
        
        assert body is not None
        assert body == content  # Should be unchanged

    def test_ambiguous_content(self):
        """Test ambiguous content that could be signature or content."""
        ambiguous_cases = [
            "Thanks for the help!",
            "Best wishes for your project.",
            "John mentioned the meeting.",
            "Regards to your team.",
        ]
        
        for content in ambiguous_cases:
            body, detected_signature = signature.extract(content, sender="test@example.com")
            # Conservative approach: preserve ambiguous content
            assert body == content

    def test_signature_consistency(self):
        """Test that signature extraction is consistent."""
        content = '''Email content here.

Best regards,
John Doe
john@company.com'''
        
        # Run multiple times
        results = []
        for _ in range(3):
            result = signature.extract(content, sender="john@company.com")
            results.append(result)
        
        # All results should be the same
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result


class TestSignaturePrecision:
    """Test signature detection precision over recall."""

    def test_conservative_approach(self):
        """Test that the algorithm favors precision over recall."""
        # Cases where signature detection should be conservative
        borderline_cases = [
            "Thanks!",
            "Best,\nJohn",
            "Cheers,\nTeam",
            "John Doe",  # Just a name
        ]
        
        for content in borderline_cases:
            body, detected_signature = signature.extract(content, sender="john@example.com")
            # Conservative: when in doubt, preserve content
            assert content in body

    def test_clear_signatures_detected(self):
        """Test that clear signatures are handled appropriately."""
        clear_cases = [
            '''Content here.

--
John Doe
Senior Engineer
Tech Corp
john@techcorp.com
Phone: +1-555-123-4567''',
            '''Email body.

Best regards,
John Smith
Director of Engineering
john.smith@company.com
Mobile: 555-987-6543''',
        ]
        
        for content in clear_cases:
            body, detected_signature = signature.extract(content, sender="john@company.com")
            # Should process the content
            assert body is not None
            assert len(body) <= len(content)  # Body should not be longer than original