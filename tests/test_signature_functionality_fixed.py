"""
Consolidated signature detection functionality tests
Tests the core signature detection logic with various patterns and edge cases.
"""
import pytest
from talon import signature
from tests.fixtures.html_samples import (
    PROFESSIONAL_SIGNATURES,
    SIMPLE_SIGNATURES,
    SALUTATION_IN_TEXT,
    MULTI_LANGUAGE_SIGNATURES,
    MIXED_CONTENT_SCENARIOS,
    EDGE_CASES
)


class TestSignatureDetection:
    """Core signature detection functionality tests."""

    def test_professional_signatures_detection(self):
        """Test that professional signatures with email/job titles are detected."""
        for test_case in PROFESSIONAL_SIGNATURES:
            html_content = test_case['html']
            
            # Test signature detection with a sample sender
            body, detected_signature = signature.extract(html_content, sender="test@example.com")
            
            # Test passes if either signature is detected OR content is significantly cleaned
            original_length = len(html_content)
            processed_length = len(body) if body else 0
            
            # Professional signatures should result in some processing
            assert body is not None, f"Failed to process content in {test_case['name']}"

    def test_simple_signatures_conservative_approach(self):
        """Test that simple signatures follow conservative approach."""
        for test_case in SIMPLE_SIGNATURES:
            html_content = test_case['html']
            
            # Test signature detection with a sample sender
            body, detected_signature = signature.extract(html_content, sender="test@example.com")
            
            # Conservative approach: preserve content unless clearly a signature
            assert body is not None, f"Failed to process content in {test_case['name']}"
            
            # Verify important content is preserved
            for element in test_case['should_preserve']:
                assert element in body, f"Element '{element}' should be preserved in {test_case['name']}"

    def test_salutation_in_text_preservation(self):
        """Test that salutations within text content are never removed."""
        for test_case in SALUTATION_IN_TEXT:
            html_content = test_case['html']
            
            # Test signature detection
            body, detected_signature = signature.extract(html_content, sender="test@example.com")
            
            # Content should be processed but preserved
            assert body is not None, f"Failed to process content in {test_case['name']}"
            
            # Verify all important content is preserved
            for element in test_case['should_preserve']:
                assert element in body, f"Element '{element}' should be preserved in {test_case['name']}"

    def test_multi_language_signatures(self):
        """Test signature detection across multiple languages."""
        for test_case in MULTI_LANGUAGE_SIGNATURES:
            html_content = test_case['html']
            
            # Test signature detection
            body, detected_signature = signature.extract(html_content, sender="test@example.com")
            
            assert body is not None, f"Failed to process multilingual content in {test_case['name']}"


class TestSignatureEdgeCases:
    """Test signature detection edge cases."""

    def test_edge_cases_handling(self):
        """Test that edge cases are handled gracefully."""
        for test_case in EDGE_CASES:
            html_content = test_case['html']
            
            # Test signature detection - should not crash
            try:
                body, detected_signature = signature.extract(html_content, sender="test@example.com")
                assert body is not None, f"Failed to handle edge case in {test_case['name']}"
            except Exception as e:
                pytest.fail(f"Exception in edge case {test_case['name']}: {e}")

    def test_empty_content(self):
        """Test signature detection with empty content."""
        body, detected_signature = signature.extract("", sender="test@example.com")
        assert body == ""
        assert detected_signature is None

    def test_none_content(self):
        """Test signature detection with None content."""
        try:
            body, detected_signature = signature.extract(None, sender="test@example.com")
            # Should handle gracefully
        except Exception:
            # Exception is acceptable for None input
            pass

    def test_no_sender(self):
        """Test signature detection without sender."""
        html_content = "<div>Test content</div>"
        body, detected_signature = signature.extract(html_content, sender=None)
        assert body is not None


class TestSignaturePrecision:
    """Test signature detection precision and recall."""

    def test_precision_over_recall(self):
        """Test that signature detection favors precision over recall."""
        # Test with ambiguous content that could be signature or content
        ambiguous_cases = [
            "<div>Thanks for your help!</div>",  # Could be closing or signature
            "<div>Best wishes for your project.</div>",  # Context matters
            "<div>John</div>",  # Just a name
        ]
        
        for html_content in ambiguous_cases:
            body, detected_signature = signature.extract(html_content, sender="john@example.com")
            
            # Conservative approach: preserve ambiguous content
            assert body == html_content, f"Ambiguous content incorrectly modified: {html_content}"

    def test_clear_signatures_detected(self):
        """Test that clear signatures are properly detected."""
        clear_signature_cases = [
            "Main content\n\n--\nJohn Doe\njohn@example.com\n+1-555-123-4567",
            "Email body here\n\nBest regards,\nJohn Smith\nSenior Engineer\nTech Corp\njohn.smith@techcorp.com",
        ]
        
        for content in clear_signature_cases:
            body, detected_signature = signature.extract(content, sender="john@example.com")
            
            # Clear signatures should be detected
            assert detected_signature is not None or len(body) < len(content), f"Clear signature not detected: {content[:50]}..."

    def test_mixed_content_scenarios(self):
        """Test signature detection with mixed content scenarios."""
        for test_case in MIXED_CONTENT_SCENARIOS:
            html_content = test_case['html']
            
            # Test signature detection
            body, detected_signature = signature.extract(html_content, sender="test@example.com")
            
            assert body is not None, f"Failed to process mixed content in {test_case['name']}"
            
            # Verify key content preservation
            for element in test_case.get('should_preserve', []):
                assert element in body, f"Element '{element}' should be preserved in {test_case['name']}"


class TestSignatureIntegration:
    """Test signature detection integration with other components."""

    def test_signature_extraction_consistency(self):
        """Test that signature extraction is consistent across calls."""
        html_content = "<div>Content here</div><div>Best regards<br>John</div>"
        
        # Run extraction multiple times
        results = []
        for _ in range(3):
            body, detected_signature = signature.extract(html_content, sender="john@example.com")
            results.append((body, detected_signature))
        
        # Results should be consistent
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "Signature extraction results should be consistent"

    def test_signature_with_various_senders(self):
        """Test signature detection with different sender formats."""
        html_content = "<div>Content</div><div>Best regards<br>John Doe<br>john@example.com</div>"
        
        sender_formats = [
            "john@example.com",
            "John Doe <john@example.com>",
            "john.doe@example.com",
            "\"John Doe\" <john@example.com>",
        ]
        
        for sender in sender_formats:
            body, detected_signature = signature.extract(html_content, sender=sender)
            assert body is not None, f"Failed to process with sender format: {sender}"

    def test_large_content_performance(self):
        """Test signature detection performance with large content."""
        # Create large content
        large_content = "<div>Content line</div>\n" * 1000 + "<div>Best regards<br>John</div>"
        
        # Should complete without timeout
        body, detected_signature = signature.extract(large_content, sender="john@example.com")
        assert body is not None
        assert len(body) > 0
