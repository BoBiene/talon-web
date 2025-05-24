"""
Consolidated multi-language and pattern-specific tests
Tests signature detection across different languages and specific patterns.
"""
import pytest
from talon import signature
from tests.fixtures.html_samples import MULTI_LANGUAGE_SIGNATURES


class TestMultiLanguageSignatures:
    """Test signature detection across different languages."""

    def test_german_signature_patterns(self):
        """Test German signature pattern detection."""
        german_signatures = [
            {
                'html': '<p>Mit freundlichen Grüßen<br>Hans Mueller<br>Projektleiter<br>hans@firma.de</p>',
                'should_remove': True,
                'elements_to_remove': ['hans@firma.de', 'Projektleiter']
            },
            {
                'html': '<p>Viele Grüße<br>Anna<br>anna@unternehmen.com</p>',
                'should_remove': True,
                'elements_to_remove': ['anna@unternehmen.com']
            },
            {
                'html': '<p>Beste Grüße<br>Thomas Schmidt<br>Senior Entwickler<br>thomas@techfirma.de</p>',
                'should_remove': True,
                'elements_to_remove': ['thomas@techfirma.de', 'Senior Entwickler']
            },
            {
                'html': '<p>Schöne Grüße<br>Maria<br>Beraterin<br>maria@beratung.com</p>',
                'should_remove': True,
                'elements_to_remove': ['maria@beratung.com', 'Beraterin']
            }
        ]
        
        for test_case in german_signatures:
            result = signature.extract(test_case['html'], "text/html")
            
            if test_case['should_remove']:
                # Professional German signatures should be removed
                for element in test_case['elements_to_remove']:
                    assert element not in result, f"German signature element '{element}' should be removed"
            else:
                # Should be preserved
                assert result == test_case['html']

    def test_german_conservative_signatures(self):
        """Test simple German signatures that should be preserved (conservative approach)."""
        simple_german = [
            '<p>Mit freundlichen Grüßen<br>Hans</p>',
            '<p>Viele Grüße<br>Anna</p>',
            '<p>Schöne Grüße<br>Thomas</p>',
            '<p>Beste Grüße<br>Maria</p>',
        ]
        
        for html_content in simple_german:
            result = signature.extract(html_content, "text/html")
            
            # Simple German signatures should be preserved (conservative approach)
            assert result == html_content, f"Simple German signature should be preserved: {html_content}"

    def test_english_signature_patterns(self):
        """Test English signature pattern detection."""
        english_signatures = [
            {
                'html': '<p>Best regards<br>John Smith<br>Senior Manager<br>john@company.com</p>',
                'should_remove': True,
                'elements_to_remove': ['john@company.com', 'Senior Manager']
            },
            {
                'html': '<p>Kind regards<br>Sarah Johnson<br>Project Lead<br>sarah@enterprise.org</p>',
                'should_remove': True,
                'elements_to_remove': ['sarah@enterprise.org', 'Project Lead']
            },
            {
                'html': '<p>Sincerely<br>Robert Brown<br>Technical Director<br>robert@tech.com</p>',
                'should_remove': True,
                'elements_to_remove': ['robert@tech.com', 'Technical Director']
            },
            {
                'html': '<p>Thanks<br>Emily Davis<br>Consultant<br>emily@consulting.net</p>',
                'should_remove': True,
                'elements_to_remove': ['emily@consulting.net', 'Consultant']
            }
        ]
        
        for test_case in english_signatures:
            result = signature.extract(test_case['html'], "text/html")
            
            if test_case['should_remove']:
                # Professional English signatures should be removed
                for element in test_case['elements_to_remove']:
                    assert element not in result, f"English signature element '{element}' should be removed"

    def test_french_signature_patterns(self):
        """Test French signature pattern detection."""
        french_signatures = [
            {
                'html': '<p>Cordialement<br>Pierre Dubois<br>Directeur<br>pierre@societe.fr</p>',
                'should_remove': True,
                'elements_to_remove': ['pierre@societe.fr', 'Directeur']
            },
            {
                'html': '<p>Bien à vous<br>Marie Martin<br>Responsable<br>marie@entreprise.fr</p>',
                'should_remove': True,
                'elements_to_remove': ['marie@entreprise.fr', 'Responsable']
            }
        ]
        
        for test_case in french_signatures:
            result = signature.extract(test_case['html'], "text/html")
            
            if test_case['should_remove']:
                # Professional French signatures should be removed
                for element in test_case['elements_to_remove']:
                    assert element not in result, f"French signature element '{element}' should be removed"

    def test_spanish_signature_patterns(self):
        """Test Spanish signature pattern detection."""
        spanish_signatures = [
            {
                'html': '<p>Saludos cordiales<br>Carlos González<br>Gerente<br>carlos@empresa.es</p>',
                'should_remove': True,
                'elements_to_remove': ['carlos@empresa.es', 'Gerente']
            },
            {
                'html': '<p>Atentamente<br>Ana López<br>Coordinadora<br>ana@compañia.es</p>',
                'should_remove': True,
                'elements_to_remove': ['ana@compañia.es', 'Coordinadora']
            }
        ]
        
        for test_case in spanish_signatures:
            result = signature.extract(test_case['html'], "text/html")
            
            if test_case['should_remove']:
                # Professional Spanish signatures should be removed
                for element in test_case['elements_to_remove']:
                    assert element not in result, f"Spanish signature element '{element}' should be removed"

    def test_multi_language_combinations(self):
        """Test multi-language signature combinations from fixtures."""
        for test_case in MULTI_LANGUAGE_SIGNATURES:
            result = signature.extract(test_case['html'], "text/html")
            
            if test_case['expected_removed']:
                # Professional multi-language signatures should be removed
                assert result != test_case['html'], f"Multi-language signature not detected in {test_case['name']}"
                
                for element in test_case['should_remove']:
                    assert element not in result, f"Element '{element}' should be removed from {test_case['name']}"
            else:
                # Multi-language content should be preserved
                assert result == test_case['html'], f"Multi-language content incorrectly removed in {test_case['name']}"
                
                for element in test_case['should_preserve']:
                    assert element in result, f"Element '{element}' should be preserved in {test_case['name']}"

    def test_mixed_language_content(self):
        """Test content that mixes languages but isn't a signature."""
        mixed_content = [
            '<div>Meeting tomorrow at 10 AM. Bitte bestätigen Sie den Termin.</div>',
            '<div>Please review the document. Merci beaucoup for your time.</div>',
            '<div>El proyecto está completo. The final report is ready.</div>',
        ]
        
        for html_content in mixed_content:
            result = signature.extract(html_content, "text/html")
            
            # Mixed language content should be preserved
            assert result == html_content, f"Mixed language content should be preserved: {html_content}"


class TestSpecificPatterns:
    """Test specific signature patterns and detection rules."""

    def test_email_pattern_variations(self):
        """Test various email pattern formats."""
        email_patterns = [
            'user@domain.com',
            'user.name@domain.co.uk',
            'user+tag@subdomain.domain.org',
            'user-name@domain-name.net',
            'user123@123domain.info',
        ]
        
        for email in email_patterns:
            # In signature context - should be removed
            signature_html = f'<p>Best regards<br>John Doe<br>Engineer<br>{email}</p>'
            result = signature.extract(signature_html, "text/html")
            assert email not in result, f"Email '{email}' should be removed from signature"
            
            # In content context - should be preserved
            content_html = f'<div>Please contact {email} for support.</div>'
            result = signature.extract(content_html, "text/html")
            assert email in result, f"Email '{email}' should be preserved in content"

    def test_job_title_pattern_variations(self):
        """Test various job title patterns."""
        job_titles = [
            'Senior Software Engineer',
            'Project Manager',
            'VP of Engineering',
            'Chief Technology Officer',
            'Lead Developer',
            'Principal Architect',
            'Director of Operations',
            'Head of Sales',
        ]
        
        for job_title in job_titles:
            # In signature context - should be removed
            signature_html = f'<p>Best regards<br>John Doe<br>{job_title}<br>john@company.com</p>'
            result = signature.extract(signature_html, "text/html")
            assert job_title not in result, f"Job title '{job_title}' should be removed from signature"
            
            # In content context - should be preserved
            content_html = f'<div>We need a new {job_title} for our team.</div>'
            result = signature.extract(content_html, "text/html")
            assert job_title in result, f"Job title '{job_title}' should be preserved in content"

    def test_phone_number_patterns(self):
        """Test phone number pattern detection in signatures."""
        phone_patterns = [
            '+1 (555) 123-4567',
            '555-123-4567',
            '(555) 123-4567',
            '+49 30 12345678',
            '+33 1 23 45 67 89',
        ]
        
        for phone in phone_patterns:
            # In signature context - should be removed
            signature_html = f'<p>Best regards<br>John Doe<br>Engineer<br>{phone}<br>john@company.com</p>'
            result = signature.extract(signature_html, "text/html")
            assert phone not in result, f"Phone number '{phone}' should be removed from signature"

    def test_company_information_patterns(self):
        """Test company information in signatures."""
        company_info = [
            'Acme Corporation',
            'Tech Solutions Inc.',
            'Global Enterprises Ltd.',
            'Innovation Partners LLC',
        ]
        
        for company in company_info:
            # In signature context with other professional elements - should be removed
            signature_html = f'<p>Best regards<br>John Doe<br>Engineer<br>{company}<br>john@company.com</p>'
            result = signature.extract(signature_html, "text/html")
            assert company not in result, f"Company '{company}' should be removed from signature"

    def test_address_patterns_in_signatures(self):
        """Test address pattern detection in signatures."""
        addresses = [
            '123 Main Street, City, State 12345',
            'Building A, Floor 5, Tech Park',
            '1234 Innovation Drive, Suite 100',
        ]
        
        for address in addresses:
            # In signature context - should be removed
            signature_html = f'<p>Best regards<br>John Doe<br>Engineer<br>{address}<br>john@company.com</p>'
            result = signature.extract(signature_html, "text/html")
            assert address not in result, f"Address '{address}' should be removed from signature"

    def test_website_patterns_in_signatures(self):
        """Test website URL pattern detection in signatures."""
        websites = [
            'www.company.com',
            'https://company.com',
            'http://www.example.org',
            'company.co.uk',
        ]
        
        for website in websites:
            # In signature context - should be removed
            signature_html = f'<p>Best regards<br>John Doe<br>Engineer<br>{website}<br>john@company.com</p>'
            result = signature.extract(signature_html, "text/html")
            assert website not in result, f"Website '{website}' should be removed from signature"

    def test_social_media_patterns(self):
        """Test social media handle detection in signatures."""
        social_handles = [
            'LinkedIn: linkedin.com/in/johndoe',
            'Twitter: @johndoe',
            'GitHub: github.com/johndoe',
        ]
        
        for handle in social_handles:
            # In signature context - should be removed
            signature_html = f'<p>Best regards<br>John Doe<br>Engineer<br>{handle}<br>john@company.com</p>'
            result = signature.extract(signature_html, "text/html")
            assert handle not in result, f"Social handle '{handle}' should be removed from signature"


class TestSignatureClassDetection:
    """Test detection of signature-specific CSS classes and attributes."""

    def test_gmail_signature_class(self):
        """Test Gmail signature class detection."""
        gmail_signatures = [
            '<div class="gmail_signature">John Doe<br>john@gmail.com</div>',
            '<div class="gmail_signature" dir="ltr">Jane Smith<br>jane@company.com</div>',
            '<span class="gmail_signature"><b>Bob Wilson</b><br>bob@enterprise.org</span>',
        ]
        
        for signature_html in gmail_signatures:
            result = signature.extract(signature_html, "text/html")
            
            # Gmail signatures should be removed
            assert '@' not in result or result == '', f"Gmail signature should be removed: {signature_html}"

    def test_outlook_signature_class(self):
        """Test Outlook signature class detection."""
        outlook_signatures = [
            '<div class="elementToProof">Sarah Connor<br>sarah@company.com</div>',
            '<p class="elementToProof">Mike Davis<br>Engineer<br>mike@tech.com</p>',
            '<div class="elementToProof" style="font-family: Calibri;">Lisa Johnson<br>lisa@business.net</div>',
        ]
        
        for signature_html in outlook_signatures:
            result = signature.extract(signature_html, "text/html")
            
            # Outlook signatures should be removed
            assert '@' not in result or result == '', f"Outlook signature should be removed: {signature_html}"

    def test_mozilla_signature_class(self):
        """Test Mozilla signature class detection."""
        mozilla_signatures = [
            '<div class="moz-signature">Tom Anderson<br>tom@mozilla.com</div>',
            '<pre class="moz-signature">-- <br>Alex Thompson<br>alex@company.com</pre>',
        ]
        
        for signature_html in mozilla_signatures:
            result = signature.extract(signature_html, "text/html")
            
            # Mozilla signatures should be removed
            assert '@' not in result or result == '', f"Mozilla signature should be removed: {signature_html}"

    def test_generic_signature_class(self):
        """Test generic signature class detection."""
        generic_signatures = [
            '<div class="signature">Generic Signature<br>user@email.com</div>',
            '<section class="email-signature">Company Signature<br>contact@company.com</section>',
            '<footer class="sig">Footer Signature<br>footer@domain.com</footer>',
        ]
        
        for signature_html in generic_signatures:
            result = signature.extract(signature_html, "text/html")
            
            # Generic signatures should be removed
            assert '@' not in result or result == '', f"Generic signature should be removed: {signature_html}"

    def test_signature_id_detection(self):
        """Test signature ID attribute detection."""
        id_signatures = [
            '<div id="signature">ID Signature<br>id@example.com</div>',
            '<div id="email-signature">Email ID Signature<br>email@test.com</div>',
            '<div id="sig">Short ID Signature<br>short@domain.org</div>',
        ]
        
        for signature_html in id_signatures:
            result = signature.extract(signature_html, "text/html")
            
            # ID-based signatures should be removed
            assert '@' not in result or result == '', f"ID signature should be removed: {signature_html}"


class TestLanguageSpecificEdgeCases:
    """Test language-specific edge cases and corner scenarios."""

    def test_german_umlauts_handling(self):
        """Test proper handling of German umlauts in signatures."""
        umlaut_signatures = [
            '<p>Schöne Grüße<br>Björn Müller<br>björn@müller.de</p>',
            '<p>Mit freundlichen Grüßen<br>Jürgen Weiß<br>jürgen@weiß.com</p>',
        ]
        
        for signature_html in umlaut_signatures:
            result = signature.extract(signature_html, "text/html")
            
            # Should handle umlauts properly and remove professional signatures
            assert 'ü' not in result or 'ö' not in result or '@' not in result

    def test_accent_characters_handling(self):
        """Test proper handling of accent characters in signatures."""
        accent_signatures = [
            '<p>Cordialement<br>François Ménard<br>françois@société.fr</p>',
            '<p>Saludos<br>José María<br>josé@compañía.es</p>',
        ]
        
        for signature_html in accent_signatures:
            result = signature.extract(signature_html, "text/html")
            
            # Should handle accents properly and remove professional signatures
            assert '@' not in result, f"Professional signature with accents should be removed: {signature_html}"

    def test_mixed_script_signatures(self):
        """Test signatures with mixed scripts (Latin + non-Latin)."""
        mixed_script = [
            '<p>Best regards / 最好的问候<br>John Zhang<br>john@company.com</p>',
            '<p>Regards / С уважением<br>Ivan Petrov<br>ivan@company.ru</p>',
        ]
        
        for signature_html in mixed_script:
            result = signature.extract(signature_html, "text/html")
            
            # Should handle mixed scripts and remove professional signatures
            assert '@' not in result, f"Mixed script signature should be removed: {signature_html}"
