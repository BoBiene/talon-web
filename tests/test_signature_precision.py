# Test zur Präzision der Signaturerkennung - hilft bei der Identifikation von False Positives/Negatives
import json
from talon.web.bootstrap import app, _remove_html_signature_patterns

def test_should_remove_signatures():
    """Test cases where signatures SHOULD be removed"""
    
    # Test 1: Clear signature with job title and email
    html_clear_sig = '''
    <div>Please review the document.</div>
    <div><br></div>
    <div>Best regards</div>
    <table><tbody><tr><td>
        <p>John Doe</p>
        <p>Operations Engineer | Customer Solutions</p>
        <p>john.doe@example.com | www.example.com</p>
    </td></tr></tbody></table>
    '''
    
    # Test 2: Simple salutation signature
    html_simple_sig = '''
    <div>Thanks for your help with this issue.</div>
    <div><br></div>
    <div>Mit freundlichen Grüßen</div>
    <div>Jane Smith</div>
    '''
    
    # Test 3: Gmail-style signature
    html_gmail_sig = '''
    <div>Let me know if you have questions.</div>
    <div><br></div>
    <div class="gmail_signature">
        <p>Bob Wilson<br>
        Senior Developer<br>
        bob@company.com</p>
    </div>
    '''
    
    test_cases = [
        (html_clear_sig, "Clear signature with job title"),
        (html_simple_sig, "Simple German salutation"),
        (html_gmail_sig, "Gmail signature block")
    ]
    
    for html, description in test_cases:
        cleaned = _remove_html_signature_patterns(html)
        print(f"\n{description}:")
        print(f"Original length: {len(html)}")
        print(f"Cleaned length: {len(cleaned)}")
        print(f"Removed: {len(html) - len(cleaned)} chars")
        
        # Should remove signature content
        assert "john.doe@example.com" not in cleaned or "Operations Engineer" not in cleaned, f"Failed to remove signature in: {description}"
        # Should keep main content
        assert any(phrase in cleaned for phrase in ["Please review", "Thanks for your help", "Let me know"]), f"Removed too much content in: {description}"


def test_should_not_remove_content():
    """Test cases where content should NOT be removed (avoid false positives)"""
    
    # Test 1: "Best" in normal context
    html_best_context = '''
    <div>What is the best approach for this problem?</div>
    <div>I think the best solution would be to use method A.</div>
    <div>Best practices suggest we should validate input.</div>
    '''
    
    # Test 2: Tables with actual content (not signatures)
    html_content_table = '''
    <div>Here are the server details:</div>
    <table>
        <tr><td>Server Name</td><td>SERVER-123</td></tr>
        <tr><td>IP Address</td><td>10.0.0.1</td></tr>
        <tr><td>Status</td><td>Running</td></tr>
    </table>
    <div>Please verify these settings.</div>
    '''
    
    # Test 3: Email addresses in content (not signatures)
    html_email_content = '''
    <div>Please contact john.doe@example.com for technical support.</div>
    <div>The Operations Engineer will help you with setup.</div>
    <div>For billing questions, reach out to billing@company.com</div>
    '''
    
    # Test 4: "Thanks" in normal context
    html_thanks_content = '''
    <div>Thanks to your feedback, we improved the system.</div>
    <div>The API now supports more endpoints, thanks to the new architecture.</div>
    '''
    
    test_cases = [
        (html_best_context, "'Best' in normal text"),
        (html_content_table, "Data table (not signature)"),
        (html_email_content, "Email addresses in content"),
        (html_thanks_content, "'Thanks' in normal context")
    ]
    
    for html, description in test_cases:
        cleaned = _remove_html_signature_patterns(html)
        print(f"\n{description}:")
        print(f"Original length: {len(html)}")
        print(f"Cleaned length: {len(cleaned)}")
        print(f"Removed: {len(html) - len(cleaned)} chars")
        
        # Should not remove much content (less than 20% unless it's actually a signature)
        removal_ratio = (len(html) - len(cleaned)) / len(html)
        assert removal_ratio < 0.2, f"Removed too much content ({removal_ratio:.1%}) in: {description}"
        
        # Should keep essential content  
        if "best approach" in html.lower():
            assert "best approach" in cleaned.lower(), f"Removed 'best approach' from normal text"
        if "server details" in html.lower():
            assert "server details" in cleaned.lower(), f"Removed server details table"
        if "contact john.doe@example.com" in html.lower():
            assert "john.doe@example.com" in cleaned.lower(), f"Removed email from normal content"


def test_edge_cases():
    """Test edge cases and problematic patterns"""
    
    # Test 1: Mixed content with signature at end
    html_mixed = '''
    <div>Hello Team,</div>
    <div><br></div>
    <div>Could you please check the server logs for any errors?</div>
    <div>The database shows some connection issues.</div>
    <div><br></div>
    <div>Thanks for your help.</div>
    <div><br></div>
    <div>Best regards</div>
    <table><tbody><tr><td>
        <p>Tech Support</p>
        <p>support@company.com</p>
    </td></tr></tbody></table>
    '''
    
    # Test 2: Short salutation that should be preserved
    html_short_preserve = '''
    <div>We found the best solution for performance optimization.</div>
    <div>Thanks to the new caching system, response times improved.</div>
    '''
    
    test_cases = [
        (html_mixed, "Mixed content with clear signature"),
        (html_short_preserve, "Short words in normal context")
    ]
    
    for html, description in test_cases:
        cleaned = _remove_html_signature_patterns(html)
        print(f"\n{description}:")
        print(f"Original: {html[:100]}...")
        print(f"Cleaned: {cleaned[:100]}...")
        print(f"Removal ratio: {(len(html) - len(cleaned)) / len(html):.1%}")


if __name__ == "__main__":
    print("Testing signature removal precision...")
    test_should_remove_signatures()
    test_should_not_remove_content() 
    test_edge_cases()
    print("\nAll tests completed!")
