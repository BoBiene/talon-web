#!/usr/bin/env python3
"""
Quick test script to verify the talon-web application works correctly
after the dependency updates and fixes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_imports():
    """Test that all main imports work correctly."""
    try:
        import talon
        from talon import signature, quotations
        import charset_normalizer  # New dependency
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic talon functionality."""
    try:
        import talon
        from talon import signature, quotations
        
        # Initialize talon
        talon.init()
        
        # Test signature extraction
        test_email = """
        Hello,
        
        This is a test email.
        
        Best regards,
        John Doe
        CEO, Example Corp
        john@example.com
        """
        
        text, sig = signature.extract(test_email, sender="john@example.com")
        print(f"✅ Signature extraction successful")
        print(f"   Text: {len(text)} chars")
        print(f"   Signature: {'Found' if sig else 'Not found'}")
        
        # Test quotations extraction
        reply_text = quotations.extract_from_plain(test_email)
        print(f"✅ Quotations extraction successful")
        print(f"   Reply text: {len(reply_text)} chars")
        
        return True
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_flask_health():
    """Test Flask health endpoint."""
    try:
        from talon.web.bootstrap import app
        
        with app.test_client() as client:
            response = client.get('/health')
            print(f"✅ Health endpoint test successful")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.get_json()}")
            return response.status_code == 200
    except Exception as e:
        print(f"❌ Flask health test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing talon-web after dependency updates...\n")
    
    success = True
    success &= test_imports()
    success &= test_basic_functionality() 
    success &= test_flask_health()
    
    print(f"\n{'🎉 All tests passed!' if success else '❌ Some tests failed!'}")
    sys.exit(0 if success else 1)
