"""
Shared pytest configuration and fixtures for talon-web tests
"""
import pytest
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import talon
from talon.web.bootstrap import app

# Initialize talon
talon.init()


@pytest.fixture
def test_client():
    """Flask test client fixture"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_html_content():
    """Sample HTML content for testing"""
    return """
    <html>
    <body>
    <h1>Test Email</h1>
    <p>This is the main email content that should be preserved.</p>
    <p>Please review the attached documents.</p>
    <hr>
    <p>Best regards<br>
    John Doe<br>
    Senior Engineer<br>
    john.doe@company.com</p>
    </body>
    </html>
    """


@pytest.fixture
def base_url():
    """Base URL for API testing"""
    return "http://localhost:5505"


@pytest.fixture
def api_headers():
    """Standard API headers"""
    return {'Content-Type': 'application/json'}
