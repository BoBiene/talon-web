from __future__ import absolute_import
# Modernized for pytest - removed nose dependencies
import unittest.mock  # replaces mock module
from unittest.mock import *  # provides Mock, patch, etc.

import talon

# pytest-compatible assertion helpers (nose.tools replacements)
def assert_false(val, msg=None):
    assert not val, msg

def assert_true(val, msg=None):
    assert val, msg

def eq_(a, b, msg=None):
    assert a == b, msg

def ok_(val, msg=None):
    assert val, msg

# Pytest-compatible assertion functions to replace nose.tools
def assert_false(val):
    assert not val

def assert_true(val):
    assert val

def eq_(a, b):
    assert a == b

def ok_(val):
    assert val

def assert_equal(a, b):
    assert a == b

def assert_not_equal(a, b):
    assert a != b

def assert_in(item, container):
    assert item in container

def assert_not_in(item, container):
    assert item not in container


EML_MSG_FILENAME = "tests/fixtures/standard_replies/yahoo.eml"
MSG_FILENAME_WITH_BODY_SUFFIX = ("tests/fixtures/signature/emails/P/"
                                 "johndoeexamplecom_body")
EMAILS_DIR = "tests/fixtures/signature/emails"
TMP_DIR = "tests/fixtures/signature/tmp"

STRIPPED = "tests/fixtures/signature/emails/stripped/"
UNICODE_MSG = ("tests/fixtures/signature/emails/P/"
               "unicode_msg")


talon.init()
