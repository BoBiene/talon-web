import pytest
from talon.web.bootstrap import _remove_html_signature_patterns

@pytest.mark.parametrize("html,expected", [
    # Grußformel im Fließtext, KEINE Signatur
    (
        '<p>Bitte richte meine besten Grüße aus an deinen Kollegen Max!</p>',
        '<p>Bitte richte meine besten Grüße aus an deinen Kollegen Max!</p>'
    ),
    # Grußformel im Fließtext, KEINE Signatur
    (
        '<div>Ich wünsche dir ein schönes Wochenende und sende viele Grüße!</div>',
        '<div>Ich wünsche dir ein schönes Wochenende und sende viele Grüße!</div>'
    ),
    # Grußformel als Signatur (alleinstehend)
    (
        '<p>Viele Grüße<br>Max Mustermann</p>',
        ''
    ),
    # Short-Salutation als Signatur (alleinstehend)
    (
        '<p>VG,<br>Max</p>',
        ''
    ),
    # Short-Salutation im Fließtext, KEINE Signatur
    (
        '<p>Das ist ok für mich, VG Max</p>',
        '<p>Das ist ok für mich, VG Max</p>'
    ),
    # Short-Salutation mit Komma im Fließtext, KEINE Signatur
    (
        '<p>Das ist ok für mich, VG, Max</p>',
        '<p>Das ist ok für mich, VG, Max</p>'
    ),
    # Englische Grußformel im Fließtext, KEINE Signatur
    (
        '<div>Please send my best regards to your team.</div>',
        '<div>Please send my best regards to your team.</div>'
    ),
    # Englische Grußformel als Signatur
    (
        '<p>Best regards,<br>John</p>',
        ''
    ),
])
def test_signature_removal_does_not_remove_salutation_in_text(html, expected):
    assert _remove_html_signature_patterns(html).strip() == expected.strip()
