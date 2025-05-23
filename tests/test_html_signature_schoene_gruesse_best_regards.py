import pytest
from talon.web.bootstrap import _remove_html_signature_patterns, strip_inline_tags

@pytest.mark.parametrize("html,expected", [
    # Grußformel im Fließtext, KEINE Signatur
    (
        '<div>Das war super, schöne Grüße / Best regards an dein Team!</div>',
        '<div>Das war super, schöne Grüße / Best regards an dein Team!</div>'
    ),
    # Grußformel als Signatur (alleinstehend)
    (
        '<p>Schöne Grüße / Best regards<br>Max Mustermann</p>',
        ''
    ),
    # Grußformel als Signatur (am Block-Anfang, mit <br>)
    (
        '<div>Schöne Grüße / Best regards,<br>Max</div>',
        ''
    ),
    # Grußformel im Fließtext, KEINE Signatur
    (
        '<p>Ich wünsche dir schöne Grüße / Best regards von uns allen.</p>',
        '<p>Ich wünsche dir schöne Grüße / Best regards von uns allen.</p>'
    ),
    # Realistische verschachtelte Signatur (wie aus Outlook kopiert)
    (
        '<div class="elementToProof"><p style="margin:0"><span>Schöne Grüße / Best regards<br>Max Mustermann</span></p></div>',
        '<div class="elementToProof"></div>'
    ),
    # Realistische verschachtelte Fließtext-Variante (darf NICHT entfernt werden)
    (
        '<div class="elementToProof"><p style="margin:0"><span>Das war super, schöne Grüße / Best regards an dein Team!</span></p></div>',
        '<div class="elementToProof"><p style="margin:0">Das war super, schöne Grüße / Best regards an dein Team!</p></div>'
    ),
])
def test_signature_removal_schoene_gruesse_best_regards(html, expected):
    # Nach dem Entfernen der Inline-Tags vergleichen
    result = _remove_html_signature_patterns(html).strip()
    expected_stripped = strip_inline_tags(expected).strip()
    assert result == expected_stripped
