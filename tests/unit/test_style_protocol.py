"""Unit tests for the Emotional Style Tag Protocol (GS2).

Contract protocol:
1. All TTT responses must start with [NEUTRAL], [EXCITED], [SERIOUS], or [LAZY].
2. TTSComponent parses and strips the style tag from spoken text so it is not synthesised.
3. Default style fallback applies when the model omits a tag.
"""

import pytest
from mindmirror.tts.component import parse_style_tag


@pytest.mark.parametrize(
    "input_text, expected_style, expected_clean",
    [
        ("[NEUTRAL] System checks passed.", "NEUTRAL", "System checks passed."),
        ("[EXCITED] We are ready to roll!", "EXCITED", "We are ready to roll!"),
        ("[SERIOUS] Critical warning on channel 3.", "SERIOUS", "Critical warning on channel 3."),
        ("[LAZY] Maybe tomorrow mate.", "LAZY", "Maybe tomorrow mate."),
        # Case insensitivity
        ("[excited] Hooray!", "EXCITED", "Hooray!"),
        ("[lazy]   Nap time.", "LAZY", "Nap time."),
        # Leading whitespace before tag
        ("   [SERIOUS] Security alert.", "SERIOUS", "Security alert."),
    ],
)
def test_style_tag_parsing_success(input_text: str, expected_style: str, expected_clean: str):
    """Validates that parse_style_tag correctly identifies valid style tags and strips them."""
    style, clean = parse_style_tag(input_text)
    assert style == expected_style
    assert clean == expected_clean


def test_style_tag_fallback_on_omitted_tag():
    """Validates that a message without a style tag defaults to NEUTRAL and preserves entire text."""
    raw = "Good morning Mario, how can I help you today?"
    style, clean = parse_style_tag(raw)
    assert style == "NEUTRAL"
    assert clean == raw


def test_style_tag_unknown_tag_fallback():
    """Validates that an unrecognized tag like [CONFUSED] is not parsed as a valid emotion and defaults to NEUTRAL."""
    raw = "[CONFUSED] What was that?"
    style, clean = parse_style_tag(raw)
    assert style == "NEUTRAL"
    assert clean == raw


def test_style_tag_empty_text():
    """Validates that empty or whitespace strings return NEUTRAL and empty clean text."""
    style, clean = parse_style_tag("   ")
    assert style == "NEUTRAL"
    assert clean == ""
