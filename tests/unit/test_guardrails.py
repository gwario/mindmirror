"""Unit tests for SecurityGuardrail (tool execution purity fencing) and ContentSanitizer."""

import pytest
from mindmirror.security.guardrails import ContentSanitizer, SecurityGuardrail, ToolPurity


def test_guardrail_allows_read_only_tool_on_speculative_branch():
    """Validates that read-only tools (e.g. web search, file read) are permitted during speculative branch exploration."""
    guardrail = SecurityGuardrail()
    assert guardrail.validate_tool_call("web_search", is_speculative=True) is True
    assert guardrail.validate_tool_call("read_file", is_speculative=True) is True
    assert guardrail.validate_tool_call("recall_memories", is_speculative=True) is True


def test_guardrail_blocks_mutating_tool_on_speculative_branch():
    """Validates that mutating tools (e.g. delete_file, execute_command) raise PermissionError if called speculatively."""
    guardrail = SecurityGuardrail()

    with pytest.raises(PermissionError) as exc_info:
        guardrail.validate_tool_call("delete_file", is_speculative=True)
    assert "Speculative branch attempted to execute mutating tool 'delete_file'" in str(exc_info.value)

    with pytest.raises(PermissionError):
        guardrail.validate_tool_call("execute_command", is_speculative=True)


def test_guardrail_allows_mutating_tool_on_confirmed_winning_branch():
    """Validates that mutating tools are permitted when not speculative (is_speculative=False)."""
    guardrail = SecurityGuardrail()
    assert guardrail.validate_tool_call("delete_file", is_speculative=False) is True
    assert guardrail.validate_tool_call("remember_fact", is_speculative=False) is True


def test_guardrail_default_safe_for_unregistered_tools():
    """Validates fail-safe behavior: any unregistered tool is assumed MUTATING by default."""
    guardrail = SecurityGuardrail()
    # Unregistered tool "send_sms"
    with pytest.raises(PermissionError):
        guardrail.validate_tool_call("send_sms", is_speculative=True)

    # Allowed when committed
    assert guardrail.validate_tool_call("send_sms", is_speculative=False) is True


def test_content_sanitizer_wraps_in_untrusted_xml_tags():
    """Validates that ContentSanitizer wraps payload in <external_untrusted_context> tags."""
    raw = "The capital of France is Paris."
    sanitized = ContentSanitizer.sanitize(raw, source="web_search")

    assert '<external_untrusted_context source="web_search">' in sanitized
    assert "The capital of France is Paris." in sanitized
    assert "</external_untrusted_context>" in sanitized


def test_content_sanitizer_strips_ansi_escapes():
    """Validates that terminal escape sequences are stripped to prevent terminal tampering."""
    dirty = "\x1b[31;1mDangerous colored text\x1b[0m and \x1b[2Jclear screen"
    sanitized = ContentSanitizer.sanitize(dirty)

    assert "\x1b" not in sanitized
    assert "Dangerous colored text and clear screen" in sanitized


def test_content_sanitizer_strips_zero_width_unicode():
    """Validates that zero-width unicode spaces used in prompt injection / steganography are stripped."""
    # Contains \u200B (zero width space) and \uFEFF (zero width no-break space)
    injected = "Normal\u200BText\uFEFFHere\x00Null"
    sanitized = ContentSanitizer.sanitize(injected)

    assert "\u200B" not in sanitized
    assert "\uFEFF" not in sanitized
    assert "\x00" not in sanitized


def test_content_sanitizer_truncates_oversized_payloads():
    """Validates that outputs longer than max_chars are safely truncated."""
    long_content = "A" * 5000
    sanitized = ContentSanitizer.sanitize(long_content, max_chars=1000)

    assert "[truncated at 1000 chars]" in sanitized
    assert len(sanitized) < 1200
