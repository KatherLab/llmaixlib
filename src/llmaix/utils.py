def string_is_empty_or_garbage(s: str) -> bool:
    """Check if a string is empty, contains only whitespace, or scanner artifacts.

    This function detects various forms of "garbage" text that might be extracted
    from scanned PDFs, including:
      - Empty strings
      - Whitespace-only strings (spaces, tabs, etc.)
      - Strings with only linebreaks
      - Strings with only control characters
      - Common scanner artifacts like isolated dots, dashes, or repeated symbols
      - Patterns of whitespace with random punctuation

    Args:
        s: Input string from a potentially scanned PDF document

    Returns:
        bool: True if the string is empty or contains only noise, False otherwise
    """
    if not s:
        return True

    # Check if string contains only whitespace or control characters
    if all(c.isspace() or ord(c) < 32 for c in s):
        return True

    # Remove all whitespace and check if remaining content is meaningful
    stripped = "".join(s.split())

    # Empty after stripping
    if not stripped:
        return True

    # Check for repetitive patterns (like '...', '---', etc.)
    if len(set(stripped)) <= 2 and len(stripped) > 3:
        # Allow for one or two unique characters, but requires multiple instances
        return True

    # Check for isolated punctuation and symbols commonly added by scanners
    if all(c in ".,-_=+*/\\|:;#@!?~^()[]{}'\"`<>" for c in stripped):
        return True

    # Check for single-character noise
    if len(stripped) <= 2:
        return True

    # Consider strings with too low text-to-whitespace ratio as noise
    # This catches patterns like "- - - - -" or ". . . . ." that might have meaning
    # but are more likely scanner artifacts
    if len(stripped) < len(s) / 5 and len(s) > 10:
        return True

    return False
