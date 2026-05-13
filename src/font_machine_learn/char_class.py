from __future__ import annotations

import unicodedata


def classify_char(char: str | None) -> str:
    if not char:
        return "unmapped"
    codepoint = ord(char)
    if (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0x30000 <= codepoint <= 0x3134F
    ):
        return "cjk"
    if 0x3040 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF:
        return "kana"
    if "A" <= char <= "Z" or "a" <= char <= "z":
        return "latin"
    if char.isdigit():
        return "digit"
    category = unicodedata.category(char)
    if category.startswith("P"):
        return "punct"
    return "symbol"


def classify_glyph(chars: list[str]) -> str:
    if not chars:
        return "unmapped"
    classes = [classify_char(char) for char in chars if char]
    if not classes:
        return "unmapped"
    if "cjk" in classes:
        return "cjk"
    if "kana" in classes:
        return "kana"
    return classes[0]
