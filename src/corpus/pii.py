"""Pattern-based PII redaction applied identically to human and AI text.

Covers emails, phone numbers, payment card numbers, and IPv4 addresses. Names and
postal addresses are not detectable by pattern and are out of scope here.
"""

# ponytail: D2 pattern-only redaction, add an NER pass before ingesting new in-house text.

import re
from collections import Counter

_PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "IP": re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"),
    "PHONE": re.compile(
        r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\w)"
        r"|(?<![\w+])\+\d{1,3}(?:[\s.-]\d{2,5}){2,4}(?!\w)"
    ),
    "CARD": re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
}


def _luhn(candidate: str) -> bool:
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def find(text: str) -> Counter:
    """Count PII matches by kind without changing the text."""
    hits: Counter = Counter()
    for kind, pattern in _PATTERNS.items():
        for match in pattern.finditer(text):
            if kind != "CARD" or _luhn(match.group()):
                hits[kind] += 1
    return hits


def redact(text: str) -> tuple[str, Counter]:
    """Replace every match with a typed placeholder such as ``[EMAIL]``.

    Passes repeat until nothing changes, because removing one match can expose an
    adjacent one (e.g. two phone numbers written back to back).
    """
    hits: Counter = Counter()

    def _sub(match: re.Match, kind: str) -> str:
        if kind == "CARD" and not _luhn(match.group()):
            return match.group()
        hits[kind] += 1
        return f"[{kind}]"

    while True:
        before = text
        for kind, pattern in _PATTERNS.items():
            text = pattern.sub(lambda m, kind=kind: _sub(m, kind), text)
        if text == before:
            return text, hits
