"""English check for the support gate, without a language-identification dependency.

English prose is dense in a small set of function words, and other languages written in Latin
script share few of them. Text is treated as English when most letters are Latin and enough
words are English function words. ADR-008: other languages return Inconclusive until they get
their own data, slice, and calibration.
"""

import re
import unicodedata
from dataclasses import dataclass

# High-frequency English function words that are rare as whole words in other Latin-script languages.
_FUNCTION_WORDS = frozenset("""
the of and to in is that it for was on are with as be this by at or from but not have has had
they you we he she his her their its which will would can could should there been were an if
what when who how than then so do does did about into more also our your my them these those
""".split())
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True)
class LanguageCheck:
    is_english: bool
    latin_share: float
    function_word_share: float


def check(text: str, min_latin: float = 0.9, min_function_words: float = 0.12) -> LanguageCheck:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return LanguageCheck(False, 0.0, 0.0)
    latin = sum(1 for c in letters if unicodedata.name(c, "").startswith("LATIN")) / len(letters)
    words = [w.lower() for w in _WORD.findall(text)]
    share = sum(1 for w in words if w in _FUNCTION_WORDS) / len(words) if words else 0.0
    return LanguageCheck(latin >= min_latin and share >= min_function_words, round(latin, 3), round(share, 3))
