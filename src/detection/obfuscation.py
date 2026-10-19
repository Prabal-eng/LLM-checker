"""Undo cheap character-level disguise before a text is checked (Ralph R07-1 to R11-1).

Edits that leave text readable fooled the detector: lookalike letters from other scripts, invisible
characters inside words, compatibility forms and accents that render as plain letters. Five reviews
each found new characters, so this module reads text back instead of listing cases, and abstains on
what it can't read back:

- Whitespace controls become spaces, and C1 bytes are read as the cp1252 punctuation they usually are.
- Every format, unassigned, private-use, surrogate, default-ignorable and control character goes.
- Accents are read back: each Latin letter is decomposed and its combining marks dropped, so a dot
  below or a macron can't disguise a word, and a mark with no letter under it is removed (Ralph R11-1).
  A few accents in ordinary text raise no note.
- A run of near-invisible spaces inside a word, between two letters, is removed.
- Compatibility forms whose NFKC form is ASCII are folded (roman numerals, fullwidth and mathematical
  letters, ligatures, special spaces, leader dots). Only letter-like folds count as disguise.
- Lookalike letters from Cyrillic, Greek and Armenian, Latin-script lookalikes, stroked Latin letters
  and letter-shaped symbols become plain letters.
- What can't be read back is counted: unusual Latin letters or other-script letters inside a word
  that also holds ASCII letters, and symbols or non-Latin digits touching ASCII letters. The service
  abstains when that is dense (three or more, or 2% of words), so ordinary text with one Greek letter
  still gets checked while a disguised text never gets a confident label.

The counts go into the result's evidence: disguise is itself worth a reviewer's attention.
"""

import re
import unicodedata
from dataclasses import dataclass

_LOOKALIKES = {
    # Cyrillic
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s", "ԁ": "d", "һ": "h", "ӏ": "l", "ԛ": "q", "ԝ": "w", "ү": "y",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "Х": "X", "У": "Y", "І": "I", "Ј": "J", "Ѕ": "S", "Ԛ": "Q", "Ԝ": "W",
    # Greek
    "α": "a", "ο": "o", "ι": "i", "ν": "v", "ρ": "p", "υ": "u", "κ": "k", "χ": "x", "ϲ": "c",
    "ϳ": "j", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Armenian
    "օ": "o", "ս": "u", "ո": "n", "հ": "h", "ց": "g", "զ": "q",
}
_LOOKALIKES.update({"\u03bc": "u", "\u03b2": "b"})  # micro and beta in "5 \u03bcg", "A\u03b2"
_WORD = re.compile(r"\w+", re.UNICODE)
# Default-ignorable code points outside category Cf (Unicode DerivedCoreProperties), including plane 14.
_INVISIBLE = re.compile("[\u034f\u115f\u1160\u17b4\u17b5\u180b-\u180f\u3164\ufe00-\ufe0f\uffa0\U000e0000-\U000e0fff]")
_SPACE_CONTROLS = re.compile("[\x0b\x0c\x1c-\x1f]")
_OTHER_CONTROLS = re.compile("[\x00-\x08\x0e-\x1b\x7f]")
_C1 = re.compile("[\x80-\x9f]")
_SPLIT_SPACE = re.compile(r"(?<=[^\W\d_])[\u2006\u2008\u2009\u200a\u205f]+(?=[^\W\d_])")
_REMOVED_CATEGORIES = frozenset({"Cf", "Cn", "Co", "Cs"})
# Latin letters that carry a stroke instead of a combining mark, so decomposition can't read them back.
_STROKED = {"\u00f8": "o", "\u00d8": "O", "\u0142": "l", "\u0141": "L", "\u0111": "d", "\u0110": "D", "\u0127": "h", "\u0126": "H", "\u0140": "l", "\u013f": "L", "\u0167": "t", "\u0166": "T"}
# Symbols drawn like letters (APL alpha, iota, rho, omega; the estimated sign; small element of).
_SYMBOL_LOOKALIKES = str.maketrans({"\u237a": "a", "\u2373": "i", "\u2374": "p", "\u2375": "w", "\u212e": "e", "\u220a": "e"})
# Symbols and marks that sit next to letters in ordinary text: Latin-1 signs (degree, copyright), euro,
# trade mark, numero, per mille, arrows, and emoji.
_SYMBOL_OK = frozenset(range(0x00A0, 0x00C0)) | frozenset(range(0x2190, 0x2200)) | frozenset(range(0x2600, 0x27C0)) \
    | frozenset(range(0x2B00, 0x2C00)) | frozenset(range(0x1F000, 0x1FB00)) | {0x20AC, 0x2116, 0x2120, 0x2122, 0x2030}
_LATIN_LOOKALIKES = {
    "LATIN SMALL LETTER DOTLESS I": "i", "LATIN SMALL LETTER DOTLESS J": "j", "LATIN SMALL LETTER SCRIPT G": "g",
    "LATIN SMALL LETTER ALPHA": "a", "LATIN CAPITAL LETTER ALPHA": "A", "LATIN SMALL LETTER LONG S": "s",
    "LATIN SMALL LETTER IOTA": "i", "LATIN CAPITAL LETTER IOTA": "I",
    "LATIN LETTER DENTAL CLICK": "l", "LATIN SMALL LETTER R WITH FISHHOOK": "r", "LATIN SMALL LETTER KRA": "k",
}
_SMALL_CAPITAL = re.compile(r"LATIN (?:CAPITAL )?LETTER SMALL CAPITAL ([A-Z])")
# Letters that stand for units or ordinals: micro sign, feminine and masculine ordinal indicators.
_ORDINARY_SIGNS = frozenset({0x00AA, 0x00B5, 0x00BA})
# Amendment 13 (Ralph R12-4): punctuation other scripts put between words (katakana middle dots,
# the word-separator dot, runic, Ethiopic, Syriac and Aegean dividers) glued a whole text into one token
# for the model, with no note. These read back as spaces (the Greek ano teleia decomposes to the middle
# dot, which the model's view already reads as a space). Other non-ASCII punctuation between two Latin
# letters can't be read back and counts toward abstaining; quotes, apostrophes, dashes, the ellipsis and
# the middle dot are ordinary there.
_WORD_SEPARATORS = str.maketrans({c: " " for c in "\u0701\u0702\u0703\u1361\u16eb\u16ec\u16ed\u2058\u2059"
                                  "\u205a\u205b\u205c\u205d\u205e\u2e30\u2e31\u30fb\uff65\U00010100\U00010101\U00010102"})
_PUNCTUATION_OK = frozenset({0x00B7, 0x2026, 0x2027})
# Abstain when this many unreadable words or symbols remain, or this share of the words.
ABSTAIN_COUNT, ABSTAIN_SHARE = 3, 0.02


@dataclass(frozen=True)
class Deobfuscated:
    text: str
    format_removed: int = 0
    letters_mapped: int = 0
    mixed_script_words: int = 0
    unusual_letter_words: int = 0
    odd_symbols: int = 0
    word_count: int = 0

    @property
    def residual(self) -> int:
        """Words and symbols normalization could not read back."""
        return self.mixed_script_words + self.unusual_letter_words + self.odd_symbols

    @property
    def abstain(self) -> bool:
        """Dense unreadable content: the service abstains instead of scoring (Ralph R10-1, R11-3)."""
        return self.residual >= ABSTAIN_COUNT or (self.residual > 0 and self.residual >= ABSTAIN_SHARE * self.word_count)

    @property
    def changed(self) -> bool:
        return bool(self.format_removed or self.letters_mapped or self.residual)


def _latin_lookalike(ch: str) -> str:
    name = unicodedata.name(ch, "")
    small_capital = _SMALL_CAPITAL.fullmatch(name)
    if small_capital:
        return small_capital.group(1) if "CAPITAL LETTER SMALL" in name else small_capital.group(1).lower()
    return _LATIN_LOOKALIKES.get(name, "")


def _is_latin(ch: str) -> bool:
    return ch.isascii() or unicodedata.name(ch, "").startswith("LATIN")


def _cp1252(match: re.Match) -> str:
    try:
        return bytes([ord(match.group(0))]).decode("cp1252")
    except UnicodeDecodeError:
        return ""


def _read_accents(text: str) -> tuple:
    """Drop combining marks on Latin letters and marks with no letter under them; keep the marks of
    other scripts. Returns (text, marks folded on Latin letters, orphan marks removed)."""
    out, folded, orphans = [], 0, 0
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.category(ch) in ("Mn", "Me"):
            base = next((c for c in reversed(out) if unicodedata.category(c) not in ("Mn", "Me")), "")
            if base.isalpha() and _is_latin(base):
                folded += 1
                continue
            if not base.isalpha():
                orphans += 1
                continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out)), folded, orphans


def deobfuscate(text: str) -> Deobfuscated:
    text = _SPACE_CONTROLS.sub(" ", text)
    decoded = _C1.sub(_cp1252, text)
    removed = len(text) - len(decoded)  # C1 bytes cp1252 leaves undefined
    text = decoded
    kept = [c for c in text if unicodedata.category(c) not in _REMOVED_CATEGORIES
            and not _INVISIBLE.match(c) and not _OTHER_CONTROLS.match(c)]
    removed += len(text) - len(kept)
    text, folded, orphans = _read_accents("".join(kept))
    removed += orphans
    joined = _SPLIT_SPACE.sub("", text)
    removed += len(text) - len(joined)
    text = joined
    separated = text.translate(_WORD_SEPARATORS)
    separators = sum(1 for a, b in zip(text, separated) if a != b)
    text = separated

    out, mapped = [], separators
    for ch in text:
        if not ch.isascii() and unicodedata.category(ch)[0] in "LNZP":
            as_ascii = unicodedata.normalize("NFKC", ch)
            if as_ascii != ch and as_ascii.isascii():
                mapped += unicodedata.category(ch)[0] in "LN" and any(c.isalpha() for c in as_ascii)
                out.append(as_ascii)
                continue
        out.append(ch)
    text = "".join(out)
    symbols = text.translate(_SYMBOL_LOOKALIKES)
    mapped += sum(1 for a, b in zip(text, symbols) if a != b)
    text = symbols
    ascii_letters = sum(c.isascii() and c.isalpha() for c in text)
    if folded > max(3, 0.02 * ascii_letters):  # dense accents are disguise; a few are ordinary
        mapped += folded
    mixed = unusual = 0

    def fix(match: re.Match) -> str:
        nonlocal mapped, mixed, unusual
        word = match.group(0)
        if word.isascii():
            return word
        has_ascii = any(c.isascii() and c.isalpha() for c in word)
        letters, odd = [], False
        for ch in word:
            if not ch.isalpha() or ch.isascii() or ord(ch) in _ORDINARY_SIGNS:
                letters.append(ch)
            elif _latin_lookalike(ch):
                letters.append(_latin_lookalike(ch))
            elif ch in _STROKED:
                letters.append(_STROKED[ch])
            elif ch in _LOOKALIKES:
                letters.append(_LOOKALIKES[ch])
            elif _is_latin(ch):
                letters.append(ch)
                odd = odd or unicodedata.name(ch, "").startswith("LATIN") and ord(ch) > 0x017F
            else:
                # A genuinely non-Latin word stays as written. One that also holds ASCII letters is
                # mixed-script: disguise that normalization can't read back (Ralph R09-2, R10-1).
                mixed += has_ascii
                return word
        unusual += odd and has_ascii
        mapped += sum(1 for a, b in zip(word, letters) if a != b)
        return "".join(letters)

    text = _WORD.sub(fix, text)
    odd_symbols = 0
    for i, ch in enumerate(text):
        if ch.isascii() or ord(ch) in _SYMBOL_OK:
            continue
        category = unicodedata.category(ch)
        if category[0] == "S" or (category == "Nd"):
            neighbours = text[i - 1:i] + text[i + 1:i + 2]
            odd_symbols += any(c.isascii() and c.isalpha() for c in neighbours)
        elif category[0] == "P" and category not in ("Pd", "Pi", "Pf") and ord(ch) not in _PUNCTUATION_OK:
            before, after = text[i - 1:i], text[i + 1:i + 2]
            odd_symbols += before.isascii() and before.isalpha() and after.isascii() and after.isalpha()
    return Deobfuscated(text, removed, mapped, mixed, unusual, odd_symbols, len(_WORD.findall(text)))
