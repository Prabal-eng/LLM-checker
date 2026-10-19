"""Load-time trims of text a generator added around its answer (Amendment 13, from Ralph R12-3).

MAGA's AI rows open with the model announcing its output ("Here is the body of the Reddit post:",
"Certainly! Here's an improved version of your abstract:") 13% of the time, against 0.04% of its human
rows; MIRAGE's 5%. A quarter of OpenTuringBench's rows (all AI) carry the prompt's metadata lines
("Date: November 26, 2013", "Category: WEDDINGS"). Left in, they teach a shortcut, the class Amendments 8
and 9 trimmed from the generated answers. Loaders apply these to core rows whatever their label."""
import re

_META_NOUN = re.compile(r"(?i)\b(?:version|article|body|response|abstract|review|post|rewrite|rewritten|draft"
                        r"|summary|text|essay|story|answer|piece|sample|revision|edit|paragraph|reply|comment"
                        r"|tweet|email|letter|blog)\b")
_PREAMBLE = re.compile(r"^\s*(?:(?i:certainly|sure|okay|ok|absolutely|of course)\s*[!,.]\s*)?"
                       r"(?i:here(?:'s|\u2019s|\s+is|\s+are)|below\s+(?:is|are))\b"
                       r"(?P<body>[^\n:]{0,200}?):[ \t]*(?:-{3,}[ \t]*)?\s*")


def trim_preamble(text: str) -> str:
    """text without a leading "Here's the <article/version/body...>:" announcement, its next letter
    capitalised; unchanged if nothing would be left."""
    match = _PREAMBLE.match(text)
    if not match or not _META_NOUN.search(match.group("body")) or match.end() >= len(text):
        return text
    rest = text[match.end():]
    return rest[:1].upper() + rest[1:]


_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DATE_LINE = re.compile(rf"(?i)^\s*\**\s*date\s*\**\s*:\s*\**\s*(?:{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}"
                        rf"|\d{{1,2}}\s+{_MONTH}\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}})\s*\**\s*$")
# A category is up to six capitalised words ("WORLD NEWS", "U.S. News", "Culture & Arts"), so prose after
# "Category:" stays.
_CATEGORY_LINE = re.compile(r"^\s*\**\s*(?i:category)\s*\**\s*:\s*\**\s*[A-Z0-9][\w.'&/,-]*"
                            r"(?:[ \t]+(?:&|[A-Z0-9][\w.'&/,-]*)){0,5}\**\s*$")
_PLACEHOLDER_BYLINE = re.compile(r"(?i)^\s*by\s*:?\s*\[[^\]]*\]\s*$")
_LABELLED_TITLE = re.compile(r"^\s*\**\s*(?i:headline|subheadline|title)\s*\**\s*:\s*\**\s*(?P<title>\S.*?)\**\s*$")
HEADER_LINES = 8


def trim_header(text: str) -> str:
    """text without the prompt's metadata lines (dated "Date:", upper-case "Category:", a "By [Your Name]"
    placeholder) among its first lines; "HEADLINE:" and "SUBHEADLINE:" labels go and their titles stay."""
    lines = text.split("\n")
    head, rest = lines[:HEADER_LINES], lines[HEADER_LINES:]
    kept = []
    for line in head:
        if _DATE_LINE.match(line) or _CATEGORY_LINE.match(line) or _PLACEHOLDER_BYLINE.match(line):
            continue
        title = _LABELLED_TITLE.match(line)
        kept.append(title.group("title") if title else line)
    if kept == head:
        return text
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(kept + rest)).strip()
    return out or text
