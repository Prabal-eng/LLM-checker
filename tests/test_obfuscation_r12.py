"""Ralph R12-4: other scripts' word separators are read back as spaces; other punctuation glued between
Latin letters counts as unreadable (Amendment 13)."""
import pytest

from src.detection.obfuscation import deobfuscate

SENTENCE = "the quick brown fox jumps over the lazy dog and then it runs far away into the deep dark forest again"


@pytest.mark.parametrize("sep", ["\u30fb", "\uff65", "\u2e31", "\u16eb", "\u205a", "\u2e30", "\u0701", "\U00010101",
                                 "\u1361"])
def test_word_separators_read_back_as_spaces(sep):
    got = deobfuscate(sep.join(SENTENCE.split()))
    assert got.text == SENTENCE
    assert got.changed and not got.abstain


@pytest.mark.parametrize("sep", ["\u30fb", "\u2e31", "\u0387", "\u00b7", "\u2027"])
def test_the_model_reads_separated_words_as_words(sep):
    from src.corpus.text import MODEL_TEXT_VIEW, text_view
    got = deobfuscate(sep.join(SENTENCE.split()))
    assert text_view(MODEL_TEXT_VIEW)(got.text) == SENTENCE and not got.abstain


def test_other_punctuation_glued_between_letters_abstains():
    got = deobfuscate("\u203b".join(SENTENCE.split()))
    assert got.odd_symbols >= 3 and got.abstain


@pytest.mark.parametrize("text", ["I don\u2019t know\u2026 wait\u2026what? \u201cfine\u201d he said",
                                  "a well\u2010known state\u2014of\u2014the\u2014art tool", "\u00bfqu\u00e9? \u00a1s\u00ed!"])
def test_ordinary_punctuation_does_not_count(text):
    assert deobfuscate(text).odd_symbols == 0
