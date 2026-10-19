"""Trigger sentences and words (spec 2026-10-01, section 2; words 2026-10-02): what takes away the largest share
of a row's AI signal (its log-odds) when removed."""
import math
import re

import numpy as np
import pytest

from src.authorship.detector import LengthRoutedDetector
from src.detectors import GPTZeroDetector
from src.detectors.base import BaseDetector, DetectionResult
from src.report.triggers import JOIN, TriggerFinder, scorer_for, split_sentences, without

PLANT = "We must delve into the rich tapestry of this response."


def logit(p):
    return math.log(p / (1 - p))


class Weighted:
    """Duck-typed like AuthorshipModelDetector: log-odds = base + the weight of each word present, so removing a
    word or sentence takes away exactly its weights."""
    platt = (1.0, 0.0)

    def __init__(self, base, **weights):
        self.base, self.weights, self.calls = base, weights, 0

    def raw_and_embeddings(self, texts):
        self.calls += len(texts)
        raw = [self.base + sum(self.weights.get(w, 0.0) for w in re.findall(r"[a-z]+", t.lower())) for t in texts]
        return np.array(raw), np.zeros((len(texts), 2))


class Planted(BaseDetector):
    """0.95 when the planted sentence is present, else 0.20."""

    def detect(self, text):
        score = 0.95 if "rich tapestry" in text else 0.20
        return DetectionResult(provider="planted", ai_score=score, is_flagged=score >= 0.5)


def test_sentences_split_at_sentence_ends_and_line_breaks_and_map_back_to_cells():
    segs = [("first", "The answer is right. It cites the source!"), ("second", "- point one\n- point two")]
    got = split_sentences(segs)
    assert [(s.column, s.text) for s in got] == [
        ("first", "The answer is right."), ("first", "It cites the source!"),
        ("second", "- point one"), ("second", "- point two")]
    for s in got:
        assert dict(segs)[s.column][s.start:s.end] == s.text
    assert [s.word_start for s in got] == [0, 4, 8, 11]


def test_no_split_inside_abbreviations_or_lowercase_continuations():
    got = split_sentences([("c", "It follows e.g. the rubric. then it ends")])
    assert [s.text for s in got] == ["It follows e.g. the rubric. then it ends"]


def test_a_text_without_punctuation_is_one_sentence_and_empty_cells_give_none():
    assert [s.text for s in split_sentences([("c", "no punctuation at all here")])] == ["no punctuation at all here"]
    assert split_sentences([("c", ""), ("d", "   ")]) == []


def test_removing_a_sentence_keeps_the_rest_of_the_row_joined_as_the_pipeline_joins_it():
    segs = [("a", "One. Two."), ("b", "Three.")]
    first = split_sentences(segs)[0]
    assert without(segs, first) == "Two." + JOIN + "Three."
    last = split_sentences(segs)[2]
    assert without(segs, last) == "One. Two."


def test_the_planted_sentence_is_found_with_its_share_of_the_ai_signal():
    result = TriggerFinder(Planted()).explain([("why", f"The rating is fair. {PLANT} It matches the rubric.")])
    assert result.kind == "sentences"
    (sentence, share), = result.sentences
    assert sentence.text == PLANT and share == pytest.approx((logit(0.95) - logit(0.20)) / logit(0.95))
    assert result.baseline == pytest.approx(0.95)
    assert len(result.drops) == 3
    assert result.summary().startswith("Without it: 95% -> 20% (100% of the AI signal)")


def test_near_the_top_of_the_scale_a_small_percent_move_is_a_large_share_of_the_signal():
    """99.3% -> 97.1% is under 3 points, but 30% of the AI signal: the sentence is a trigger."""
    text = "The rating is fair overall. We must delve into this. It matches the rubric well."
    result = TriggerFinder(Weighted(3.5, delve=1.5)).explain([("why", text)])
    assert result.kind == "sentences"
    (sentence, share), = result.sentences
    assert sentence.text == "We must delve into this." and share == pytest.approx(0.30)
    assert result.summary() == 'Without it: 99% -> 97% (30% of the AI signal)\nWords: "delve" (30%)'


def test_words_that_raise_the_score_are_named_with_their_share():
    text = "The rating is fair overall. We must delve into this. It matches the rubric well."
    result = TriggerFinder(Weighted(3.5, delve=1.5)).explain([("why", text)])
    (word,) = result.words
    assert (word.column, word.text, text[word.start:word.end]) == ("why", "delve", "delve")
    assert word.share == pytest.approx(0.30)


def test_neighbouring_trigger_words_form_one_phrase_scored_as_a_whole():
    text = "The rating is fair. We must look at the rich tapestry here. It ends."
    result = TriggerFinder(Weighted(3.0, rich=0.4, tapestry=0.6)).explain([("why", text)])
    (phrase,) = result.words
    assert phrase.text == "rich tapestry" and text[phrase.start:phrase.end] == "rich tapestry"
    assert phrase.share == pytest.approx(0.25)


def test_a_phrase_of_only_common_words_is_tried_but_not_shown():
    text = "The rating is fair. We must delve into the answer. It ends."
    result = TriggerFinder(Weighted(3.0, delve=0.6, the=0.6)).explain([("why", text)])
    assert [w.text for w in result.words] == ["delve"]
    assert "the" in [w.text.lower() for w in result.word_drops]


def test_a_signal_spread_across_sentences_names_its_words_instead():
    text = ("Alpha one here. Bravo two here. Charlie three here. Delta four here. Echo five here. "
            "Foxtrot six here.")
    weights = dict(alpha=0.5, bravo=0.5, charlie=0.5, delta=0.5, echo=0.5, foxtrot=0.5)
    result = TriggerFinder(Weighted(2.0, **weights)).explain([("why", text)])
    assert result.kind == "spread" and result.sentences == []
    assert len(result.words) == 3                      # the words of the 3 sentences that matter most
    assert result.summary().startswith('Words: "') and "(10%)" in result.summary()


def test_when_no_sentence_or_word_stands_out_the_trigger_says_so():
    class Flat(Planted):
        def detect(self, text):
            return DetectionResult(provider="flat", ai_score=0.9, is_flagged=True)

    result = TriggerFinder(Flat()).explain([("why", "One thing. Another thing. A third thing.")])
    assert result.kind == "spread" and result.sentences == [] and result.words == []
    assert result.summary() == "No single sentence or word stands out: the AI signal is spread across the text"


def test_a_single_sentence_names_its_words():
    result = TriggerFinder(Planted()).explain([("why", PLANT)])
    assert result.kind == "single" and [w.text for w in result.words] == ["rich tapestry"]
    assert result.summary() == 'Words: "rich tapestry" (100%)'


def test_words_are_capped_and_can_be_switched_off():
    text = "Some words in a sentence that keeps going for a while. And another sentence here too."
    model = Weighted(3.0, words=0.5)
    result = TriggerFinder(model, max_words=4).explain([("why", text)])
    assert len(result.word_drops) <= 4
    model.calls = 0
    result = TriggerFinder(model, words=False).explain([("why", text)])
    assert result.words == [] and result.word_drops == [] and model.calls == 3   # the text and its 2 sentences


def test_words_past_the_model_window_are_never_tried():
    text = "Lead words here. " + " ".join(f"Filler number {i} is here." for i in range(10)) + " Late delve here."
    result = TriggerFinder(Weighted(3.0, delve=1.0), window_words=20).explain([("why", text)])
    assert all(w.text != "delve" for w in result.word_drops)


def test_a_text_re_scored_at_or_under_fifty_percent_is_not_explained():
    result = TriggerFinder(Weighted(-1.0)).explain([("why", "One thing. Another thing.")])
    assert result.kind == "skipped" and "50%" in result.note


def test_only_sentences_inside_the_model_window_are_tried_and_at_most_max_sentences():
    long_text = " ".join(f"Sentence number {i} is here." for i in range(40)) + " " + PLANT  # 5 words each; scores high
    result = TriggerFinder(Planted(), max_sentences=12, window_words=30).explain([("c", long_text)])
    assert len(result.drops) == 6                       # sentences starting at words 0..25
    result = TriggerFinder(Planted(), max_sentences=3).explain([("c", long_text)])
    assert len(result.drops) == 3


def test_paid_apis_are_never_called():
    assert scorer_for(GPTZeroDetector(api_key="x"), "text") is None
    assert TriggerFinder(GPTZeroDetector(api_key="x")).explain([("c", "One. Two.")]).kind == "skipped"


def test_the_routed_detector_scores_with_the_model_that_scored_the_row():
    class Model(BaseDetector):
        def __init__(self, name):
            super().__init__()
            self.name, self.calls = name, 0

        def detect(self, text):
            self.calls += 1
            return DetectionResult(provider=self.name, ai_score=0.5, is_flagged=False)

    short, main = Model("short"), Model("main")
    routed = LengthRoutedDetector(main, short, [(25, 100)])
    scorer_for(routed, " ".join(["word"] * 30))(["a", "b"])       # 30 words: the short model's range
    assert (short.calls, main.calls) == (2, 0)
    scorer_for(routed, " ".join(["word"] * 150))(["a"])
    assert main.calls == 1


def test_the_model_path_gives_log_odds_from_the_models_own_probability_map():
    class FakeModel:  # duck-typed like AuthorshipModelDetector: raw_and_embeddings and platt
        platt = (2.0, -1.0)

        def raw_and_embeddings(self, texts):
            return np.array([0.0, 3.0]), np.zeros((len(texts), 2))

    assert list(scorer_for(FakeModel(), "text")(["a", "b"])) == [-1.0, 5.0]


def test_other_detectors_give_the_log_odds_of_their_score():
    assert scorer_for(Planted(), "text")(["rich tapestry"])[0] == pytest.approx(logit(0.95))
