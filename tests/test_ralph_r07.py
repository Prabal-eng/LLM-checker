"""Regression tests for the Ralph review of Increments 4 and 4b (R07-1 to R07-9)."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.authorship import benchmark, registry
from src.authorship.detector import AuthorshipModelDetector
from src.detection import (AuthorshipService, BandRule, DecisionPolicy, DetectionDecision, TextContext,
                           combine_column_results)
from src.detection.obfuscation import deobfuscate
from src.detectors.base import BaseDetector, DetectionResult
from tests.test_authorship_registry import _record, _with_short

POLICY = DecisionPolicy(min_word_count=25, bands=(
    BandRule(25, 100, human_max_score=0.05, ai_min_score=0.99),
    BandRule(100, None, human_max_score=0.30, ai_min_score=0.80),
))
SENTENCE = "the report is clear and the chart is fine but the legend is cut off near the edge."


def _text(n):
    words = SENTENCE.split()
    return " ".join((words * (n // len(words) + 1))[:n])


class KeywordDetector(BaseDetector):
    """Scores 0.95 when the text mentions 'robots', else 0.1; records what it was given."""

    def __init__(self):
        super().__init__(0.5)
        self.seen = []

    def detect(self, text):
        self.seen.append(text)
        score = 0.95 if "robots" in text else 0.1
        return DetectionResult(provider="k", ai_score=score, is_flagged=score > 0.5, analysis={})


def _service(detector):
    return AuthorshipService(detector, policy=POLICY, model_validated=True, detector_version="t",
                             require_english=True)


# --- R07-1: character-level disguise ----------------------------------------------------------

def test_lookalike_letters_and_invisible_characters_are_undone():
    plain = "Robots write the whole report and the chart."
    disguised = plain.replace("o", "о").replace("a", "а").replace("e", "е")
    assert deobfuscate(disguised).text == plain
    assert deobfuscate("ro​bots and soft­hyphen").text == "robots and softhyphen"
    assert deobfuscate("zero​width").text == "zerowidth"
    russian = "Привет, как дела?"
    assert deobfuscate(russian).text == russian  # real Cyrillic words stay as written


def test_disguised_ai_text_is_scored_as_the_reader_sees_it():
    text = _text(140) + " robots"
    disguised = text.replace("robots", "r​ob​ots").replace("e", "е").replace("o", "о")
    detector = KeywordDetector()
    result = _service(detector).check_text(disguised)
    assert detector.seen[-1] == text
    assert result.decision is DetectionDecision.LIKELY_AI_GENERATED
    marker = next(e for e in result.evidence if e.code == "OBFUSCATION_NORMALIZED")
    assert marker.value["format_characters_removed"] == 2 and marker.value["lookalike_letters_mapped"] > 0


# --- R07-2: long texts are read in full -------------------------------------------------------

class WindowDetector(BaseDetector):
    """Returns one window per 160 words, with fixed per-window scores."""

    def __init__(self, scores, inside=None):
        super().__init__(0.5)
        self.scores, self.inside = scores, inside

    def detect(self, text):
        windows = [{"text": _text(160), "score": s, "in_distribution": True if self.inside is None else self.inside[i],
                    "ood_distance": 0.1} for i, s in enumerate(self.scores)]
        return DetectionResult(provider="w", ai_score=self.scores[0], is_flagged=False,
                               analysis={"windows": windows, "in_distribution": all(w["in_distribution"] for w in windows)})


@pytest.mark.parametrize(("scores", "decision", "reason"), [
    ([0.1, 0.95, 0.95, 0.95], DetectionDecision.INCONCLUSIVE, "MIXED_AUTHORSHIP_SIGNALS"),  # human opening, AI body
    ([0.95, 0.1, 0.5, 0.5], DetectionDecision.INCONCLUSIVE, "MIXED_AUTHORSHIP_SIGNALS"),   # AI opening, human body
    ([0.95, 0.5, 0.5, 0.95], DetectionDecision.LIKELY_AI_GENERATED, "CALIBRATED_HIGH_SCORE"),
    ([0.1, 0.2, 0.5], DetectionDecision.LIKELY_HUMAN_WRITTEN, "CALIBRATED_LOW_SCORE"),
    ([0.5, 0.95, 0.95, 0.95], DetectionDecision.INCONCLUSIVE, "SCORE_IN_ABSTENTION_BAND"),  # later windows never create a label
])
def test_the_opening_decides_and_later_windows_can_only_veto(scores, decision, reason):
    result = _service(WindowDetector(scores)).check_text(_text(700))
    assert result.decision is decision and result.reason_codes == [reason]
    assert next(e for e in result.evidence if e.code == "WINDOWED_SCORING").value["windows"] == len(scores)


def test_any_out_of_distribution_window_abstains_and_flag_only_never_labels_human():
    ood = _service(WindowDetector([0.1, 0.1], inside=[True, False])).check_text(_text(400))
    assert ood.reason_codes == ["UNSUPPORTED_DOMAIN"]
    rater = _service(WindowDetector([0.1, 0.1])).check_text(_text(400), context=TextContext.RATER_COMMENT)
    assert rater.decision is DetectionDecision.INCONCLUSIVE and rater.reason_codes == ["BELOW_AI_THRESHOLD_FLAG_ONLY"]
    vetoed = _service(WindowDetector([0.95, 0.1])).check_text(_text(400), context=TextContext.RATER_COMMENT)
    assert vetoed.reason_codes == ["MIXED_AUTHORSHIP_SIGNALS"]  # a human-reading window still vetoes a flag


def test_detector_windows_only_texts_longer_than_its_window(monkeypatch):
    detector = AuthorshipModelDetector("unused", platt=(1.0, 0.0), max_tokens=64, text_view="collapse_whitespace")
    monkeypatch.setattr(detector, "_ensure_loaded", lambda: None)
    detector._tokenizer = lambda text, **kwargs: {"input_ids": text.split()}

    def fake_scores(texts):
        return np.array([5.0 if "robots" in t else -5.0 for t in texts]), np.eye(len(texts), 4, dtype=np.float32)

    monkeypatch.setattr(detector, "raw_and_embeddings", fake_scores)
    short, long_text = _text(40), _text(400) + " robots robots robots"
    results = detector.detect_batch([short, long_text])
    assert "windows" not in results[0].analysis
    windows = results[1].analysis["windows"]
    assert windows[0]["text"] == long_text  # the opening is the text itself, read from its start
    assert len(windows) >= 3 and windows[-1]["score"] > 0.99


# --- R07-5: a flagged cell flags its row ------------------------------------------------------

def test_a_cell_flagged_ai_flags_the_row():
    service = _service(KeywordDetector())
    human_cell = service.check_text(_text(150), context=TextContext.RATER_COMMENT)
    ai_cell = service.check_text(_text(150) + " robots", context=TextContext.RATER_COMMENT)
    row = service.check_text(_text(300), context=TextContext.RATER_COMMENT)
    assert row.decision is DetectionDecision.INCONCLUSIVE
    result = combine_column_results({"a": human_cell, "b": ai_cell}, row_result=row)
    assert result.decision is DetectionDecision.LIKELY_AI_GENERATED
    assert [e.code for e in result.evidence[:2]] == ["ROW_COMBINED_TEXT", "CELL_FLAGGED_AI"]
    assert result.likely_ai_columns == ["b"]


# --- R07-4 and R07-8: gate integrity ----------------------------------------------------------

def test_directory_digest_catches_tokenizer_and_same_size_rewrites(tmp_path):
    record = _record(tmp_path)
    art = tmp_path / "models" / "m" / "epoch-1"
    record["artifact"]["files_sha256"] = registry.directory_sha256(art)
    assert registry.verify(record, tmp_path)[0]
    (art / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert registry.verify(record, tmp_path) == (False, "model config or tokenizer files changed")
    (art / "tokenizer.json").unlink()
    weights = art / "model.safetensors"
    stat = weights.stat()
    weights.write_bytes(b"WEIGHTS")  # same size as b"weights"
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert registry.verify(record, tmp_path) == (False, "model artifact hash mismatch")


def test_missing_short_model_never_borrows_its_thresholds(tmp_path):
    record = _with_short(tmp_path, _record(tmp_path))
    record["calibration"]["main_bands"] = [dict(b, model="main") for b in record["calibration"]["bands"]]
    record["calibration"]["bands"] = [dict(record["calibration"]["bands"][0], supported=True, lower_words=25,
                                           upper_words=100, ai_min=0.8, human_max=0.1, model="short"),
                                      record["calibration"]["bands"][1]]
    (tmp_path / "models" / "s" / "epoch-0" / "model.safetensors").unlink()
    service = registry.service_from_record(record, tmp_path, validated=True)
    assert not hasattr(service.detector, "short")
    assert service.policy.band_for(30) is None  # the main model's own 25-99 band is unsupported
    assert registry.verify_files(record, tmp_path) == (False, "short-text model artifact missing")


# --- R07-9: an interrupted run is still recorded ---------------------------------------------

def test_a_run_is_recorded_before_any_text_is_scored(tmp_path, monkeypatch):
    pytest.importorskip("torch")  # CI installs requirements.txt, which leaves torch out
    path = tmp_path / "registry.json"
    registry.save({"model_id": "m", "policy_version": "v", "benchmark": None, "locked_benchmark_history": []}, path)
    locked = pd.DataFrame({c: ["x"] for c in benchmark._COLUMNS}).assign(split="test", label="ai")
    monkeypatch.setattr(benchmark.pd, "read_parquet", lambda *a, **k: locked.copy())
    monkeypatch.setattr(benchmark, "read_lock", lambda _: {"sha256": "h"})
    monkeypatch.setattr(benchmark, "locked_hash", lambda _: "h")
    monkeypatch.setattr(registry, "verify_files", lambda record: (True, "files verified"))

    class Crashing:
        model_validated = False

        def check_batch(self, texts, context):
            raise RuntimeError("power loss")

    monkeypatch.setattr(registry, "build_service", lambda _path: Crashing())
    with pytest.raises(RuntimeError):
        benchmark.run(registry_path=path, report_path=tmp_path / "benchmark.json")
    history = json.loads(path.read_text(encoding="utf-8"))["locked_benchmark_history"]
    assert history[-1]["status"] == "started" and history[-1]["corpus_version"] == benchmark.VERSION
    with pytest.raises(SystemExit, match="already ran"):
        benchmark.run(registry_path=path, report_path=tmp_path / "benchmark.json")


def test_an_existing_report_counts_as_a_run(tmp_path):
    path = tmp_path / "registry.json"
    registry.save({"model_id": "m", "policy_version": "v", "benchmark": None, "locked_benchmark_history": []}, path)
    report = tmp_path / "benchmark.json"
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="already ran"):
        benchmark.run(registry_path=path, report_path=report)


# --- R07-3: retired groups stay out of evaluation --------------------------------------------

def test_audit_rejects_evaluation_rows_from_a_retired_test_group():
    from src.corpus.audit import audit
    from src.corpus.schema import Label, LicenceClass, Role
    from src.corpus.sources._common import SourceSpec, make_sample

    spec = SourceSpec(name="raid", licence="MIT", licence_class=LicenceClass.PERMISSIVE)
    sample = make_sample(spec, text="an attacked variant", label=Label.AI, role=Role.STRESS, record_id="1",
                         domain="d", group_key="1", generator="gpt-4", transform="attack:paraphrase")
    df = pd.DataFrame([sample.model_dump(mode="json")]).assign(split="stress")
    report = audit(df, None, retired_groups=frozenset({"raid:1"}))
    assert any("earlier version's locked-test group" in e for e in report.errors)


def test_invisible_marks_and_latin_lookalikes_are_undone():
    """Ralph R08-1: combining joiners, variation selectors and Latin-script lookalike letters
    (dotless i, script g, small capitals) passed through R07-1's fix untouched."""
    from src.detection.obfuscation import deobfuscate
    marked = deobfuscate("the\u034f plan\ufe0f works\u180b well")
    assert marked.text == "the plan works well" and marked.format_removed == 3
    looks = deobfuscate("\u028f\u1d0fu \u0251re \u0261ood \ua731tudents \u0131n \u1d04lass")
    assert looks.text == "you are good students in class" and looks.letters_mapped == 7
    kept = deobfuscate("We met at a caf\u00e9 and ordered a cr\u00eape near the old office downtown")
    assert kept.text == "We met at a cafe and ordered a crepe near the old office downtown"
    assert not kept.changed  # a few ordinary accents are read back without a note (Amendment 9)


def test_more_cheap_disguises_are_undone_or_noted():
    """Ralph R09-2: ligatures, leader dots, hair spaces inside words, Latin-named lookalikes, Armenian
    lookalikes, unassigned default-ignorables and control characters."""
    from src.detection.obfuscation import deobfuscate
    assert deobfuscate("the \ufb01le \ufb02ows\u2024").text == "the file flows."
    assert deobfuscate("wo\u200ard pl\u200aan").text == "word plan"
    assert deobfuscate("\u01c0ate \u027eun \u0578ow \u0570ere").text == "late run now here"
    cleaned = deobfuscate("the\U000e0080 plan\x90 works")
    assert cleaned.text == "the plan works" and cleaned.format_removed == 2
    assert deobfuscate("x\u180ey").format_removed == 1  # R09-8: counted once
    shielded = deobfuscate("h\u0435llo w\u043erld\u0436")  # a Cyrillic letter keeps one word unmapped
    assert shielded.changed and shielded.mixed_script_words == 1


# --- R10-1: disguise is closed as a class ------------------------------------------------------

def test_compatibility_forms_and_invisible_classes_are_folded():
    """Ralph R10-1: small roman numerals, unassigned and default-ignorable characters, runs of thin
    spaces inside words, and combining marks on ASCII letters all passed through."""
    cases = {
        "\u2170\u217c\u217c go": "ill go",            # small roman numerals read as letters
        "the\ufff0 plan\u2065 works": "the plan works",  # unassigned / default-ignorable
        "wo\u200a\u200ard pl\u205fan": "word plan",     # a run of thin spaces, a math space
    }
    for disguised, plain in cases.items():
        cleaned = deobfuscate(disguised)
        assert cleaned.text == plain and cleaned.changed, (disguised, cleaned)
    # A dot above on ASCII i is read back like any accent (Amendment 9); only a dense scatter is noted.
    assert deobfuscate("thi\u0307s i\u0307s").text == "this is"
    assert deobfuscate(" ".join(["thi\u0307s i\u0307s a li\u0307ttle ti\u0307p"] * 5)).changed


def test_ordinary_typography_raises_no_disguise_note():
    """Ralph R10-4: an ellipsis, curly apostrophes, the narrow no-break space in '10:30 AM' and thin
    digit groups are ordinary text, not disguise."""
    cleaned = deobfuscate("Wait\u2026 it\u2019s 10:30\u202fAM on day 1\u2009000 of caf\u00e9 life")
    assert not cleaned.changed, cleaned


def test_whitespace_controls_space_words_and_cp1252_bytes_are_read():
    """Ralph R10-4: VT, FF and the file separators are whitespace, and C1 bytes are usually
    mis-decoded cp1252 punctuation; deleting them glued words."""
    cleaned = deobfuscate("one\x0btwo\x0cthree\x1cfour it\x92s")
    assert cleaned.text == "one two three four it\u2019s" and not cleaned.changed


def test_residual_disguise_abstains_instead_of_scoring():
    """Ralph R10-1: what normalization can't read back (unusual Latin letters or other scripts inside
    English words) makes the service abstain with DISGUISE_SUSPECTED, never label."""
    text = _text(140) + " robots"
    for letter in ("\u028b", "\u026f", "\u0185", "\u0436"):  # v with hook, turned m, tone six, Cyrillic zhe
        disguised = text.replace("chart", "ch" + letter + "rt")
        result = _service(KeywordDetector()).check_text(disguised)
        assert result.decision is DetectionDecision.INCONCLUSIVE, letter
        assert "DISGUISE_SUSPECTED" in result.reason_codes, letter
    assert deobfuscate("\u0138ind \ua7aet").text == "kind It"  # kra and capital small-capital I map
    assert deobfuscate("5\u00b5m na\u00efve").changed is False  # micro sign and one accent are ordinary


# --- R11-1: diacritics are read back, not judged -----------------------------------------------

def test_diacritics_and_orphan_marks_are_read_back_as_plain_letters():
    """Ralph R11-1: a mark that composes with its letter (a + dot below) became an 'ordinary' letter,
    and a mark after a space was kept; both turned confident AI into Likely human."""
    cases = {
        "the pla\u0323n wo\u0323rks": "the plan works",       # composing dot below
        "the pl\u1ea1n w\u1ecdrks": "the plan works",         # the same, precomposed
        "th\u0117 pl\u0101n": "the plan",                       # dot above and macron (Latin Extended-A)
        "wo\u200a\u0307rd pl\u0331an": "word plan",            # a mark after a hair space; macron below
    }
    for disguised, plain in cases.items():
        assert deobfuscate(disguised).text == plain, disguised


def test_symbol_lookalikes_map_and_foreign_digits_inside_words_count():
    """Ralph R11-1: symbols such as APL alpha and the estimated sign, and non-Latin digits, sat
    inside words where the word-level check never looked."""
    assert deobfuscate("b\u237ad m\u212e").text == "bad me"
    digits = deobfuscate("g\u0966od w\u0966rk f\u0966r m\u0966re")
    assert digits.residual >= 3


def test_lone_foreign_letters_do_not_force_abstention():
    """Ralph R11-3: '5 \u03bcg' or 'A\u03b2' in ordinary scientific text abstained. Abstention now needs
    dense unreadable content: at least three words or 2% of them."""
    text = _text(140) + " robots and the A\u03b2 level was 5 \u03bcg per dose"
    result = _service(KeywordDetector()).check_text(text)
    assert result.decision is DetectionDecision.LIKELY_AI_GENERATED
    assert "DISGUISE_SUSPECTED" not in result.reason_codes


def test_narrow_no_break_space_between_words_stays_a_space():
    """Ralph R11-3: U+202F between letters glued words ('returnedfroma holiday')."""
    assert deobfuscate("returned\u202ffrom a holiday").text == "returned from a holiday"
