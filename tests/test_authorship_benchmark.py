"""Locked-benchmark pass criteria, pseudo-rows, and the one-shot guard."""

import pandas as pd
import pytest

from src.authorship import benchmark, registry


def _test_split(false_ai=0, humans=1000, ais=1000, inconclusive_ai=100):
    rows = [{"source_dataset": "s", "y": 0, "decision": "likely_human_written", "reason": "CALIBRATED_LOW_SCORE"}] * (humans - false_ai)
    rows += [{"source_dataset": "s", "y": 0, "decision": "likely_ai_generated", "reason": "CALIBRATED_HIGH_SCORE"}] * false_ai
    rows += [{"source_dataset": "s", "y": 1, "decision": "likely_ai_generated", "reason": "CALIBRATED_HIGH_SCORE"}] * (ais - inconclusive_ai)
    rows += [{"source_dataset": "s", "y": 1, "decision": "inconclusive", "reason": "SCORE_IN_ABSTENTION_BAND"}] * inconclusive_ai
    return pd.DataFrame(rows)


def test_gate_passes_a_clean_slice():
    passed, failures = benchmark.gate(_test_split())
    assert passed and failures == []


def test_gate_fails_when_false_ai_bound_exceeds_one_percent():
    passed, failures = benchmark.gate(_test_split(false_ai=10))
    assert not passed and "false-AI" in failures[0]


def test_gate_fails_on_low_coverage():
    passed, failures = benchmark.gate(_test_split(inconclusive_ai=900, humans=300, ais=1000).assign(
        decision=lambda d: d["decision"].where(d["y"] == 1, "inconclusive"),
        reason=lambda d: d["reason"].where(d["y"] == 1, "SCORE_IN_ABSTENTION_BAND")))
    assert not passed and any("coverage" in f for f in failures)


def test_unsupported_text_does_not_count_against_coverage():
    df = pd.concat([_test_split(), pd.DataFrame([{"source_dataset": "s", "y": 0, "decision": "inconclusive",
                                                  "reason": "UNSUPPORTED_DOMAIN"}] * 5000)])
    assert benchmark.gate(df)[0]


def test_pseudo_rows_group_each_raters_texts_in_sixes():
    df = pd.DataFrame({"group_key": ["r1"] * 7 + ["r2"] * 2, "source_record_id": [f"row:{i}" for i in range(9)],
                       "text": [f"t{i}" for i in range(9)], "sample_id": [f"id{i}" for i in range(9)]})
    rows = benchmark.pseudo_rows(df)
    assert [len(ids) for ids in rows["ids"]] == [6, 1, 2]
    assert all(len({i[2:] for i in ids}) == len(ids) for ids in rows["ids"])


def test_benchmark_refuses_a_second_run_without_a_reason(tmp_path):
    path = tmp_path / "registry.json"
    registry.save({"model_id": "m", "policy_version": "v", "benchmark": {"passed": False}}, path)
    with pytest.raises(SystemExit, match="already ran"):
        benchmark.run(registry_path=path)


def test_a_new_model_still_needs_a_reason_once_the_locked_splits_were_seen(tmp_path):
    path = tmp_path / "registry.json"
    registry.save({"model_id": "m2", "policy_version": "v", "benchmark": None,
                   "locked_benchmark_history": [{"model_id": "m1", "passed": False,
                                                 "corpus_version": benchmark.VERSION}]}, path)
    with pytest.raises(SystemExit, match="already ran"):
        benchmark.run(registry_path=path)


def test_runs_on_an_earlier_corpus_version_do_not_block_a_fresh_test(tmp_path, monkeypatch):
    path = tmp_path / "registry.json"
    registry.save({"model_id": "m3", "policy_version": "v", "benchmark": None,
                   "locked_benchmark_history": [{"model_id": "m1", "passed": False}]}, path)  # untagged = v1
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "no-corpus"))
    with pytest.raises(Exception) as err:
        benchmark.run(registry_path=path, report_path=tmp_path / "benchmark.json")
    assert "already ran" not in str(err.value)  # it went on to read the (missing) corpus


class _Result:
    def __init__(self, decision, reason, score):
        from src.detection import DetectionDecision

        self.decision, self.reason_codes, self.score = DetectionDecision(decision), [reason], score


class _Service:
    """Flags every text that mentions 'robot'; labels the rest human."""

    def check_batch(self, texts, context):
        return [_Result("likely_ai_generated", "CALIBRATED_HIGH_SCORE", 0.99) if "robot" in t
                else _Result("likely_human_written", "CALIBRATED_LOW_SCORE", 0.01) for t in texts]


def test_excerpt_report_scores_short_excerpts_by_band_for_each_service():
    long_ai = " ".join(f"The robot wrote sentence {i} with care." for i in range(40))
    long_human = " ".join(f"My cat sat on mat number {i} today." for i in range(40))
    test = pd.DataFrame({"sample_id": ["a", "b", "c"], "text": [long_ai, long_human, "too short"],
                         "y": [1, 0, 0], "source_dataset": ["s", "s", "s"]})
    out = benchmark.excerpt_report(test, {"routed": _Service(), "main_only": _Service()})
    assert out["excerpts"] == 2
    bands = out["routed"]["overall_by_band"]
    assert set(bands) <= {"25", "50"} and sum(b["humans"] + b["ais"] for b in bands.values()) == 2
    assert sum(b["detected_ai"] for b in bands.values()) == 1
    assert out["main_only"]["by_source_band"]["s"]
