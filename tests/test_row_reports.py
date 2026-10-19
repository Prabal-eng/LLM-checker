"""Row reports: the pipeline's one-per-row record of tier, bar, decided text and trigger (spec 2026-10-01)."""
import pandas as pd

from src.config import PipelineConfig
from src.detection import AuthorshipService
from src.detection.policy import BandRule, DecisionPolicy
from src.detectors.base import BaseDetector, DetectionResult
from src.pipeline import QAPipeline
from src.report.rows import agent_table, decided_cells, tier_counts, trigger_word_table
from src.report.tiers import HIGH, LIKELY_AI, LOW, NOT_CHECKED

PLANT = "We must delve into the rich tapestry of this response."


class Planted(BaseDetector):
    def detect(self, text):
        score = 0.995 if "certainly" in text else 0.95 if "rich tapestry" in text else 0.20
        return DetectionResult(provider="planted", ai_score=score, is_flagged=score >= 0.5,
                               analysis={"evidence": ["planted"]})


def _service():
    policy = DecisionPolicy(min_word_count=5, bands=(BandRule(5, None, 0.05, 0.99),))
    return AuthorshipService(Planted(), policy=policy, model_validated=True, detector_version="planted-v1")


def _run(tmp_path, rows, cols=("why",), **trigger_cfg):
    src = tmp_path / "in.csv"
    pd.DataFrame(rows).to_csv(src, index=False)
    config = PipelineConfig()
    config.structural.text_columns = list(cols)
    config.structural.skip_unassigned = False
    for key, value in trigger_cfg.items():
        setattr(config.triggers, key, value)
    pipe = QAPipeline(config=config, authorship_service=_service())
    return pipe, pipe.run(str(src), str(tmp_path / "out.xlsx"))


ROWS = [
    {"rater_id": "ann", "task_id": "T1", "why": f"The rating is fair overall. {PLANT} It matches the rubric well."},
    {"rater_id": "bo", "task_id": "T2", "why": "The rating is fair and the response cites the source it used."},
    {"rater_id": "ann", "task_id": "T3", "why": "It is certainly a clear and complete answer to the question."},
    {"rater_id": "cy", "task_id": "T4", "why": "Too short."},
]


def test_each_row_gets_a_tier_a_bar_and_its_decided_text(tmp_path):
    pipe, res = _run(tmp_path, ROWS)
    tiers = [r.tier for r in pipe.last_reports]
    assert tiers == [HIGH, LOW, LIKELY_AI, NOT_CHECKED]
    assert pipe.last_reports[0].bar == 0.99 and pipe.last_reports[3].bar is None
    assert pipe.last_reports[3].reason == "too short"
    assert pipe.last_reports[0].decided == [("why", ROWS[0]["why"])]
    assert pipe.last_reports[0].task_id == "T1" and pipe.last_reports[0].agent == "ann"
    assert res.summary["tier_counts"][HIGH] == 1 and res.summary["tier_counts"][LIKELY_AI] == 1
    assert res.ai_flagged_count == 1                       # decisions unchanged


def test_triggers_are_found_for_high_and_likely_ai_rows_only(tmp_path):
    pipe, res = _run(tmp_path, ROWS)
    high, low = pipe.last_reports[0], pipe.last_reports[1]
    assert high.trigger.kind == "sentences" and high.trigger.sentences[0][0].text == PLANT
    assert low.trigger is None
    assert res.summary["triggers"]["enabled"] is True and res.summary["triggers"]["rows"] == 2


def test_trigger_words_are_found_and_tabled_across_the_sheet(tmp_path):
    pipe, res = _run(tmp_path, ROWS + [{"rater_id": "dee", "task_id": "T5", "why": "Certainly fine."}])
    assert [w.text for w in pipe.last_reports[0].trigger.words] == ["rich tapestry"]
    assert [w.text for w in pipe.last_reports[2].trigger.words] == ["certainly"]
    table = {e["phrase"]: e for e in res.summary["trigger_words"]}
    assert table["certainly"] == {"phrase": "certainly", "trigger_rows": 1, "agents": 1, "agent_names": ["ann"],
                                  "average_share": 1.0, "rows_containing": 2, "share_of_rows": 0.4}
    assert table["rich tapestry"]["rows_containing"] == 1


def test_trigger_words_can_be_switched_off_alone(tmp_path):
    pipe, res = _run(tmp_path, ROWS, words=False)
    assert pipe.last_reports[0].trigger.kind == "sentences" and pipe.last_reports[0].trigger.words == []
    assert res.summary["trigger_words"] == []


def test_triggers_can_be_switched_off(tmp_path):
    pipe, res = _run(tmp_path, ROWS, enabled=False)
    assert all(r.trigger is None for r in pipe.last_reports)
    assert res.summary["triggers"] == {"enabled": False, "rows": 0, "seconds": 0.0}
    assert res.summary["trigger_words"] == []


def test_a_trigger_error_never_fails_the_audit(tmp_path, monkeypatch):
    from src.report import triggers

    def boom(self, segments):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(triggers.TriggerFinder, "explain", boom)
    pipe, res = _run(tmp_path, ROWS)
    assert pipe.last_reports[0].trigger.kind == "error"
    assert pipe.last_reports[0].trigger.summary() == "trigger not found (error)"


def test_agents_are_ranked_most_flagged_first(tmp_path):
    _, res = _run(tmp_path, ROWS)
    agents = res.summary["agents"]
    assert [a["agent"] for a in agents] == ["ann", "bo", "cy"]
    assert agents[0] == {"agent": "ann", "rows": 2, "likely_ai": 1, "high": 1, "moderate": 0, "not_checked": 0,
                         "highest_score": 0.995, "share_likely_ai": 0.5}
    assert agents[2]["not_checked"] == 1 and agents[2]["highest_score"] is None


def test_a_row_decided_on_its_combined_text_uses_both_cells_and_a_lone_second_cell_is_its_own(tmp_path):
    rows = [{"rater_id": "a", "first": "The rating is fair overall here.", "second": f"{PLANT} It matches."},
            {"rater_id": "b", "first": "", "second": f"The rating is fair overall. {PLANT}"}]
    pipe, _ = _run(tmp_path, rows, cols=("first", "second"))
    both, lone = pipe.last_reports
    assert [c for c, _ in both.decided] == ["first", "second"] and both.decided_on == "all columns together"
    assert lone.decided == [("second", rows[1]["second"])] and lone.decided_on == "second"
    assert lone.trigger.sentences[0][0].column == "second"


def test_decided_cells_prefers_a_cell_that_flagged_the_row_on_its_own():
    class Row:
        evidence = [type("E", (), {"code": "CELL_FLAGGED_AI"})()]
        likely_ai_columns = ["b"]
        column_scores = {"a": 0.2, "b": 0.995}

    assert decided_cells([("a", "x y"), ("b", "z w")], Row(), combined_checked=True) == [("b", "z w")]
    assert decided_cells([("a", ""), ("b", "")], Row(), combined_checked=False) == []


def test_tier_counts_and_agent_table_handle_no_rows():
    assert set(tier_counts([])) and all(v == 0 for v in tier_counts([]).values())
    assert agent_table([]) == [] and trigger_word_table([]) == []


def test_a_sheet_without_an_agent_column_reads_no_agent(tmp_path):
    pipe, res = _run(tmp_path, [{"why": "The rating is fair and the response cites the source it used."}])
    assert pipe.last_reports[0].agent == "(no agent)"


def test_a_first_id_column_is_not_taken_for_the_agent_column(tmp_path):
    """Discovery falls back to a sheet's first column when none is named like a rater column; an ID column there
    is not an agent column, and it is still the task id (final review, Important 2)."""
    rows = [{"prompt_id": "P1", "why": "The rating is fair and the response cites the source it used."},
            {"prompt_id": "P2", "why": "It matches the rubric and the answer covers each point asked."}]
    pipe, res = _run(tmp_path, rows)
    assert [r.agent for r in pipe.last_reports] == ["(no agent)", "(no agent)"]
    assert [r.task_id for r in pipe.last_reports] == ["P1", "P2"]
    assert res.summary["agent_column"] is None


def test_a_column_named_like_a_rater_column_is_the_agent_column(tmp_path):
    _, res = _run(tmp_path, ROWS)
    assert res.summary["agent_column"] == "rater_id"
