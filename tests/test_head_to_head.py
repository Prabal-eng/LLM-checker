"""Increment 4d head-to-head: today's models against the 4d models on the v4 locked test."""

import json

import pytest

from src.authorship import headtohead, registry


def test_the_frozen_incumbent_record_is_the_one_that_passed_v3():
    record = headtohead.load_incumbent()
    assert record["model_id"] == "distilroberta-authorship-v5"
    assert record["text_normalization"] == "format2+typography+collapse_whitespace"
    assert record["benchmark"]["passed"] is True


def test_a_changed_incumbent_record_is_refused(tmp_path):
    record = json.loads(headtohead.INCUMBENT_PATH.read_text(encoding="utf-8"))
    record["calibration"]["bands"][0]["ai_min"] = 0.5  # a hand edit after the v3 run
    path = tmp_path / "incumbent.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SystemExit, match="seal"):
        headtohead.load_incumbent(path)
    other = {**json.loads(headtohead.INCUMBENT_PATH.read_text(encoding="utf-8")), "model_id": "someone-else"}
    path.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(SystemExit, match="distilroberta-authorship-v5"):
        headtohead.load_incumbent(path)


import pandas as pd

from src.authorship import benchmark
from src.corpus.paths import VERSION


class _Result:
    def __init__(self, decision, reason, score):
        from src.detection import DetectionDecision

        self.decision, self.reason_codes, self.score = DetectionDecision(decision), [reason], score


class Flagger:
    """Flags texts containing `word`; labels the rest human. Records what it saw."""

    detector_version = "flagger"
    policy = type("Policy", (), {"version": "p"})()

    def __init__(self, word="robot"):
        self.word, self.seen = word, []

    def check_text(self, text, context):
        return self.check_batch([text], context)[0]

    def check_batch(self, texts, context):
        self.seen += list(texts)
        return [_Result("likely_ai_generated", "CALIBRATED_HIGH_SCORE", 0.99) if self.word in t
                else _Result("likely_human_written", "CALIBRATED_LOW_SCORE", 0.01) for t in texts]


def _locked(n=40):
    long_ai = " ".join(f"The robot wrote sentence {{i}} with care." for _ in range(20))
    long_human = " ".join(f"My cat sat on mat number {{i}} today." for _ in range(20))
    rows = []
    for i in range(n):
        for split, source, family in (("test", "raid", "openai"), ("test", "wasitai2026", "openai"),
                                      ("test", "maga", "openai"), ("test_unseen_generator", "stackexchange_heldout", "tii")):
            ai = i % 2 == 1 or split == "test_unseen_generator"
            rows.append({"sample_id": f"{source}{split}{i}", "split": split, "source_dataset": source,
                         "label": "ai" if ai else "human", "generator_family": family if ai else None,
                         "text": (long_ai if ai else long_human).replace("{i}", str(i)), "domain": "d",
                         "transform": "none", "writer_group": None, "group_key": f"g{source}{i}",
                         "source_record_id": str(i)})
    rows.append({"sample_id": "stress0", "split": "stress", "source_dataset": "raid", "label": "ai",
                 "generator_family": "openai", "text": long_ai.replace("{i}", "s"), "domain": "d",
                 "transform": "attack:paraphrase", "writer_group": None, "group_key": "gs", "source_record_id": "s"})
    for j in range(3):
        rows.append({"sample_id": f"in{j}", "split": "inhouse", "source_dataset": "legacy_inhouse", "label": "human",
                     "generator_family": None, "text": f"rater comment {j} about the chart", "domain": "rater",
                     "transform": "none", "writer_group": None, "group_key": "rater-a", "source_record_id": f"row:{j}"})
    df = pd.DataFrame(rows)
    return df.assign(y=(df["label"] == "ai").astype(int))


def _corpus_dir(tmp_path, monkeypatch, df):
    (tmp_path / "corpus" / VERSION).mkdir(parents=True)
    df.drop(columns="y").to_parquet(tmp_path / "corpus" / VERSION / "corpus.parquet", index=False)
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))


def test_model_report_builds_every_section_from_one_models_decisions(tmp_path, monkeypatch):
    df = _locked()
    _corpus_dir(tmp_path, monkeypatch, df)
    record = {"model_id": "m", "text_normalization": "collapse_whitespace"}
    decided, sections = benchmark.model_report(record, Flagger(), df)
    assert len(decided) == int((df["split"] != "inhouse").sum())
    assert {"test", "short_text_excerpts", "test_unseen_generator", "stress", "inhouse_unverified"} <= set(sections)
    assert sections["test"]["by_source"]["raid"]["detected_ai"] == 20


def _decided(rows):
    """rows: (split, source, y, decision) tuples."""
    return pd.DataFrame([{"sample_id": f"s{i}", "split": s, "source_dataset": src, "y": y, "decision": d,
                          "text": "word " * (30 if src == "short" else 150)}
                         for i, (s, src, y, d) in enumerate(rows)])


AI, HUMAN = "likely_ai_generated", "likely_human_written"


def test_shared_ground_keeps_fresh_frontier_and_unseen_rows_only():
    d = _decided([("test", "raid", 1, AI), ("test", "wasitai2026", 1, AI), ("test", "maga", 1, AI),
                  ("test", "stackexchange_heldout", 1, AI), ("test_unseen_generator", "stackexchange_heldout", 1, AI),
                  ("test", "arena140k", 1, AI), ("stress", "raid", 1, AI)])
    assert headtohead.shared_ground(d).tolist() == [True, True, False, False, True, False, False]


def test_the_rule_needs_five_points_on_the_pooled_shared_ground():
    four_d = _decided([("test", "raid", 1, AI)] * 60 + [("test", "raid", 1, HUMAN)] * 40)
    incumbent = four_d.assign(decision=[AI] * 56 + [HUMAN] * 44)
    rule = headtohead.improvement(four_d, incumbent)
    assert rule["gain_points"] == pytest.approx(4.0) and not rule["holds"]
    incumbent = four_d.assign(decision=[AI] * 55 + [HUMAN] * 45)
    assert headtohead.improvement(four_d, incumbent)["holds"]


def test_only_gated_slices_can_block_the_rule():
    # lede loses 50 points on 20 AI rows (not gated); raid gains; the rule holds.
    four_d = _decided([("test", "raid", 1, AI)] * 200 + [("test", "lede", 1, HUMAN)] * 20)
    incumbent = four_d.assign(decision=[AI] * 100 + [HUMAN] * 100 + [AI] * 20)
    rule = headtohead.improvement(four_d, incumbent)
    assert rule["holds"] and rule["losing_slices"] == []
    # With 120 lede AI rows the same loss is on a gated slice and blocks the rule.
    four_d = _decided([("test", "raid", 1, AI)] * 400 + [("test", "lede", 1, HUMAN)] * 120)
    incumbent = four_d.assign(decision=[AI] * 100 + [HUMAN] * 300 + [AI] * 120)
    rule = headtohead.improvement(four_d, incumbent)
    assert rule["gain_points"] > 5 and rule["losing_slices"] == ["lede"] and not rule["holds"]


def test_head_to_head_rows_must_pair():
    four_d = _decided([("test", "raid", 1, AI)] * 3)
    with pytest.raises(ValueError):
        headtohead.improvement(four_d, four_d.iloc[::-1].reset_index(drop=True))


def test_outcome_covers_every_case():
    table = {(True, True, True): "register_4d", (True, True, False): "register_4d",
             (True, False, True): "keep_incumbent", (False, True, True): "keep_incumbent",
             (False, False, True): "keep_incumbent", (True, False, False): "register_4d",
             (False, True, False): "close_gate", (False, False, False): "close_gate"}
    for (four_d, rule, incumbent), expected in table.items():
        assert headtohead.outcome(four_d, rule, incumbent) == expected


def test_goals_report_overall_short_and_frontier_detection():
    d = _decided([("test", "raid", 1, AI), ("test", "raid", 1, HUMAN), ("test", "short", 1, AI),
                  ("test", "lede", 1, AI), ("test", "wasitai2026", 1, HUMAN), ("test", "raid", 0, HUMAN)])
    g = headtohead.goals(d)
    assert g["overall"]["detected"] == pytest.approx(3 / 5)
    assert g["words_25_99"]["detected"] == pytest.approx(1.0) and g["words_25_99"]["ais"] == 1
    assert g["frontier"]["detected"] == pytest.approx(0.5)


from src.corpus import lock as corpus_lock
from src.corpus.schema import LOCKED_SPLITS


def _setup(tmp_path, monkeypatch):
    df = _locked()
    _corpus_dir(tmp_path, monkeypatch, df)
    locked = df[df["split"].isin([s.value for s in LOCKED_SPLITS])].drop(columns="y")
    lock_file = tmp_path / "test.lock.json"
    lock_file.write_text(json.dumps({"sha256": corpus_lock.locked_hash(locked)}), encoding="utf-8")
    reg = tmp_path / "registry.json"
    registry.save({"model_id": "4d-main", "policy_version": "v", "text_normalization": "collapse_whitespace",
                   "corpus_version": VERSION, "benchmark": None, "locked_benchmark_history": []}, reg)
    return dict(registry_path=reg, report_path=tmp_path / "head_to_head.json", lock_file=lock_file)


def test_head_to_head_registers_4d_when_it_passes_and_beats_today(tmp_path, monkeypatch):
    paths = _setup(tmp_path, monkeypatch)
    four_d, today = Flagger("robot"), Flagger("never-present")
    report = headtohead.run(factories={"4d": lambda: four_d, "incumbent": lambda: today}, **paths)
    assert report["outcome"] == "register_4d"
    assert report["improvement_rule"]["holds"] and report["improvement_rule"]["gain_points"] == 100.0
    record = registry.load(paths["registry_path"])
    assert record["benchmark"]["passed"] is True
    assert record["benchmark"]["record_sha256"] == registry.record_sha256(record, record["benchmark"])
    assert record["locked_benchmark_history"][-1]["incumbent_model_id"] == "distilroberta-authorship-v5"
    with pytest.raises(SystemExit, match="already ran"):
        headtohead.run(factories={"4d": lambda: four_d, "incumbent": lambda: today}, **paths)


def test_head_to_head_pairs_the_same_rows(tmp_path, monkeypatch):
    paths = _setup(tmp_path, monkeypatch)
    four_d, today = Flagger("robot"), Flagger("robot")
    headtohead.run(factories={"4d": lambda: four_d, "incumbent": lambda: today}, **paths)
    assert four_d.seen[:200] == today.seen[:200]


def test_an_interrupted_head_to_head_still_counts_as_a_run(tmp_path, monkeypatch):
    paths = _setup(tmp_path, monkeypatch)

    class Crashes(Flagger):
        def check_batch(self, texts, context):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        headtohead.run(factories={"4d": Crashes, "incumbent": Flagger}, **paths)
    record = registry.load(paths["registry_path"])
    assert record["locked_benchmark_history"][-1]["status"] == "started"
    with pytest.raises(SystemExit, match="already ran"):
        headtohead.run(factories={"4d": Flagger, "incumbent": Flagger}, **paths)


def test_head_to_head_refuses_a_changed_lock(tmp_path, monkeypatch):
    paths = _setup(tmp_path, monkeypatch)
    paths["lock_file"].write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(SystemExit, match="lock"):
        headtohead.run(factories={"4d": Flagger, "incumbent": Flagger}, **paths)


from src.authorship import speed


class Slow(Flagger):
    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds

    def check_batch(self, texts, context):
        import time

        time.sleep(self.seconds * len(texts))
        return super().check_batch(texts, context)


def test_speed_times_single_texts_and_scales_the_sheet_sample():
    rows = [[f"cell {r}.{i}" for i in range(6)] for r in range(10)]  # 60 cells, 10 joined rows, scaled to 500 rows
    result = speed.measure(Slow(0.01), singles=["a"] * 10, rows=rows)
    assert 8 <= result["p95_ms"] <= 60
    assert 15 <= result["sheet_seconds"] <= 120 and result["sheet_rows_timed"] == 10
    assert result["sheet_seconds_batched"] < result["sheet_seconds"]


def test_speed_limits_follow_the_device():
    gpu = speed.within({"p95_ms": 400.0, "sheet_seconds": 100.0, "peak_vram_gb": 3.6}, "cuda")
    assert not gpu["passed"] and gpu["checks"] == {"peak_vram_gb": False, "p95_ms": True, "sheet_seconds": True}
    # Build brief Revision 4: the CPU is held to 20 s a text; its sheet time is reported, not gated.
    cpu = speed.within({"p95_ms": 19_000.0, "sheet_seconds": 9_000.0}, "cpu")
    assert cpu["passed"] and cpu["checks"] == {"p95_ms": True}
    assert not speed.within({"p95_ms": 21_000.0, "sheet_seconds": 100.0}, "cpu")["passed"]


# --- Plan C final review fixes ---------------------------------------------------------------------------


def test_each_models_service_is_built_for_its_own_pass_and_released(tmp_path, monkeypatch):
    import gc
    import weakref

    paths = _setup(tmp_path, monkeypatch)
    refs = {}

    def four_d():
        svc = Flagger("robot")
        refs["4d"] = weakref.ref(svc)
        return svc

    def incumbent():
        gc.collect()
        assert refs["4d"]() is None, "the 4d models are still loaded while today's models score"
        return Flagger("never-present")

    report = headtohead.run(factories={"4d": four_d, "incumbent": incumbent}, **paths)
    assert report["outcome"] == "register_4d"


def test_the_main_only_excerpt_service_reuses_the_routed_main_model(tmp_path):
    from tests.test_authorship_registry import _record, _with_short

    record = _with_short(tmp_path, _record(tmp_path))
    record["calibration"]["main_bands"] = [dict(b, model="main") for b in record["calibration"]["bands"]]
    routed = registry.service_from_record(record, tmp_path, validated=True)
    main = benchmark.main_only_service(record, routed)
    assert main.detector is routed.detector.main
    assert main.model_validated and [b.lower_words for b in main.policy.bands] == [100]


def test_the_head_to_head_never_touches_cuda_with_injected_services(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    paths = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def touched(*args, **kwargs):
        raise AssertionError("the test touched CUDA")

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", touched)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", touched)
    report = headtohead.run(factories={"4d": lambda: Flagger("robot"), "incumbent": Flagger}, **paths)
    assert report["models"]["4d"]["peak_inference_vram_gb"] is None


def test_unseen_generator_rows_of_sources_4d_trained_on_are_not_shared_ground():
    d = _decided([("test_unseen_generator", "raid", 1, AI), ("test_unseen_generator", "wasitai2026", 1, AI),
                  ("test_unseen_generator", "stackexchange_heldout", 1, AI), ("test_unseen_generator", "maga", 1, AI),
                  ("test_unseen_generator", "dactyl", 1, AI), ("test_unseen_generator", "arena140k", 1, AI),
                  ("test_unseen_generator", "openturingbench", 1, AI), ("test_unseen_generator", "stackexchange", 1, AI)])
    assert headtohead.shared_ground(d).tolist() == [True, True, True, False, False, False, False, False]
    rule = headtohead.improvement(d, d)
    assert set(rule["slices"]["test_unseen_generator"]["by_source"]) == {"raid", "wasitai2026", "stackexchange_heldout"}


class Recorder:
    detector_version = "recorder"
    policy = type("Policy", (), {"version": "p"})()

    def __init__(self):
        self.text_calls, self.batches = 0, []

    def check_text(self, text, context):
        self.text_calls += 1
        return _Result("likely_human_written", "CALIBRATED_LOW_SCORE", 0.01)

    def check_batch(self, texts, context):
        self.batches.append((list(texts), context))
        return [_Result("likely_human_written", "CALIBRATED_LOW_SCORE", 0.01) for _ in texts]


def test_the_sheet_is_timed_the_way_the_app_scores_it():
    # The app prefetches every cell and each row's joined cells in batched rater-comment calls.
    from src.detection import TextContext

    svc = Recorder()
    speed.time_sheet(svc, [["first cell", "second cell"], ["only cell", ""]])
    scored = sorted(t for texts, context in svc.batches for t in texts)
    assert scored == sorted(["first cell", "second cell", "first cell\n\nsecond cell", "only cell", ""])
    assert {context for _, context in svc.batches} == {TextContext.RATER_COMMENT} and svc.text_calls == 0


# --- Ralph R12 (Amendment 13) ----------------------------------------------------------------------------


class Errs(Flagger):
    """Fails on one text the way the service reports a detector exception."""

    def check_batch(self, texts, context):
        out = super().check_batch(texts, context)
        return [_Result("inconclusive", "DETECTOR_ERROR", None) if "sentence 3 " in t else r
                for t, r in zip(texts, out)]


def test_a_detector_error_stops_the_run_unsealed(tmp_path, monkeypatch):
    paths = _setup(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="detector error"):
        headtohead.run(factories={"4d": lambda: Errs("robot"), "incumbent": Flagger}, **paths)
    record = registry.load(paths["registry_path"])
    assert record["benchmark"] is None
    assert record["locked_benchmark_history"][-1]["status"] == "detector_error"
    assert not paths["report_path"].exists()


def test_the_head_to_head_refuses_a_record_that_is_not_a_v4_4d_record(tmp_path, monkeypatch):
    paths = _setup(tmp_path, monkeypatch)
    for change in ({"corpus_version": "v3"}, {"model_id": headtohead.INCUMBENT_MODEL_ID}):
        record = registry.load(paths["registry_path"])
        registry.save({**record, **change}, paths["registry_path"])
        with pytest.raises(SystemExit, match="4d record"):
            headtohead.run(factories={"4d": Flagger, "incumbent": Flagger}, **paths)
        registry.save({**record}, paths["registry_path"])
    assert registry.load(paths["registry_path"])["locked_benchmark_history"] == []


def test_the_head_to_head_checks_the_4d_record_verifies_before_the_lock_is_spent(tmp_path, monkeypatch):
    paths = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(registry, "verify_files", lambda rec, root=None: (True, "files ok"))
    with pytest.raises(SystemExit, match="does not verify"):
        headtohead.run(**paths)  # the record states no precision and no scoring-code digest
    assert registry.load(paths["registry_path"])["locked_benchmark_history"] == []


def test_unseen_generator_rows_of_excluded_sources_are_reported_by_source():
    d = _decided([("test_unseen_generator", "raid", 1, AI), ("test_unseen_generator", "openturingbench", 1, AI),
                  ("test_unseen_generator", "openturingbench", 1, HUMAN)])
    unseen = headtohead.improvement(d, d)["slices"]["test_unseen_generator"]
    assert set(unseen["by_source"]) == {"raid"}
    assert unseen["excluded_by_source"]["openturingbench"] == {"ais": 2, "4d": 0.5, "incumbent": 0.5}


def test_the_held_out_stack_exchange_sites_are_reported_one_by_one(tmp_path, monkeypatch):
    """Ralph R12-5: the gate holds stackexchange_heldout as one slice, so each site's false AI is reported."""
    df = _locked()
    extra = df[df["source_dataset"] == "raid"].head(8).copy()
    extra["source_dataset"] = "stackexchange_heldout"
    extra["sample_id"] = [f"se{i}" for i in range(len(extra))]
    extra["domain"] = ["academia", "cooking"] * 4
    df = pd.concat([df, extra], ignore_index=True)
    _corpus_dir(tmp_path, monkeypatch, df)
    _, sections = benchmark.model_report({"model_id": "m", "text_normalization": "collapse_whitespace"}, Flagger(), df)
    sites = sections["test"]["stackexchange_heldout_by_site"]
    assert set(sites) == {"academia", "cooking"} and sites["academia"]["humans"] + sites["academia"]["ais"] == 4


def test_the_rule_reports_the_pooled_gain_without_lede_too():
    """Ralph R12-15: LEDE is most of the shared ground, so the pooled figure is also reported without it."""
    d4 = _decided([("test", "lede", 1, AI), ("test", "lede", 1, AI), ("test", "raid", 1, AI), ("test", "raid", 1, HUMAN)])
    di = _decided([("test", "lede", 1, HUMAN), ("test", "lede", 1, HUMAN), ("test", "raid", 1, AI), ("test", "raid", 1, HUMAN)])
    rule = headtohead.improvement(d4, di)
    assert rule["gain_points"] == 50.0
    assert rule["without_lede"] == {"4d": 0.5, "incumbent": 0.5, "ais": 2, "gain_points": 0.0}


@pytest.mark.parametrize("reason", ["DETECTOR_INVALID_EVIDENCE", "DETECTOR_REPORTED_ERROR"])
def test_every_detector_failure_reason_stops_the_run(tmp_path, monkeypatch, reason):
    """Ralph R13-7: DETECTOR_INVALID_EVIDENCE is a detector failure too."""
    paths = _setup(tmp_path, monkeypatch)

    class Fails(Flagger):
        def check_batch(self, texts, context):
            out = super().check_batch(texts, context)
            return [_Result("inconclusive", reason, None) if "sentence 3 " in t else r for t, r in zip(texts, out)]

    with pytest.raises(SystemExit, match="detector error"):
        headtohead.run(factories={"4d": lambda: Fails("robot"), "incumbent": Flagger}, **paths)


def test_the_diagnostic_rows_are_read_before_the_run_is_recorded(tmp_path, monkeypatch):
    """Ralph R13-7: a memory failure reading report-only rows after the run is recorded would spend the lock."""
    paths = _setup(tmp_path, monkeypatch)
    seen = []
    real = benchmark.read_diagnostic

    def spy():
        seen.append(len(registry.load(paths["registry_path"])["locked_benchmark_history"]))
        return real()

    monkeypatch.setattr(benchmark, "read_diagnostic", spy)
    headtohead.run(factories={"4d": lambda: Flagger("robot"), "incumbent": Flagger}, **paths)
    assert seen == [0]


def test_the_cpu_speed_run_hides_the_gpu_and_refuses_if_it_is_still_visible(monkeypatch):
    """Ralph R13-8: an empty CUDA_VISIBLE_DEVICES doesn't hide the GPU on Windows, so the CPU pass timed the GPU
    again and wrote no cpu entry. --device cpu sets -1 and refuses to run while CUDA is still visible."""
    pytest.importorskip("torch")  # CI installs requirements.txt, which leaves torch out
    import os

    import torch

    seen = {}
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(speed, "run", lambda device=None: seen.update(device=device, env=os.environ.get("CUDA_VISIBLE_DEVICES")) or {})
    speed.main(["--device", "cpu"])
    assert seen == {"device": "cpu", "env": "-1"}

    monkeypatch.undo()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(SystemExit, match="GPU is still visible"):
        speed.run(device="cpu")


def test_the_served_inference_batch_is_16():
    """Amendment 16: batch 32 peaked at 3.94 GB with both large models loaded (limit 3.5 GB); batch 16 peaked at
    2.85 GB on the same sheet, the largest candidate under the limit."""
    import inspect

    from src.authorship.detector import AuthorshipModelDetector

    assert inspect.signature(AuthorshipModelDetector.__init__).parameters["batch_size"].default == 16


def test_the_gpu_sheet_limit_is_900_seconds():
    """Amendment 16 (user decision after the 566 s measurement): the GPU sheet limit is 900 s; memory and p95 keep
    their pre-registered limits."""
    assert speed.LIMITS["cuda"] == {"peak_vram_gb": 3.5, "p95_ms": 500.0, "sheet_seconds": 900.0}
    assert speed.within({"p95_ms": 190.0, "sheet_seconds": 600.0, "peak_vram_gb": 3.0}, "cuda")["passed"]
    assert not speed.within({"p95_ms": 190.0, "sheet_seconds": 901.0, "peak_vram_gb": 3.0}, "cuda")["passed"]
