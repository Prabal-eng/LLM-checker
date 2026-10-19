# Increment 4d Training Implementation Plan (Plan B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the screening winner on corpus v4 with quarter-epoch early stopping, record its inference precision in the sealed registry, and measure the two optional steps (perplexity stack, style-aware thresholds) on validation so each is adopted or dropped by its pre-registered rule.

**Architecture:** `FineTunedClassifier.fit` gains validation checks inside an epoch and resumes from a check. `src/authorship/train.py` trains a named screening candidate with its screening recipe, keeps the best check, and merges LoRA adapters before recording the artifact. The detector reads its autocast precision from the registry record (sealed code; the v5 seal is already broken on this branch). Two new validation-only modules measure the optional steps and write text-free reports. Integrating an adopted optional step into the sealed scoring path is a separate plan (B2), written only if a step is adopted.

**Tech Stack:** Python 3.12, PyTorch, transformers, peft, bitsandbytes, scikit-learn, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-increment-4d-detection-design.md` (sections "Full training on corpus v4", "Perplexity signal", "Style-aware thresholds"), bound by `docs/bob/llm-authorship-checker/increment-4d-preregistration.md` (section "Models").

## Global Constraints

- Validation every quarter epoch; stop after two checks without improvement, at most three epochs, keep the best check.
- Length-grouped batches.
- LoRA adapters are merged before saving.
- The inference precision is recorded in the registry and read by the detector.
- Registry stays fail-closed: a missing field, a hash mismatch or a report that doesn't match the seal closes the gate. The new precision field is covered by the seal.
- Calibration targets: 0.8% false AI and 8% missed AI, per word band.
- Perplexity stack: fitted on half of validation, measured on the other half; adopted only if it adds at least 2 points of detection at 1% false AI and the GPU limits hold.
- Style-aware thresholds: a cell with fewer than 600 validation humans uses the pooled band bar; adopted only if it adds at least 2 points of validation detection with every cell within target.
- GPU limits (unchanged): peak inference memory ≤ 3.5 GB; single text ≤ 500 ms at p95; a 500 × 6 sheet ≤ 120 s.
- Only train and validation rows are read. No locked split is touched by anything in this plan.
- The full suite passes on a machine with no GPU or model files; tests that build a model force the CPU and never touch CUDA (screening or training may hold the GPU while tests run).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never open, commit or inspect the files `x` and `inspect_wb.py`. Never train on the in-house root sheets.

## Review Focus

1. **Resume after a crash between a check's metrics write and its resume write:** the check must be re-run and re-scored, never skipped, and the stop decision recomputed from the saved history. (Task 2 test `test_train_resumes_inside_an_epoch_after_a_crash` covers the re-run; the reviewer checks the ordering of writes.)
2. **A LoRA artifact that differs from what validation scored:** the merged model must give the adapter model's logits. (Task 2 `test_export_merged_matches_the_adapter_model`.)
3. **A pre-4d record scored through the new detector (head-to-head in Plan C):** it must score under bf16 autocast, as it was calibrated, while its gate stays closed without a recorded precision. (Task 3 tests.)
4. **An excerpt and its parent split across the stack's halves:** would leak a text into both fit and measurement. (Task 4 `test_halves_keep_excerpts_with_their_parent`.)
5. **A training source with no register mapping:** must fail loudly, not train under a guessed register. (Task 5 `test_every_training_pair_has_a_register` and `test_an_unmapped_pair_fails_loudly`.)

---

### Task 1: Validation checks inside an epoch, with resume from a check

**Files:**
- Modify: `src/bakeoff/neural.py` (add `check_points`; extend `FineTunedClassifier.fit`)
- Create: `tests/tiny_encoder.py` (test helper: a two-layer RoBERTa and a word-level tokenizer saved to disk)
- Create: `tests/test_training_4d.py`

**Interfaces:**
- Produces: `check_points(micro_batches: int, checks: int, accumulate: int) -> list[int]`
- Produces: `FineTunedClassifier.fit(train, *, balance_sources=False, on_epoch_end=None, resume=None, checks_per_epoch: int = 1, on_check: Optional[Callable[[dict], bool]] = None)`. `on_check` receives `{"epoch", "check" (1-based), "micro" (completed micro-batches this epoch), "model", "optimizer", "schedule", "scaler"}` and returns True to stop training. `resume` accepts that dict and continues after its micro-batch.
- Produces (tests): `tests.tiny_encoder.tiny_encoder(path) -> Path`, `tests.tiny_encoder.rows(n) -> pd.DataFrame`

- [ ] **Step 1: Write the test helper**

`tests/tiny_encoder.py`:

```python
"""A two-layer RoBERTa and a word-level tokenizer saved to disk, so training code runs on the CPU in
seconds with no download."""

from pathlib import Path

import pandas as pd

WORDS = ("the", "chart", "labels", "are", "clear", "vocab_a", "gamma", "delta", "river", "stone")


def tiny_encoder(path: Path) -> Path:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

    if (path / "config.json").exists():
        return path
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{w: i + 4 for i, w in enumerate(WORDS)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>",
                                   bos_token="<s>", eos_token="</s>")
    config = RobertaConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=64, max_position_embeddings=80, num_labels=2, pad_token_id=0)
    torch.manual_seed(0)
    RobertaForSequenceClassification(config).save_pretrained(path)
    fast.save_pretrained(path)
    return path


def rows(n: int = 64) -> pd.DataFrame:
    """n texts, half AI (they contain vocab_a), from two sources."""
    human = "the chart labels are clear river stone"
    ai = "the chart labels are clear vocab_a gamma delta"
    return pd.DataFrame({"text": [ai if i % 2 else human for i in range(n)], "y": [i % 2 for i in range(n)],
                         "source_dataset": ["a" if i % 4 < 2 else "b" for i in range(n)]})
```

- [ ] **Step 2: Write the failing tests**

`tests/test_training_4d.py`:

```python
"""Increment 4d full training: checks inside an epoch, early stopping, LoRA merging."""

import pytest

from src.bakeoff import neural
from src.bakeoff.neural import check_points
from tests.tiny_encoder import rows, tiny_encoder


def test_check_points_fall_on_optimizer_steps_and_end_the_epoch():
    assert check_points(100, 4, 2) == [26, 50, 76, 100]
    assert check_points(7, 4, 2) == [2, 4, 6, 7]
    assert check_points(10, 1, 2) == [10]


def _tiny_classifier(tmp_path, monkeypatch, epochs):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(neural, "_device", lambda: torch.device("cpu"))  # never touch a busy GPU
    return neural.FineTunedClassifier(model_name=str(tiny_encoder(tmp_path / "tiny")), max_tokens=16,
                                      max_train=None, epochs=epochs, batch_size=8, accumulate=2,
                                      text_view="collapse_whitespace")


def test_fit_checks_each_quarter_epoch_and_stops_when_told(tmp_path, monkeypatch):
    clf = _tiny_classifier(tmp_path, monkeypatch, epochs=3)
    calls = []

    def on_check(state):
        calls.append((state["epoch"], state["check"], state["micro"]))
        return len(calls) == 3

    clf.fit(rows(64), balance_sources=True, checks_per_epoch=4, on_check=on_check)
    assert calls == [(0, 1, 2), (0, 2, 4), (0, 3, 6)]


def test_fit_resumes_inside_an_epoch_from_a_check_state(tmp_path, monkeypatch):
    first = _tiny_classifier(tmp_path, monkeypatch, epochs=2)
    saved = {}

    def stop_at_second(state):
        if state["check"] == 2:
            saved.update(state)
            return True
        return False

    first.fit(rows(64), balance_sources=True, checks_per_epoch=4, on_check=stop_at_second)
    second = _tiny_classifier(tmp_path, monkeypatch, epochs=2)
    calls = []
    second.fit(rows(64), balance_sources=True, checks_per_epoch=4, resume=saved,
               on_check=lambda s: calls.append((s["epoch"], s["check"])) or False)
    assert calls == [(0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (1, 4)]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_training_4d.py -q`
Expected: FAIL — `ImportError: cannot import name 'check_points'`.

- [ ] **Step 4: Implement**

In `src/bakeoff/neural.py`, after `length_grouped_order`, add:

```python
def check_points(micro_batches: int, checks: int, accumulate: int) -> list:
    """Completed micro-batch counts after which an epoch's validation checks run: evenly spaced, each
    right after an optimizer step, the last at the epoch's end (Increment 4d: a check every quarter
    epoch)."""
    points = {micro_batches}
    for k in range(1, checks):
        points.add(min(micro_batches, math.ceil(micro_batches * k / checks / accumulate) * accumulate))
    return sorted(points)
```

In `FineTunedClassifier.fit`, change the signature and docstring:

```python
    def fit(self, train: pd.DataFrame, *, balance_sources: bool = False,
            on_epoch_end: Optional[Callable[[int, dict], None]] = None, resume: Optional[dict] = None,
            checks_per_epoch: int = 1, on_check: Optional[Callable[[dict], bool]] = None) -> None:
        """Fine-tune on train.

        balance_sources weights each (source, label) cell so every source contributes equal human
        and AI mass, removing source-level label imbalance as a shortcut. on_epoch_end receives the
        finished epoch and a state dict (model, optimizer, schedule) that resume accepts to continue
        after the next epoch boundary. on_check runs checks_per_epoch times an epoch, right after an
        optimizer step; it receives a state dict (epoch, check, micro, model, optimizer, schedule,
        scaler) that resume accepts to continue from that micro-batch, and returns True to stop.
        """
```

Replace the resume block

```python
        first_epoch = 0
        if resume is not None:
            self.model.load_state_dict(resume["model"])
            optimizer.load_state_dict(resume["optimizer"])
            schedule.load_state_dict(resume["schedule"])
            first_epoch = resume["epoch"] + 1
```

with

```python
        first_epoch, skip = 0, 0
        if resume is not None:
            self.model.load_state_dict(resume["model"])
            optimizer.load_state_dict(resume["optimizer"])
            schedule.load_state_dict(resume["schedule"])
            if resume.get("scaler"):
                scaler.load_state_dict(resume["scaler"])
            # A check's state carries its micro-batch; an epoch-end state resumes at the next epoch.
            done = resume.get("micro", micro_batches)
            first_epoch, skip = (resume["epoch"], done) if done < micro_batches else (resume["epoch"] + 1, 0)
        check_of = {p: i + 1 for i, p in enumerate(check_points(micro_batches, checks_per_epoch, self.accumulate))}
        stop = False
```

In the epoch loop, skip already-trained micro-batches first:

```python
            for micro, start in enumerate(range(0, len(texts), self.batch_size)):
                if epoch == first_epoch and micro < skip:
                    continue
```

After the `if micro % 1000 == 0:` logging block (still inside the micro-batch loop) add:

```python
                if on_check is not None and micro + 1 in check_of:
                    self.model.eval()
                    stop = bool(on_check({"epoch": epoch, "check": check_of[micro + 1], "micro": micro + 1,
                                          "model": self.model.state_dict(), "optimizer": optimizer.state_dict(),
                                          "schedule": schedule.state_dict(), "scaler": scaler.state_dict()}))
                    self.model.train()
                    if stop:
                        break
            if stop:
                break
```

(the existing `if on_epoch_end is not None:` block follows unchanged).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_training_4d.py tests/test_screening.py tests/test_authorship_train.py -q`
Expected: PASS (3 new, all screening and training tests still pass).

- [ ] **Step 6: Commit**

```bash
git add src/bakeoff/neural.py tests/tiny_encoder.py tests/test_training_4d.py
git commit -m "feat: validation checks inside an epoch, resumable from a check"
```

---

### Task 2: Train a screening candidate with early stopping and LoRA merging

**Files:**
- Modify: `src/bakeoff/neural.py` (add `export_merged`; drop `on_epoch_end` from `fit`)
- Modify: `src/authorship/train.py` (rewrite `train`, `record_file`, CLI)
- Modify: `tests/test_authorship_train.py` (FakeClassifier speaks the check protocol)
- Test: `tests/test_training_4d.py`

**Interfaces:**
- Consumes: `FineTunedClassifier.fit(..., checks_per_epoch, on_check, resume)` from Task 1; `screen.CANDIDATES`.
- Produces: `export_merged(base_model: str, adapter_dir: Path, out_dir: Path) -> Path`
- Produces: `train.model_id(candidate: str, variant: str = "main") -> str` (`"<candidate>-authorship-4d"`, `"<candidate>-authorship-short-4d"`)
- Produces: `train.record_file(version: str = VERSION, variant: str = "main", candidate: str = "distilroberta-base") -> Path` = `data/benchmark/<version>/model/<candidate>/training_record.json` (or `training_record_short.json`)
- Produces: `train.train(classifier=None, *, candidate="distilroberta-base", epochs=3, checks_per_epoch=4, patience=2, out_dir=None, record=None, data=None, variant="main") -> dict`. Record keys used later: `model_id`, `candidate`, `variant`, `artifact_path`, `artifact_sha256`, `max_tokens`, `precision`, `text_normalization`, `checks`, `chosen_check`.
- Produces: `train.best_check(history: dict) -> str`, `train.stale_checks(history: dict) -> int` (history keys are `"<epoch>.<check>"`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_training_4d.py`:

```python
def test_export_merged_matches_the_adapter_model(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    base_dir = tiny_encoder(tmp_path / "tiny")
    model = neural.apply_lora(AutoModelForSequenceClassification.from_pretrained(base_dir), {"r": 4, "alpha": 8})
    torch.manual_seed(1)
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.5)  # LoRA B starts at zero; make the adapter matter
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    adapter = tmp_path / "check-0.3"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    out = neural.export_merged(str(base_dir), adapter, tmp_path / "merged-0.3")
    merged = AutoModelForSequenceClassification.from_pretrained(out).eval()
    enc = tokenizer(["the chart labels vocab_a", "river stone"], return_tensors="pt", padding=True)
    with torch.inference_mode():
        assert torch.allclose(merged(**enc).logits, model(**enc).logits, atol=1e-5)
    assert not (out / "adapter_config.json").exists()
    assert (out / "tokenizer.json").exists()
```

Replace `FakeClassifier` and the two epoch tests in `tests/test_authorship_train.py` (keep `_Saver`, `_data`, the source-balance test and the excerpt test) with:

```python
class FakeClassifier:
    """Validation improves for three checks, then worsens; records which checks it trained."""

    model_name, max_tokens = "fake", 16
    NOISE = [2.0, 1.0, 0.1, 0.5, 1.5, 2.0, 2.5, 3.0, 3.0, 3.0, 3.0, 3.0]  # per check; lower scores better

    def __init__(self, epochs=3):
        self.epochs, self.trained, self.noise = epochs, [], 2.0
        self.model = self.tokenizer = _Saver()

    def fit(self, train, *, balance_sources, checks_per_epoch, on_check, resume):
        assert balance_sources
        start = 0 if resume is None else resume["epoch"] * checks_per_epoch + resume["check"]
        for i in range(start, self.epochs * checks_per_epoch):
            epoch, check = divmod(i, checks_per_epoch)
            self.noise = self.NOISE[i]
            self.trained.append(f"{epoch}.{check + 1}")
            if on_check({"epoch": epoch, "check": check + 1, "micro": 0}):
                return

    def score(self, texts):
        rng = np.random.default_rng(len(self.trained))
        return np.array([("vocab_a" in t) + rng.normal(scale=self.noise) for t in texts])


def test_train_stops_two_checks_after_the_best_and_keeps_only_its_weights(tmp_path):
    clf = FakeClassifier()
    out = tmp_path / "m"
    result = train_mod.train(clf, out_dir=out, record=tmp_path / "rec.json", data=_data())
    assert clf.trained == ["0.1", "0.2", "0.3", "0.4", "1.1"]
    assert result["chosen_check"] == "0.3" and result["stopped_early"] is True
    assert sorted(p.name for p in out.glob("check-*")) == ["check-0.3"]
    assert result["artifact_sha256"] == train_mod.sha256_file(out / "check-0.3" / "model.safetensors")
    assert result["locked_split_reads"] == 0 and result["lora_merged"] is False
    assert not (out / "resume.pkl").exists()
    saved = json.loads((tmp_path / "rec.json").read_text(encoding="utf-8"))
    assert saved["text_normalization"] == "format3+typography+collapse_whitespace"
    assert saved["candidate"] == "distilroberta-base" and saved["precision"] == "bf16"
    # No corpus text leaks into the committed record: every value is a number, hash, path, or setting.
    assert "vocab" not in json.dumps(saved)


def test_train_resumes_inside_an_epoch_after_a_crash(tmp_path):
    class Crash(FakeClassifier):
        def fit(self, train, **kwargs):
            inner = kwargs["on_check"]

            def on_check(state):
                inner(state)
                if state["check"] == 2:
                    raise KeyboardInterrupt  # the process dies after the second check's resume state
                return False

            super().fit(train, **{**kwargs, "on_check": on_check})

    out = tmp_path / "m"
    with pytest.raises(KeyboardInterrupt):
        train_mod.train(Crash(), out_dir=out, record=tmp_path / "rec.json", data=_data())
    second = FakeClassifier()
    result = train_mod.train(second, out_dir=out, record=tmp_path / "rec.json", data=_data())
    assert second.trained[0] == "0.3"
    assert set(result["checks"]) == {"0.1", "0.2", "0.3", "0.4", "1.1"}


def test_a_lora_run_records_the_merged_model(tmp_path, monkeypatch):
    from src.bakeoff import neural

    def fake_export(base, adapter, out):
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.safetensors").write_bytes(b"merged-" + adapter.name.encode())
        return out

    monkeypatch.setattr(neural, "export_merged", fake_export)
    clf = FakeClassifier()
    clf.lora = {"r": 16, "alpha": 32}
    result = train_mod.train(clf, out_dir=tmp_path / "m", record=tmp_path / "rec.json", data=_data())
    assert result["artifact_path"].endswith("merged-0.3") and result["lora_merged"] is True
    assert result["artifact_sha256"] == train_mod.sha256_file(tmp_path / "m" / "merged-0.3" / "model.safetensors")


def test_best_check_prefers_detection_then_auroc_then_the_earlier_check():
    h = {"0.1": {"tpr_at_1pct_fpr": 0.5, "auroc": 0.9}, "0.2": {"tpr_at_1pct_fpr": 0.6, "auroc": 0.8},
         "0.3": {"tpr_at_1pct_fpr": 0.6, "auroc": 0.8}, "1.1": {"tpr_at_1pct_fpr": 0.4, "auroc": 0.99}}
    assert train_mod.best_check(h) == "0.2"
    assert train_mod.stale_checks(h) == 2


def test_model_ids_and_records_are_per_candidate():
    assert train_mod.model_id("modernbert-base") == "modernbert-base-authorship-4d"
    assert train_mod.model_id("modernbert-base", "short") == "modernbert-base-authorship-short-4d"
    assert train_mod.record_file(variant="short", candidate="modernbert-base").parts[-2:] == (
        "modernbert-base", "training_record_short.json")
    with pytest.raises(ValueError):
        train_mod.recipe("no-such-model")
```

Add `import pytest` to the imports of `tests/test_authorship_train.py`, and in `test_short_variant_trains_on_short_texts_and_excerpts` change the call and final assertion to:

```python
    result = train_mod.train(ShortFake(), out_dir=tmp_path / "s", record=tmp_path / "rec.json",
                             data=df, variant="short")
    assert seen["words"].between(25, 99).all()
    assert seen["excerpts"] > 0 and result["train_excerpts"] == seen["excerpts"]
    assert result["model_id"] == "distilroberta-base-authorship-short-4d" and result["variant"] == "short"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_training_4d.py tests/test_authorship_train.py -q`
Expected: FAIL — `AttributeError: module 'src.bakeoff.neural' has no attribute 'export_merged'` and `TypeError` from `train()` on the new keywords.

- [ ] **Step 3: Implement `export_merged` and drop `on_epoch_end`**

In `src/bakeoff/neural.py`, after `apply_lora`, add:

```python
def export_merged(base_model: str, adapter_dir, out_dir):
    """Merge a saved LoRA adapter (and its trained head) into a fresh fp32 copy of the base model, on the
    CPU, and save a plain model with the adapter's tokenizer, so the detector loads it like any other."""
    from pathlib import Path

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out_dir = Path(out_dir)
    base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)
    return out_dir
```

In `FineTunedClassifier.fit`, remove the `on_epoch_end` parameter, its docstring sentence, and the block

```python
            if on_epoch_end is not None:
                self.model.eval()
                on_epoch_end(epoch, {"epoch": epoch, "model": self.model.state_dict(),
                                     "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict()})
                self.model.train()
```

and simplify the resume line to `first_epoch, skip = (resume["epoch"], resume["micro"]) if resume["micro"] < micro_batches else (resume["epoch"] + 1, 0)` (every state now comes from a check).

- [ ] **Step 4: Rewrite `src/authorship/train.py`**

```python
"""Train a screening candidate on the full train split, keeping the best validation check.

    python -m src.authorship.train --candidate modernbert-base [--variant main|short] [--epochs 3]

Validation runs every quarter epoch; training stops after two checks without improvement, at most
three epochs, and keeps the best check (Increment 4d). The recipe (base model, precision, optimizer,
LoRA) is the candidate's screening recipe, with length-grouped batches. LoRA adapters are merged into
the base weights before the artifact is recorded, so the detector loads a plain model. Weights go to
CORPUS_DIR/models/<model id>/ (outside the repo); the training record, which holds no text, goes to
data/benchmark/<version>/model/<candidate>/. Each check saves a resume state, so a crash costs at most
a quarter epoch. Only train and validation rows are read.

The short variant is the short-text specialist: the same recipe on 25-99-word texts (natural ones
plus sentence-aligned excerpts of longer ones) at 160 tokens.
"""

import argparse
import hashlib
import json
import logging
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..bakeoff.data import folds, load_fit_data, prepare
from ..bakeoff.metrics import at_threshold, auroc, threshold_at_fpr
from ..bakeoff.run import selection_data
from ..corpus.lock import read as read_lock
from ..corpus.paths import REPO_BENCHMARK_DIR, VERSION, corpus_dir, lock_path
from ..corpus.text import MODEL_TEXT_VIEW, text_view_fingerprint

logger = logging.getLogger(__name__)

TEXT_NORMALIZATION = MODEL_TEXT_VIEW
VARIANTS = {"main": {"max_tokens": 256, "record": "training_record.json"},
            "short": {"max_tokens": 160, "record": "training_record_short.json"}}
EPOCHS = 3            # at most
CHECKS_PER_EPOCH = 4  # validation every quarter epoch
PATIENCE = 2          # stop after this many checks without improvement


def model_id(candidate: str, variant: str = "main") -> str:
    return f"{candidate}-authorship-short-4d" if variant == "short" else f"{candidate}-authorship-4d"


def recipe(candidate: str) -> dict:
    """The candidate's screening settings (base model, learning rate, precision, optimizer, LoRA)."""
    from ..bakeoff.screen import CANDIDATES

    for spec in CANDIDATES:
        if spec["name"] == candidate:
            return dict(spec["kwargs"])
    raise ValueError(f"unknown candidate {candidate!r}")


def model_dir(model: str) -> Path:
    return corpus_dir() / "models" / model


def record_file(version: str = VERSION, variant: str = "main", candidate: str = "distilroberta-base") -> Path:
    return REPO_BENCHMARK_DIR / version / "model" / candidate / VARIANTS[variant]["record"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_digest(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def validation_metrics(val: pd.DataFrame, scores: np.ndarray) -> dict:
    y = val["y"].to_numpy()
    threshold = threshold_at_fpr(scores[y == 0], 0.01)
    return {"auroc": auroc(y, scores), "tpr_at_1pct_fpr": at_threshold(y, scores, threshold)["tpr"]}


def _order(history: dict) -> list:
    return sorted(history, key=lambda label: tuple(int(p) for p in label.split(".")))


def best_check(history: dict) -> str:
    """The check with the highest detection at 1% false AI, AUROC breaking ties; a full tie keeps the
    earlier check."""
    best = None
    for label in _order(history):
        m = history[label]
        if best is None or (m["tpr_at_1pct_fpr"], m["auroc"]) > (history[best]["tpr_at_1pct_fpr"],
                                                                 history[best]["auroc"]):
            best = label
    return best


def stale_checks(history: dict) -> int:
    """Checks run since the best one."""
    order = _order(history)
    return len(order) - 1 - order.index(best_check(history))


def train(classifier=None, *, candidate: str = "distilroberta-base", epochs: int = EPOCHS,
          checks_per_epoch: int = CHECKS_PER_EPOCH, patience: int = PATIENCE, out_dir: Optional[Path] = None,
          record: Optional[Path] = None, data: Optional[pd.DataFrame] = None, variant: str = "main") -> dict:
    """Fit on all train rows, check validation every quarter epoch, stop early, record the best check."""
    from ..bakeoff import neural

    spec = VARIANTS[variant]
    settings = recipe(candidate)
    out_dir = out_dir or model_dir(model_id(candidate, variant))
    record = record or record_file(variant=variant, candidate=candidate)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = data if data is not None else selection_data(load_fit_data())
    if variant == "short":
        from .excerpts import short_frame

        df = prepare(short_frame(df))
    val_fold = next(iter(folds(df)))
    fit, val = val_fold.fit, val_fold.calib
    clf = classifier or neural.FineTunedClassifier(max_train=None, epochs=epochs, text_view=TEXT_NORMALIZATION,
                                                   max_tokens=spec["max_tokens"], group_by_length=True,
                                                   **settings)

    metrics_file = out_dir / "checks.json"
    history = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
    # Plain pickle keeps this module importable without torch; tensors inside pickle natively.
    resume_file = out_dir / "resume.pkl"
    resume = pickle.loads(resume_file.read_bytes()) if resume_file.exists() else None
    if resume is not None:
        logger.info("resuming after epoch %d check %d", resume["epoch"], resume["check"])
    last = f"{epochs - 1}.{checks_per_epoch}"

    def on_check(state: dict) -> bool:
        label = f"{state['epoch']}.{state['check']}"
        check_dir = out_dir / f"check-{label}"
        clf.model.save_pretrained(check_dir)
        clf.tokenizer.save_pretrained(check_dir)
        history[label] = validation_metrics(val, clf.score(val["text"].tolist()))
        logger.info("check %s validation %s", label, history[label])
        metrics_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
        best = best_check(history)
        for old in out_dir.glob("check-*"):  # only the best check's weights are kept
            if old.name != f"check-{best}":
                shutil.rmtree(old)
        # Written last: a crash before this line re-runs and re-scores the check on resume.
        tmp = resume_file.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(state))
        tmp.replace(resume_file)  # atomic, so a crash mid-write never leaves a corrupt resume state
        return stale_checks(history) >= patience

    # A crash always leaves history short of the stopping condition (the process died before on_check
    # returned True), so only a run that already stopped counts as finished.
    finished = bool(history) and (stale_checks(history) >= patience or last in history)
    if not finished:
        clf.fit(fit, balance_sources=True, checks_per_epoch=checks_per_epoch, on_check=on_check, resume=resume)

    best = best_check(history)
    chosen = out_dir / f"check-{best}"
    lora = getattr(clf, "lora", None)
    if lora:
        chosen = neural.export_merged(clf.model_name, chosen, out_dir / f"merged-{best}")
    weights = next(p for p in (chosen / "model.safetensors", chosen / "pytorch_model.bin") if p.exists())
    lock = read_lock(lock_path())
    result = {
        "model_id": model_id(candidate, variant),
        "candidate": candidate,
        "variant": variant,
        "base_model": getattr(clf, "model_name", "unknown"),
        "recipe": settings,
        "precision": getattr(clf, "precision", "bf16"),
        "lora_merged": bool(lora),
        "chosen_check": best,
        "stopped_early": last not in history,
        "checks": history,
        "checks_per_epoch": checks_per_epoch,
        "patience": patience,
        "max_epochs": epochs,
        "artifact_path": str(chosen),
        "artifact_sha256": sha256_file(weights),
        "peak_train_vram_gb": getattr(clf, "peak_vram_gb", None),
        "train_rows": len(fit),
        "train_ids_sha256": ids_digest(fit["sample_id"]),
        "validation_rows": len(val),
        "train_excerpts": int(fit["excerpt"].sum()) if "excerpt" in fit else 0,
        "balance_sources": True,
        "group_by_length": getattr(clf, "group_by_length", True),
        "text_normalization": TEXT_NORMALIZATION,
        "text_view_sha256": text_view_fingerprint(TEXT_NORMALIZATION),
        "max_tokens": getattr(clf, "max_tokens", None),
        "corpus_lock_sha256": lock.get("sha256") if lock else None,
        "locked_split_reads": 0,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if resume_file.exists():
        resume_file.unlink()  # a finished run needs no resume state; it can be over 1 GB
    return result


def main(argv: Optional[list[str]] = None) -> int:
    from ..bakeoff.screen import CANDIDATES

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True, choices=[c["name"] for c in CANDIDATES])
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="main")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("transformers", "httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    print(json.dumps(train(candidate=args.candidate, epochs=args.epochs, variant=args.variant), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_training_4d.py tests/test_authorship_train.py tests/test_screening.py -q`
Expected: PASS.

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest -q -x > .superpowers/sdd/2026-09-28-increment-4d-training/suite-task2.txt 2>&1; tail -3 .superpowers/sdd/2026-09-28-increment-4d-training/suite-task2.txt`
Expected: all pass. (`calibrate.py` still imports `record_file` with its old defaults; the new default candidate keeps that call valid.)

- [ ] **Step 7: Commit**

```bash
git add src/bakeoff/neural.py src/authorship/train.py tests/test_authorship_train.py tests/test_training_4d.py
git commit -m "feat: train a screening candidate with quarter-epoch early stopping and merged LoRA"
```

---

### Task 3: Inference precision in the detector, registry and calibration

**Files:**
- Modify: `src/authorship/detector.py` (sealed)
- Modify: `src/authorship/registry.py` (sealed)
- Modify: `src/authorship/calibrate.py`
- Modify: `tests/test_authorship_registry.py` (fixtures record a precision)
- Test: `tests/test_training_4d.py`

**Interfaces:**
- Consumes: training record keys `precision`, `candidate`, `artifact_path`, `artifact_sha256`, `max_tokens`, `text_normalization` (Task 2); `train.record_file(..., candidate=)`.
- Produces: `detector.PRECISIONS = ("bf16", "fp16", "fp32")`; `AuthorshipModelDetector(..., precision: str = "bf16")` with attribute `precision`; `registry.LEGACY_PRECISION = "bf16"`; registry entry key `precision` (main) and `short_model.precision`.
- Produces: `calibrate.calibration_file(version=VERSION, candidate="distilroberta-base") -> Path` = `data/benchmark/<version>/model/<candidate>/calibration.json`; `calibrate.calibrate(..., candidate="distilroberta-base")`; CLI `--candidate`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_training_4d.py`:

```python
def test_the_detector_rejects_an_unknown_precision_before_loading():
    from src.authorship.detector import AuthorshipModelDetector

    with pytest.raises(ValueError):
        AuthorshipModelDetector("unused", platt=(1.0, 0.0), precision="int8")
    assert AuthorshipModelDetector("unused", platt=(1.0, 0.0)).precision == "bf16"


def test_the_detector_autocasts_in_its_recorded_precision():
    torch = pytest.importorskip("torch")
    from src.authorship.detector import AuthorshipModelDetector

    def dtype(p):
        return AuthorshipModelDetector("unused", platt=(1.0, 0.0), precision=p).autocast_dtype()

    assert dtype("fp16") is torch.float16 and dtype("bf16") is torch.bfloat16 and dtype("fp32") is None


def test_a_fp16_detector_scores_on_the_cpu(tmp_path):
    pytest.importorskip("torch")
    from src.authorship.detector import AuthorshipModelDetector

    detector = AuthorshipModelDetector(tiny_encoder(tmp_path / "tiny"), platt=(1.0, 0.0), max_tokens=16,
                                       device="cpu", text_view="collapse_whitespace", precision="fp16")
    raw, emb = detector.raw_and_embeddings(["the chart labels vocab_a", "river stone"])
    assert raw.shape == (2,) and emb.shape[0] == 2


def test_calibration_passes_the_training_precision_and_writes_per_candidate():
    from src.authorship import calibrate

    training = {"artifact_path": "unused", "text_normalization": "collapse_whitespace", "max_tokens": 16}
    assert calibrate._detector({**training, "precision": "fp16"}).precision == "fp16"
    assert calibrate._detector(training).precision == "bf16"  # records trained before 4d
    assert calibrate.calibration_file(candidate="deberta-v3-base").parts[-2:] == ("deberta-v3-base",
                                                                                   "calibration.json")
```

In `tests/test_authorship_registry.py`, add `"precision": "bf16",` to the record in `_record` (after `"model_id": "m",`) and `"precision": "bf16",` to the `short_model` dict in `_with_short` (after `"max_tokens": 160,`), then append:

```python
def test_a_record_without_an_inference_precision_stays_closed(tmp_path):
    record = _record(tmp_path)
    del record["precision"]
    assert registry.verify(_sealed(record), tmp_path) == (False, "no recorded inference precision")
    record["precision"] = "int8"
    assert registry.verify(_sealed(record), tmp_path) == (False, "no recorded inference precision")


def test_a_short_model_without_an_inference_precision_stays_closed(tmp_path):
    record = _with_short(tmp_path, _record(tmp_path))
    del record["short_model"]["precision"]
    assert registry.verify(_sealed(record), tmp_path) == (
        False, "no recorded inference precision for the short-text model")


def test_the_service_scores_in_the_recorded_precision(tmp_path):
    record = _with_short(tmp_path, _record(tmp_path))
    record["precision"], record["short_model"]["precision"] = "fp16", "fp32"
    service = registry.service_from_record(_sealed(record), tmp_path, validated=True)
    assert service.detector.main.precision == "fp16" and service.detector.short.precision == "fp32"


def test_a_pre_4d_record_scores_under_bf16_as_it_was_calibrated(tmp_path):
    record = _record(tmp_path)
    del record["precision"]
    service = registry.service_from_record(_sealed(record), tmp_path, validated=True)
    assert service.detector.precision == registry.LEGACY_PRECISION == "bf16"
    assert registry.verify(record, tmp_path)[0] is False  # the gate itself stays closed
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_training_4d.py tests/test_authorship_registry.py -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'precision'` and the registry precision tests fail (verify returns "validated").

- [ ] **Step 3: Implement in `src/authorship/detector.py`**

After `WINDOW_WORDS = 160 ...` add:

```python
PRECISIONS = ("bf16", "fp16", "fp32")  # GPU autocast dtype for inference; fp32 means none, and the CPU runs fp32
```

Change `AuthorshipModelDetector.__init__` to take `precision: str = "bf16"` (last keyword) and, before `super().__init__`'s following lines finish, validate it:

```python
    def __init__(self, model_dir: Path, platt: Tuple[float, float], gate: Optional[OodGate] = None,
                 max_tokens: int = 256, batch_size: int = 32, device: Optional[str] = None,
                 text_view: Optional[str] = None, expected_files_sha256: Optional[str] = None,
                 precision: str = "bf16") -> None:
        super().__init__(threshold=0.5)
        from ..corpus.text import text_view as view_of

        if precision not in PRECISIONS:
            # Read from the registry record (Increment 4d); fail here, before any scoring.
            raise ValueError(f"unknown inference precision {precision!r}; use one of {PRECISIONS}")
        self.precision = precision
        self.text_view = text_view
```

(keep the rest of `__init__` unchanged). Add the method:

```python
    def autocast_dtype(self):
        """The GPU autocast dtype for the recorded precision; None means plain fp32."""
        import torch

        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[self.precision]
```

In `raw_and_embeddings`, replace

```python
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
```

with

```python
                with torch.autocast(device.type, dtype=dtype or torch.float32,
                                    enabled=device.type == "cuda" and dtype is not None):
```

and set `dtype = self.autocast_dtype()` next to `device = next(self._model.parameters()).device`.

- [ ] **Step 4: Implement in `src/authorship/registry.py`**

After `POLICY_VERSION = "v2.2.0"` add:

```python
# Records calibrated before Increment 4d carry no precision: the detector then hard-coded bf16 autocast,
# so that is how they are scored (Plan C scores today's models through their own records). Such a
# record's gate stays closed.
LEGACY_PRECISION = "bf16"
```

In `verify`, after the `text view changed since calibration` check, add:

```python
    from .detector import PRECISIONS

    if record.get("precision") not in PRECISIONS:
        return False, "no recorded inference precision"
    short = record.get("short_model")
    if short and short.get("precision") not in PRECISIONS:
        return False, "no recorded inference precision for the short-text model"
```

In `service_from_record`, give the inner `model` helper a `precision` argument and pass it through:

```python
    def model(directory: Path, platt, ood: dict, bank: Path, max_tokens: int, artifact: dict,
              precision: str) -> AuthorshipModelDetector:
        gate = OodGate(np.load(bank), ood["k"], ood["threshold"]) if bank.exists() else None
        return AuthorshipModelDetector(directory, tuple(platt), gate, max_tokens=max_tokens,
                                       text_view=view,
                                       expected_files_sha256=artifact.get("files_sha256"),
                                       precision=precision)

    detector = model(artifact_dir(record, root), record["calibration"]["platt"], record["calibration"]["ood"],
                     bank_path(record, root), record.get("max_tokens", 256), record["artifact"],
                     record.get("precision", LEGACY_PRECISION))
```

and in the short-model branch add `short.get("precision", LEGACY_PRECISION)` as the last argument of its `model(...)` call.

- [ ] **Step 5: Implement in `src/authorship/calibrate.py`**

Change `calibration_file` and `_detector`:

```python
def calibration_file(version: str = VERSION, candidate: str = "distilroberta-base") -> Path:
    return REPO_BENCHMARK_DIR / version / "model" / candidate / "calibration.json"


def _detector(training: dict) -> AuthorshipModelDetector:
    return AuthorshipModelDetector(Path(training["artifact_path"]), platt=(1.0, 0.0),
                                   max_tokens=training.get("max_tokens") or 256,
                                   text_view=training.get("text_normalization"),
                                   precision=training.get("precision", registry.LEGACY_PRECISION))
```

Give `calibrate` a `candidate: str = "distilroberta-base"` keyword; read records with `training_record_file(candidate=candidate)` and `training_record_file(variant="short", candidate=candidate)`; write `out = calibration_file(candidate=candidate)`; in `short_entry` add `"precision": short_training.get("precision", registry.LEGACY_PRECISION),` and set its `"training_record"` to `f"data/benchmark/{VERSION}/model/{candidate}/training_record_short.json"`; in `entry` add `"precision": training.get("precision", registry.LEGACY_PRECISION),` after `"max_tokens"`, set `"report"` to `f"data/benchmark/{VERSION}/model/{candidate}/calibration.json"` and `"training_record"` to `f"data/benchmark/{VERSION}/model/{candidate}/training_record.json"`. In `main`, add `parser.add_argument("--candidate", required=True)` and pass `candidate=args.candidate`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_training_4d.py tests/test_authorship_registry.py tests/test_ralph_r07.py tests/test_authorship_support.py -q`
Expected: PASS.

- [ ] **Step 7: Run the whole suite**

Run: `python -m pytest -q > .superpowers/sdd/2026-09-28-increment-4d-training/suite-task3.txt 2>&1; tail -3 .superpowers/sdd/2026-09-28-increment-4d-training/suite-task3.txt`
Expected: all pass. Tests that compare `scoring_code_sha256()` against a stored value compute it afresh, so the sealed-code edit moves no test.

- [ ] **Step 8: Commit**

```bash
git add src/authorship/detector.py src/authorship/registry.py src/authorship/calibrate.py tests/test_authorship_registry.py tests/test_training_4d.py
git commit -m "feat: record the inference precision in the registry and score in it"
```

---

### Task 4: Perplexity stack measurement

**Files:**
- Create: `src/authorship/stack.py`
- Create: `tests/test_optional_steps.py`

**Interfaces:**
- Consumes: `calibrate.load_split`, `calibrate.with_excerpts`, `calibrate.short_rows`, `calibrate._detector` (Task 3), `train.record_file` (Task 2), `neural.ZeroShotScorer`.
- Produces: `stack.half_of(sample_ids: pd.Series) -> np.ndarray`, `stack.stack_gain(frame: pd.DataFrame) -> dict` (frame columns: `sample_id`, `y`, `raw`, `binoculars`, `fast_base`, `fast_instruct`), `stack.adopted(gain_points: float, limits: dict) -> bool`, `stack.run(candidate: str) -> dict`; report at `data/benchmark/v4/model/<candidate>/stack.json`.

- [ ] **Step 1: Write the failing tests**

`tests/test_optional_steps.py`:

```python
"""Increment 4d optional steps, measured on validation: the perplexity stack and style-aware thresholds."""

import numpy as np
import pandas as pd

from src.authorship import stack

OK_LIMITS = {"peak_vram_gb": 3.0, "p95_ms": 300.0, "sheet_seconds": 90.0}


def test_halves_keep_excerpts_with_their_parent():
    ids = pd.Series([f"s{i}" for i in range(200)] + [f"s{i}:x" for i in range(200)])
    half = stack.half_of(ids)
    assert (half[:200] == half[200:]).all()
    assert 60 < half[:200].sum() < 140


def _frame(signal: float, n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    y = np.arange(n) % 2
    return pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "y": y,
                         "raw": y * 1.0 + rng.normal(size=n), "binoculars": y * signal + rng.normal(size=n),
                         "fast_base": rng.normal(size=n), "fast_instruct": rng.normal(size=n)})


def test_the_stack_is_adopted_when_a_zero_shot_score_adds_detection():
    result = stack.stack_gain(_frame(signal=2.0))
    assert result["gain_points"] >= 2.0
    assert stack.adopted(result["gain_points"], OK_LIMITS)


def test_the_stack_is_dropped_when_zero_shot_scores_are_noise():
    result = stack.stack_gain(_frame(signal=0.0))
    assert result["gain_points"] < 2.0
    assert not stack.adopted(result["gain_points"], OK_LIMITS)


def test_adoption_needs_every_gpu_limit():
    assert not stack.adopted(5.0, {**OK_LIMITS, "peak_vram_gb": 3.6})
    assert not stack.adopted(5.0, {**OK_LIMITS, "p95_ms": 600.0})
    assert not stack.adopted(5.0, {**OK_LIMITS, "sheet_seconds": 130.0})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_optional_steps.py -q`
Expected: FAIL — `ImportError: cannot import name 'stack'`.

- [ ] **Step 3: Implement `src/authorship/stack.py`**

```python
"""Increment 4d optional step: does a perplexity stack add detection? Measured on validation only.

    python -m src.authorship.stack --candidate modernbert-base

For each trained model (the main model on 100+-word validation texts, the short specialist on
25-99-word texts and excerpts), a logistic stack combines the classifier's raw score with the
Increment 3 zero-shot scores (Qwen2.5-0.5B observer and its instruct performer: Binoculars and
Fast-DetectGPT), all read on the model's text view. The stack is fitted on one half of validation and
measured on the other; halves split by parent text, so an excerpt never sits apart from its source.
It is adopted only if it adds at least 2 points of detection at 1% false AI and the GPU limits hold
with every model loaded (pre-registration). Writes a text-free report; reads only validation rows.
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..bakeoff.metrics import at_threshold, threshold_at_fpr
from ..corpus.paths import REPO_BENCHMARK_DIR, VERSION
from ..corpus.text import MODEL_TEXT_VIEW, text_view

logger = logging.getLogger(__name__)

FEATURES = ("raw", "binoculars", "fast_base", "fast_instruct")
MIN_GAIN_POINTS = 2.0
GPU_PEAK_GB, GPU_P95_MS, GPU_SHEET_S = 3.5, 500.0, 120.0
SHEET_TEXTS = 3000  # a 500-row sheet with six text columns


def half_of(sample_ids: pd.Series) -> np.ndarray:
    """0 or 1 per row, by a stable hash of the parent text's id (an excerpt's id is its parent's plus ':x')."""
    parents = sample_ids.astype(str).str.removesuffix(":x")
    return parents.map(lambda s: int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % 2).to_numpy()


def detection(y: np.ndarray, scores: np.ndarray) -> float:
    """Share of AI texts above the threshold that keeps 1% of these humans above it."""
    return float(at_threshold(y, scores, threshold_at_fpr(scores[y == 0], 0.01))["tpr"])


def stack_gain(frame: pd.DataFrame) -> dict:
    """Fit the stack on half 0; measure classifier-alone and stacked detection at 1% false AI on half 1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    half = half_of(frame["sample_id"])
    fit, test = half == 0, half == 1
    features = frame[list(FEATURES)]
    features = features.fillna(features[fit].median())  # zero-shot scores are NaN for one-token texts
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features[fit], frame.loc[fit, "y"])
    y = frame.loc[test, "y"].to_numpy()
    alone = detection(y, frame.loc[test, "raw"].to_numpy())
    stacked = detection(y, model.decision_function(features[test]))
    return {"fit_rows": int(fit.sum()), "test_rows": int(test.sum()),
            "test_humans": int((y == 0).sum()), "test_ais": int((y == 1).sum()),
            "classifier_tpr_at_1pct_fpr": round(alone, 4), "stack_tpr_at_1pct_fpr": round(stacked, 4),
            "gain_points": round(100 * (stacked - alone), 2),
            "coefficients": dict(zip(FEATURES, np.round(model[-1].coef_[0], 4).tolist()))}


def adopted(gain_points: float, limits: dict) -> bool:
    return (gain_points >= MIN_GAIN_POINTS and limits["peak_vram_gb"] <= GPU_PEAK_GB
            and limits["p95_ms"] <= GPU_P95_MS and limits["sheet_seconds"] <= GPU_SHEET_S)


def gpu_limits(detectors: list, scorer, texts: list) -> dict:
    """Peak GPU memory with every model loaded, p95 single-text time for the main model plus the
    zero-shot pair, and a 500 x 6 sheet estimate from batched throughput."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the GPU limits are measured on the GPU")
    for d in detectors:
        d._ensure_loaded()
    torch.cuda.reset_peak_memory_stats()
    singles = []
    for t in texts[:50]:
        start = time.perf_counter()
        detectors[0].raw_and_embeddings([t])
        scorer.score_frame([t])
        singles.append(1000 * (time.perf_counter() - start))
    batch = texts[:500]
    start = time.perf_counter()
    detectors[0].raw_and_embeddings(batch)
    scorer.score_frame(batch)
    per_text = (time.perf_counter() - start) / len(batch)
    return {"peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "p95_ms": round(float(np.percentile(singles, 95)), 1),
            "sheet_seconds": round(per_text * SHEET_TEXTS, 1)}


def report_file(candidate: str) -> Path:
    return REPO_BENCHMARK_DIR / VERSION / "model" / candidate / "stack.json"


def run(candidate: str) -> dict:
    from ..bakeoff.neural import ZeroShotScorer
    from .calibrate import _detector, load_split, short_rows, with_excerpts
    from .train import record_file

    val = load_split("val")
    val = pd.concat([val.assign(excerpt=False), with_excerpts(val)], ignore_index=True)
    view = text_view(MODEL_TEXT_VIEW)
    scorer = ZeroShotScorer()
    report = {"rule": f"adopted when the stack adds {MIN_GAIN_POINTS} points of detection at 1% false AI on "
                      f"the held-out half and the GPU limits hold", "models": {}}
    detectors = []
    for variant, rows in (("main", val[val["words"] >= 100]), ("short", short_rows(val))):
        path = record_file(variant=variant, candidate=candidate)
        if not path.exists():
            continue
        training = json.loads(path.read_text(encoding="utf-8"))
        detector = _detector(training)
        detectors.append(detector)
        logger.info("%s: scoring %d validation texts", variant, len(rows))
        raw, _ = detector.raw_and_embeddings(rows["text"].tolist())
        zero_shot = scorer.score_frame([view(t) for t in rows["text"]])
        frame = pd.concat([rows[["sample_id", "y"]].reset_index(drop=True), zero_shot], axis=1).assign(raw=raw)
        report["models"][variant] = {"model_id": training["model_id"], **stack_gain(frame)}
    report["gpu_limits"] = gpu_limits(detectors, scorer, val["text"].sample(n=500, random_state=0).tolist())
    for entry in report["models"].values():
        entry["adopted"] = adopted(entry["gain_points"], report["gpu_limits"])
    report["locked_split_reads"] = 0
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    out = report_file(candidate)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("transformers", "httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    print(json.dumps(run(args.candidate), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_optional_steps.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/authorship/stack.py tests/test_optional_steps.py
git commit -m "feat: measure the perplexity stack on held-out validation (Increment 4d option)"
```

---

### Task 5: Register classifier and style-aware threshold measurement

**Files:**
- Create: `src/authorship/registers.py`
- Test: `tests/test_optional_steps.py`

**Interfaces:**
- Consumes: `calibration.choose_thresholds`, `calibration.band_of`, `calibration.slice_target`, `calibration.probability`, `calibration.MIN_HUMANS`; `metrics.upper_bound`; `calibrate.load_split`, `calibrate.with_excerpts`, `calibrate.calibration_file(candidate=)` (Task 3); `train.record_file(..., candidate=)` (Task 2); the calibration score caches `calibration_scores_<sha[:12]>.npz` beside each artifact (written by `calibrate._score`).
- Produces: `registers.REGISTERS`, `registers.register_label(source: str, domain: str) -> str`, `registers.RegisterClassifier` (`fit(texts, labels) -> self`, `predict(texts) -> np.ndarray`), `registers.cell_bars(val, pooled, max_fpr, max_fnr) -> dict`, `registers.register_gain(val, pooled, max_fpr, max_fnr) -> dict`, `registers.run(candidate) -> dict`; report at `data/benchmark/v4/model/<candidate>/registers.json`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_optional_steps.py`:

```python
import pytest

from src.authorship import registers

# Every core (source, domain) pair that trains in corpus v4: v3's pairs plus the new sources.
V4_TRAINING_PAIRS = (
    [("aita2026", "aita"), ("oasst1", "assistant_reply"), ("arena140k", "chat")]
    + [("beemo", d) for d in ("closed_qa", "generation", "open_qa", "rewrite", "summarize")]
    + [("dolly", d) for d in ("brainstorming", "classification", "closed_qa", "creative_writing", "general_qa",
                              "information_extraction", "open_qa", "summarization")]
    + [("hc3", d) for d in ("finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai")]
    + [("mage", d) for d in ("cmv", "cnn", "dialogsum", "eli5", "hswag", "imdb", "pubmed", "roct", "sci_gen",
                             "squad", "tldr", "wp", "xsum", "yelp")]
    + [("mirage", d) for d in ("academic", "comment", "news", "website")]
    + [("no_robots", d) for d in ("brainstorm", "chat", "classify", "closed_qa", "coding", "extract",
                                  "generation", "open_qa", "rewrite", "summarize")]
    + [("raid", d) for d in ("abstracts", "books", "news", "poetry", "recipes", "reddit", "reviews", "wiki")]
    + [("stackexchange", d) for d in ("cooking", "fitness", "gardening", "interpersonal", "money", "outdoors",
                                      "parenting", "pets")]
    + [("maga", d) for d in ("amazon_reviews", "cc_news", "natural_questions", "npr_news", "reddit", "s2orc",
                             "trustpilot_reviews", "wikipedia", "yahoo_answers")]
    + [("dactyl", d) for d in ("abstracts", "news", "reviews", "student_essays", "tweets")]
)


def test_every_training_pair_has_a_register():
    assert {registers.register_label(s, d) for s, d in V4_TRAINING_PAIRS} == set(registers.REGISTERS)


def test_an_unmapped_pair_fails_loudly():
    with pytest.raises(KeyError):
        registers.register_label("mage", "no_such_domain")
    with pytest.raises(KeyError):
        registers.register_label("no_such_source", "x")


def test_the_register_classifier_learns_registers_from_text():
    vocab = {"qa_forum": "answer question thread upvote", "assistant": "certainly here is a helpful list",
             "news_expository": "officials said on tuesday report", "essay_academic": "we propose results method",
             "social_review": "loved it five stars lol"}
    rng = np.random.default_rng(0)
    texts, labels = [], []
    for register, words in vocab.items():
        for _ in range(60):
            texts.append(" ".join(rng.permutation(words.split() + ["common", "words", "here"])))
            labels.append(register)
    clf = registers.RegisterClassifier().fit(texts, labels)
    assert (clf.predict(texts) == np.array(labels)).mean() > 0.95


def _cells(n_humans_a: int, n_humans_b: int) -> pd.DataFrame:
    """Band 100, two registers. Register a's humans score low, so its own bar can sit below the pooled bar."""
    rng = np.random.default_rng(1)
    parts = []
    for register, humans, human_scores in (("a", n_humans_a, (0.0, 0.3)), ("b", n_humans_b, (0.0, 0.95))):
        parts.append(pd.DataFrame({"register": register, "y": 0, "words": 150, "source_dataset": f"src_{register}",
                                   "prob": rng.uniform(*human_scores, size=humans)}))
        parts.append(pd.DataFrame({"register": register, "y": 1, "words": 150, "source_dataset": f"src_{register}",
                                   "prob": rng.uniform(0.4, 1.0, size=humans)}))
    return pd.concat(parts, ignore_index=True)


def test_a_cell_with_600_humans_gets_its_own_bar_and_a_smaller_cell_keeps_the_pooled_bar():
    val = _cells(700, 500)
    bars = registers.cell_bars(val, {100: 0.97}, max_fpr=0.008, max_fnr=0.08)
    assert bars[(100, "a")]["own"] and bars[(100, "a")]["ai_min"] < 0.97
    assert not bars[(100, "b")]["own"] and bars[(100, "b")]["ai_min"] == 0.97


def test_style_thresholds_are_adopted_only_with_gain_and_every_cell_in_target():
    good = registers.register_gain(_cells(700, 500), {100: 0.97}, max_fpr=0.008, max_fnr=0.08)
    assert good["gain_points"] >= 2.0 and good["every_cell_in_target"] and good["adopted"]
    # A pooled bar that lets register b's humans through breaks that cell's false-AI target.
    bad = registers.register_gain(_cells(700, 500), {100: 0.5}, max_fpr=0.008, max_fnr=0.08)
    assert not bad["every_cell_in_target"] and not bad["adopted"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_optional_steps.py -q`
Expected: FAIL — `ImportError: cannot import name 'registers'`.

- [ ] **Step 3: Implement `src/authorship/registers.py`**

```python
"""Increment 4d optional step: style-aware thresholds, measured on validation only.

    python -m src.authorship.registers --candidate modernbert-base

A TF-IDF logistic classifier assigns each text one of five registers; it trains on train rows
labelled by source and domain. At check time the register would be predicted from the text, never
taken from the source, so validation rows are assigned by prediction too. A bar is set per length band
x register cell on that cell's own validation humans at the pre-registered targets; a cell with fewer
than 600 validation humans keeps the pooled band bar. The step is adopted only if it adds at least 2
points of validation detection with every cell within target. Writes a text-free report. Run after
calibration: it reuses calibration's cached validation scores and bars.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..bakeoff.metrics import upper_bound
from ..corpus.paths import REPO_BENCHMARK_DIR, VERSION
from ..corpus.text import MODEL_TEXT_VIEW, text_view
from . import calibration as cal

logger = logging.getLogger(__name__)

REGISTERS = ("qa_forum", "assistant", "news_expository", "essay_academic", "social_review")
MIN_CELL_HUMANS = 600
MIN_GAIN_POINTS = 2.0
FIT_PER_CELL = 10_000  # train rows per (register, label) for the classifier

# A source's register, then (source, domain) pairs that differ. Creative writing (stories, poetry,
# books) has no register of its own; it reads as long-form prose, so it sits with essays.
_SOURCE_REGISTER = {
    "aita2026": "social_review", "arena140k": "assistant", "beemo": "assistant", "dolly": "assistant",
    "hc3": "qa_forum", "no_robots": "assistant", "oasst1": "assistant", "stackexchange": "qa_forum",
}
_PAIR_REGISTER = {
    ("hc3", "wiki_csai"): "news_expository",
    ("mage", "cmv"): "essay_academic", ("mage", "cnn"): "news_expository", ("mage", "dialogsum"): "news_expository",
    ("mage", "eli5"): "qa_forum", ("mage", "hswag"): "essay_academic", ("mage", "imdb"): "social_review",
    ("mage", "pubmed"): "essay_academic", ("mage", "roct"): "essay_academic", ("mage", "sci_gen"): "essay_academic",
    ("mage", "squad"): "news_expository", ("mage", "tldr"): "news_expository", ("mage", "wp"): "essay_academic",
    ("mage", "xsum"): "news_expository", ("mage", "yelp"): "social_review",
    ("mirage", "academic"): "essay_academic", ("mirage", "comment"): "social_review",
    ("mirage", "news"): "news_expository", ("mirage", "website"): "news_expository",
    ("raid", "abstracts"): "essay_academic", ("raid", "books"): "essay_academic",
    ("raid", "news"): "news_expository", ("raid", "poetry"): "essay_academic",
    ("raid", "recipes"): "news_expository", ("raid", "reddit"): "qa_forum",
    ("raid", "reviews"): "social_review", ("raid", "wiki"): "news_expository",
    ("maga", "amazon_reviews"): "social_review", ("maga", "cc_news"): "news_expository",
    ("maga", "natural_questions"): "qa_forum", ("maga", "npr_news"): "news_expository",
    ("maga", "reddit"): "qa_forum", ("maga", "s2orc"): "essay_academic",
    ("maga", "trustpilot_reviews"): "social_review", ("maga", "wikipedia"): "news_expository",
    ("maga", "yahoo_answers"): "qa_forum",
    ("dactyl", "abstracts"): "essay_academic", ("dactyl", "news"): "news_expository",
    ("dactyl", "reviews"): "social_review", ("dactyl", "student_essays"): "essay_academic",
    ("dactyl", "tweets"): "social_review",
}


def register_label(source: str, domain: str) -> str:
    """The register a (source, domain) trains as; an unmapped pair fails, so no source trains under a
    guessed register."""
    if (source, domain) in _PAIR_REGISTER:
        return _PAIR_REGISTER[(source, domain)]
    if source in _SOURCE_REGISTER:
        return _SOURCE_REGISTER[source]
    raise KeyError(f"no register for {source}:{domain}")


class RegisterClassifier:
    """Word 1-2-gram TF-IDF and a multinomial logistic regression: CPU only, well under a millisecond a text."""

    def fit(self, texts, labels) -> "RegisterClassifier":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000,
                                                   sublinear_tf=True),
                                   LogisticRegression(max_iter=2000, C=4.0))
        self.model.fit(list(texts), list(labels))
        return self

    def predict(self, texts) -> np.ndarray:
        return np.asarray(self.model.predict(list(texts)))


def cell_bars(val: pd.DataFrame, pooled: dict, max_fpr: float, max_fnr: float) -> dict:
    """ai_min per (band, register): the cell's own bar where it has MIN_CELL_HUMANS validation humans and
    calibration supports it there, else the pooled band bar. val needs y, words, prob, register and
    source_dataset."""
    val = val.assign(band=val["words"].map(cal.band_of))
    out = {}
    for (band, register), cell in val.groupby(["band", "register"]):
        humans = int((cell["y"] == 0).sum())
        bar, own = pooled.get(band), False
        if humans >= MIN_CELL_HUMANS:
            fitted = {b.lower_words: b for b in cal.choose_thresholds(cell, max_fpr=max_fpr, max_fnr=max_fnr)}
            mine = fitted.get(band)
            if mine is not None and mine.supported:
                bar, own = mine.ai_min, True
        out[(int(band), register)] = {"ai_min": bar, "own": own, "humans": humans,
                                      "ais": int((cell["y"] == 1).sum())}
    return out


def _flagged(val: pd.DataFrame, bars: dict) -> np.ndarray:
    keys = zip(val["words"].map(cal.band_of), val["register"])
    ai_min = np.array([np.inf if bars.get((b, r), {}).get("ai_min") is None else bars[(b, r)]["ai_min"]
                       for b, r in keys], dtype=float)
    return val["prob"].to_numpy() >= ai_min


def register_gain(val: pd.DataFrame, pooled: dict, max_fpr: float, max_fnr: float) -> dict:
    """Validation detection under pooled band bars and under per-cell bars, and whether every cell with
    MIN_HUMANS humans keeps its false-AI bound within target under the per-cell bars."""
    bars = cell_bars(val, pooled, max_fpr, max_fnr)
    pooled_bars = {k: {"ai_min": pooled.get(k[0])} for k in bars}
    ai = val["y"].to_numpy() == 1
    before = float(_flagged(val, pooled_bars)[ai].mean())
    after = float(_flagged(val, bars)[ai].mean())
    flagged = _flagged(val, bars)
    cells, in_target = [], True
    for (band, register), entry in bars.items():
        rows = (val["words"].map(cal.band_of).to_numpy() == band) & (val["register"].to_numpy() == register)
        humans = rows & ~ai
        n = int(humans.sum())
        false_ai = int((flagged & humans).sum())
        bound = upper_bound(false_ai, n)
        ok = n < cal.MIN_HUMANS or bound <= cal.slice_target(max_fpr, n)
        in_target &= ok
        cells.append({"band": band, "register": register, **entry, "false_ai": false_ai,
                      "false_ai_upper95": bound, "within_target": ok})
    gain = round(100 * (after - before), 2)
    return {"pooled_detection": round(before, 4), "register_detection": round(after, 4), "gain_points": gain,
            "every_cell_in_target": bool(in_target), "cells": cells,
            "adopted": bool(gain >= MIN_GAIN_POINTS and in_target)}


def report_file(candidate: str) -> Path:
    return REPO_BENCHMARK_DIR / VERSION / "model" / candidate / "registers.json"


def _served_probabilities(val: pd.DataFrame, candidate: str, calibration: dict) -> pd.Series:
    """Each validation row's probability from the model that serves its band, read from calibration's
    cached raw scores and mapped through that model's recorded Platt map."""
    from .train import record_file

    serving = {b["lower_words"]: b.get("model", "main") for b in calibration["policy_bands"]}
    probs = {}
    for variant in ("main", "short"):
        path = record_file(variant=variant, candidate=candidate)
        if variant not in calibration or not path.exists():
            continue
        training = json.loads(path.read_text(encoding="utf-8"))
        cache = Path(training["artifact_path"]).parent / f"calibration_scores_{training['artifact_sha256'][:12]}.npz"
        scores = np.load(cache)
        probs[variant] = pd.Series(cal.probability(scores["val_raw"], *calibration[variant]["platt"]),
                                   index=scores["val_ids"].astype(str))
    band = val["words"].map(cal.band_of)
    return pd.Series([probs[serving.get(b, "main")].get(sid, np.nan) for b, sid in zip(band, val["sample_id"])],
                     index=val.index)


def run(candidate: str) -> dict:
    from ..corpus.sources._common import take_per_stratum
    from .calibrate import calibration_file, load_split, with_excerpts

    calibration = json.loads(calibration_file(candidate=candidate).read_text(encoding="utf-8"))
    targets = calibration["targets"]
    view = text_view(MODEL_TEXT_VIEW)
    train = load_split("train")  # its columns include domain
    train = train.assign(register=[register_label(s, d) for s, d in zip(train["source_dataset"], train["domain"])])
    fit = take_per_stratum(train, ["register", "y"], FIT_PER_CELL, "sample_id")
    classifier = RegisterClassifier().fit(fit["text"].map(view), fit["register"])
    val = load_split("val")
    val = pd.concat([val.assign(excerpt=False), with_excerpts(val)], ignore_index=True)
    val = val.assign(register=classifier.predict(val["text"].map(view)),
                     prob=_served_probabilities(val, candidate, calibration)).dropna(subset=["prob"])
    pooled = {b["lower_words"]: b["ai_min"] for b in calibration["policy_bands"] if b["supported"]}
    report = register_gain(val, pooled, targets["max_false_ai_upper95"], targets["max_false_human_upper95"])
    report["classifier_fit_rows"] = len(fit)
    report["locked_split_reads"] = 0
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    out = report_file(candidate)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(args.candidate)
    print(json.dumps({k: report[k] for k in ("pooled_detection", "register_detection", "gain_points",
                                             "every_cell_in_target", "adopted")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_optional_steps.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Run the whole suite and commit**

Run: `python -m pytest -q > .superpowers/sdd/2026-09-28-increment-4d-training/suite-task5.txt 2>&1; tail -3 .superpowers/sdd/2026-09-28-increment-4d-training/suite-task5.txt`
Expected: all pass.

```bash
git add src/authorship/registers.py tests/test_optional_steps.py
git commit -m "feat: measure style-aware thresholds on validation (Increment 4d option)"
```

---

### Task 6: Run full training, calibration and the two measurements

Runs after Plan A Task 10 (corpus v4 built and audited, `test.lock.json` committed) and after the screening winner is recorded as a pre-registration amendment. One GPU job at a time; each run detached through `scratchpad\run-job.ps1` with `HF_HOME=D:\llm-authorship-corpus\hf-cache`.

**Files:**
- Commit: `data/benchmark/v4/model/<candidate>/{training_record.json, training_record_short.json, calibration.json, stack.json, registers.json}`, `models/registry.json` (4d branch only)
- Modify: `docs/bob/llm-authorship-checker/increment-4d-preregistration.md` (amendments)

- [ ] **Step 0: Resume smoke test on the GPU (final review recommendation)**

On about 2,000 training rows (`python -c` calling `train.train(candidate=..., data=<2k-row sample>, out_dir=<scratch>)`), kill the job mid-check and resume, once for the winner's recipe (8-bit AdamW for ModernBERT-base, fp16 with GradScaler for DeBERTa). Expected: the resumed run continues from the saved check, VRAM after the resume stays at the first run's level, and the record is written. Delete the scratch directory afterwards.

After the corpus build, run `register_label` over every (source, domain) of the v4 train split; an unmapped pair is fixed (test-first) before Step 3.

- [ ] **Step 1: Train the winner, main then short**

Run (detached, one after the other): `python -m src.authorship.train --candidate <winner> --variant main`, then `python -m src.authorship.train --candidate <winner> --variant short`.
Expected: each prints a record with `chosen_check`, `stopped_early`, `locked_split_reads: 0`. If screening's top two were within 2 points, train the runner-up the same way, then keep the one with the higher mean of main and short best-check `tpr_at_1pct_fpr` (the screening rule applied to full training; within 1 point the faster wins) and record the choice as an amendment before calibration.

- [ ] **Step 2: Calibrate**

Run: `python -m src.authorship.calibrate --candidate <chosen> --max-false-ai 0.008 --max-false-human 0.08`
Expected: `policy_bands`, `routing`, `peak_inference_vram_gb` ≤ 3.5 printed; `models/registry.json` rewritten with `benchmark: null`, `precision` recorded for both models.

- [ ] **Step 3: Measure the optional steps**

Run: `python -m src.authorship.stack --candidate <chosen>` then `python -m src.authorship.registers --candidate <chosen>`.
Expected: `stack.json` and `registers.json` written, each with an `adopted` verdict.

- [ ] **Step 4: Record the verdicts as an amendment, commit and push**

Add "Amendment N — full training and optional steps" to the pre-registration: the chosen model and check, calibration bands, and each optional step's measured gain and verdict. If either step is adopted, write Plan B2 (integration into sealed scoring code, a fresh calibration) before Plan C.

```bash
git add data/benchmark/v4/model models/registry.json docs/bob/llm-authorship-checker/increment-4d-preregistration.md
git commit -m "data: 4d full training, calibration and optional-step verdicts"
git push
```
