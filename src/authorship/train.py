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


# Full-training speed settings (approved 2026-09-28, Amendment 7): with its frozen base in half precision,
# DeBERTa-v3-large with LoRA fits without gradient checkpointing, which recomputes every layer's forward pass
# during backward (about 30% extra work). Micro-batches of 4 accumulated 4 times keep screening's effective
# batch of 16. "screening" trains exactly as screening did, the fallback if memory is too tight.
FAST_SETTINGS = {"deberta-v3-large-lora": {"batch_size": 4, "accumulate": 4, "gradient_checkpointing": False}}
# Amendment 12: "fast" filled the 4 GB card (Amendment 11). One micro-batch of 16 with checkpointing takes the
# same effective batch in one step: 8.71 rows/s against screening's 7.84, at a 2.1 GB peak.
FULL_BATCH_SETTINGS = {"deberta-v3-large-lora": {"batch_size": 16, "accumulate": 1}}
SETTINGS = ("fast", "screening", "fullbatch")


def settings_for(candidate: str, settings: str = "fast") -> dict:
    """The candidate's screening recipe, with the named settings' overrides ("screening" has none)."""
    if settings not in SETTINGS:
        raise ValueError(f"unknown settings {settings!r}; use one of {SETTINGS}")
    overrides = {"fast": FAST_SETTINGS, "fullbatch": FULL_BATCH_SETTINGS}.get(settings, {})
    return {**recipe(candidate), **overrides.get(candidate, {})}


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


def _write_resume(path: Path, state: dict) -> None:
    """Stream the resume state to disk (torch.save writes tensor by tensor, so no multi-GB copy is built in
    RAM) and swap it in atomically, so a crash mid-write never leaves a corrupt resume state."""
    import torch

    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _read_resume(path: Path) -> Optional[dict]:
    """The saved resume state, read onto the CPU: load_state_dict copies it to the GPU, and a second GPU copy
    would crowd a 4 GB card for the rest of the run."""
    if not path.exists():
        return None
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


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
          record: Optional[Path] = None, data: Optional[pd.DataFrame] = None, variant: str = "main",
          settings: str = "fullbatch") -> dict:
    """Fit on all train rows, check validation every quarter epoch, stop early, record the best check."""
    from ..bakeoff import neural

    spec = VARIANTS[variant]
    settings_name, settings = settings, settings_for(candidate, settings)
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

    # Checks and resume states belong to one run: reusing them for other data or settings would record old
    # weights under new provenance (final review I3).
    fingerprint = {"candidate": candidate, "variant": variant, "recipe": settings, "max_epochs": epochs,
                   "checks_per_epoch": checks_per_epoch, "patience": patience,
                   "train_ids_sha256": ids_digest(fit["sample_id"]), "validation_ids_sha256": ids_digest(val["sample_id"]),
                   "text_view_sha256": text_view_fingerprint(TEXT_NORMALIZATION),
                   "max_tokens": getattr(clf, "max_tokens", None)}
    run_file = out_dir / "run.json"
    if run_file.exists() and json.loads(run_file.read_text(encoding="utf-8")) != json.loads(json.dumps(fingerprint)):
        raise SystemExit(f"{out_dir} holds checks from a different run; move it aside to start fresh")
    run_file.write_text(json.dumps(fingerprint, indent=2, sort_keys=True), encoding="utf-8")

    metrics_file = out_dir / "checks.json"
    history = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
    resume_file = out_dir / "resume.pt"
    resume = _read_resume(resume_file)
    if resume is not None:
        logger.info("resuming after epoch %d check %d", resume["epoch"], resume["check"])
        # A check scored after the saved state is re-run and re-scored, so its old score can't count.
        saved = (resume["epoch"], resume["check"])
        history = {k: v for k, v in history.items() if tuple(int(p) for p in k.split(".")) <= saved}
    last = f"{epochs - 1}.{checks_per_epoch}"

    def on_check(state: dict) -> bool:
        label = f"{state['epoch']}.{state['check']}"
        check_dir = out_dir / f"check-{label}"
        clf.model.save_pretrained(check_dir)
        clf.tokenizer.save_pretrained(check_dir)
        if getattr(clf, "lora", None):
            # Base weights the checkpoint lacked (DeBERTa's pooler) start random and stay frozen; the merge
            # needs the same values the validated adapter model ran with (final review C2).
            neural.save_new_weights(getattr(clf, "new_weights", {}), check_dir)
        history[label] = validation_metrics(val, clf.score(val["text"].tolist()))
        logger.info("check %s validation %s", label, history[label])
        metrics_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
        # The resume state goes before any pruning: a crash in between re-runs this check from the last
        # saved state, and every check that state can reach still has its weights (final review C1).
        _write_resume(resume_file, state)
        best = best_check(history)
        for old in out_dir.glob("check-*"):  # only the best check's weights are kept
            if old.name != f"check-{best}":
                shutil.rmtree(old, ignore_errors=True)
        return stale_checks(history) >= patience

    # Only a run whose history already meets the stopping rule counts as finished; a crash before that
    # resumes from the last saved check.
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
        "settings": settings_name,
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
        "validation_ids_sha256": ids_digest(val["sample_id"]),  # Ralph R12-18
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
    parser.add_argument("--settings", choices=SETTINGS, default="fullbatch",  # Amendment 12
                        help="fast: no gradient checkpointing, same effective batch; screening: as screened; "
                             "fullbatch: the effective batch in one checkpointed micro-batch")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("transformers", "httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    print(json.dumps(train(candidate=args.candidate, epochs=args.epochs, variant=args.variant,
                           settings=args.settings), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
