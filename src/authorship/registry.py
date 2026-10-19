"""Model registry: which artifact is calibrated how, and whether it passed the locked benchmark.

The registry (committed, no text) replaces the hard-coded validation gate. A model is
validated only when its benchmark passed and the weights and embedding bank on disk still match
the recorded hashes; anything else keeps the service abstaining with MODEL_NOT_VALIDATED.

An optional `short_model` entry adds a short-text specialist for the word-count ranges it won on
validation; its weights and bank are verified the same way.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..corpus.paths import corpus_dir
from ..corpus.text import text_view_fingerprint
from ..detection import AuthorshipService, BandRule, DecisionPolicy

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "models" / "registry.json"
POLICY_VERSION = "v2.2.0"
# Records calibrated before Increment 4d carry no precision: the detector then hard-coded bf16 autocast,
# so that is how they are scored (the head-to-head scores today's models through their own records).
# Such a record's gate stays closed.
LEGACY_PRECISION = "bf16"


def registry_path() -> Path:
    """AUTHORSHIP_REGISTRY overrides the committed registry (tests point it at an empty path)."""
    return Path(os.getenv("AUTHORSHIP_REGISTRY", str(REGISTRY_PATH)))


def load(path: Optional[Path] = None) -> Optional[dict]:
    path = path or registry_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save(record: dict, path: Optional[Path] = None) -> None:
    path = path or registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Hashed afresh on every call: a cache keyed on size and mtime let a same-size rewrite with
    its mtime restored pass (Ralph R07-8)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(directory: Path) -> str:
    """One digest over every file in an artifact directory: weights, config, and tokenizer."""
    lines = [f"{p.relative_to(directory).as_posix()}\t{sha256_file(p)}"
             for p in sorted(directory.rglob("*")) if p.is_file()]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def artifact_dir(record: dict, root: Optional[Path] = None) -> Path:
    return (root or corpus_dir()) / record["artifact"]["relative_path"]


def bank_path(record: dict, root: Optional[Path] = None) -> Path:
    return (root or corpus_dir()) / record["calibration"]["ood"]["bank_relative_path"]


def weights_file(directory: Path) -> Path:
    return next((p for p in (directory / "model.safetensors", directory / "pytorch_model.bin") if p.exists()),
                directory / "model.safetensors")


def short_parts(record: dict, root: Optional[Path] = None) -> Optional[Tuple[Path, Path]]:
    """(artifact dir, bank) of the short-text specialist, or None when there is none."""
    short = record.get("short_model")
    if not short:
        return None
    base = root or corpus_dir()
    return base / short["artifact"]["relative_path"], base / short["ood"]["bank_relative_path"]


def _check(directory: Path, artifact: dict, bank: Path, bank_sha: str, name: str) -> Optional[str]:
    weights = weights_file(directory)
    if not weights.exists():
        return f"{name} artifact missing"
    if sha256_file(weights) != artifact["sha256"]:
        return f"{name} artifact hash mismatch"
    if not artifact.get("files_sha256"):
        return f"{name} has no recorded config and tokenizer hash"  # Ralph R08-4: never skip the check
    if directory_sha256(directory) != artifact["files_sha256"]:
        return f"{name} config or tokenizer files changed"
    if not bank.exists() or sha256_file(bank) != bank_sha:
        return f"{name} embedding bank missing or changed"
    return None


def _check_short(record: dict, root: Optional[Path] = None) -> Optional[str]:
    parts = short_parts(record, root)
    if not parts:
        return None
    short = record["short_model"]
    return _check(parts[0], short["artifact"], parts[1], short["ood"]["bank_sha256"], "short-text model")


def verify_files(record: dict, root: Optional[Path] = None) -> Tuple[bool, str]:
    """(ok, reason) for the recorded files alone: both models' weights, config, tokenizer, banks."""
    problem = _check(artifact_dir(record, root), record["artifact"], bank_path(record, root),
                     record["calibration"]["ood"]["bank_sha256"], "model") or _check_short(record, root)
    return (False, problem) if problem else (True, "files verified")


# Everything besides the model that decides an outcome: disguise handling, the decision policy and
# service, windowing, the language check, the registry's own wiring and the text views (Ralph R10-3).
_SCORING_CODE = ("src/detection/obfuscation.py", "src/detection/policy.py", "src/detection/service.py",
                 "src/detection/contracts.py", "src/authorship/detector.py", "src/authorship/excerpts.py",
                 "src/authorship/language.py", "src/authorship/registry.py", "src/corpus/text.py",
                 "src/authorship/calibration.py", "src/detectors/base.py",  # the Platt map too (Ralph R11-2)
                 "src/detection/__init__.py")  # the service's exports (Ralph R11-2, R12-18)
_REPO = Path(__file__).resolve().parents[2]


def text_sha256(path: Path) -> str:
    """SHA-256 of a text file with CRLF read as LF, so a Windows checkout and the committed blob agree."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def scoring_code_sha256() -> str:
    digest = hashlib.sha256()
    for rel in _SCORING_CODE:
        digest.update(rel.encode("utf-8") + b"\0" + (_REPO / rel).read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


# Bookkeeping that a benchmark run itself adds; everything else in the record is what the run scored.
_RUN_FIELDS = ("benchmark", "locked_benchmark_history")


# The result fields a seal covers, so a failed run can't be edited into a pass (Ralph R09-4).
_SEALED_RESULT = ("passed", "ran_at", "failures", "report", "report_sha256")


def record_sha256(record: dict, result: Optional[dict] = None) -> str:
    """Digest of the registry entry a benchmark scored (thresholds, calibration, routing, files, text view,
    scoring code) together with that run's result and every locked run on record (Ralph R10-5)."""
    scored = {k: v for k, v in record.items() if k not in _RUN_FIELDS}
    outcome = {k: (result or {}).get(k) for k in _SEALED_RESULT}
    runs = record.get("locked_benchmark_history") or []
    payload = json.dumps({"entry": scored, "result": outcome, "runs": runs}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seal(record: dict, **result) -> dict:
    """The benchmark block for a finished run, bound to the entry it scored and to its own result."""
    return {**result, "record_sha256": record_sha256(record, result)}


def verify(record: Optional[dict], root: Optional[Path] = None) -> Tuple[bool, str]:
    """(validated, reason). Every failure keeps the gate closed."""
    if not record:
        return False, "no registry"
    benchmark = record.get("benchmark") or {}
    if not benchmark.get("passed"):
        return False, "locked benchmark not passed"
    if benchmark.get("record_sha256") != record_sha256(record, benchmark):
        return False, "registry changed after the locked benchmark"
    problem = calibration_problem(record)
    if problem:
        return False, problem
    if benchmark.get("report"):
        report = _REPO / str(benchmark["report"])
        if not report.is_file():
            return False, "benchmark report missing"  # Ralph R11-4: a deleted report can't vouch for a pass
        if text_sha256(report) != benchmark.get("report_sha256"):
            return False, "benchmark report changed after the run"
    ok, reason = verify_files(record, root)
    return (True, "validated") if ok else (False, reason)


def calibration_problem(record: dict) -> Optional[str]:
    """Why a calibrated record could not serve under today's code (text view, precision, scoring code), or None."""
    if not record.get("text_normalization"):
        return "no recorded text view"
    try:
        current_view = text_view_fingerprint(record["text_normalization"])
    except ValueError:
        return "unknown text view"
    if record.get("text_view_sha256") != current_view:
        return "text view changed since calibration"
    from .detector import PRECISIONS

    if record.get("precision") not in PRECISIONS:
        return "no recorded inference precision"
    short = record.get("short_model")
    if short and short.get("precision") not in PRECISIONS:
        return "no recorded inference precision for the short-text model"
    if record.get("scoring_code_sha256") != scoring_code_sha256():
        return "scoring code changed since calibration"
    return None


def verify_calibrated(record: dict, root: Optional[Path] = None) -> Tuple[bool, str]:
    """verify() without the benchmark fields: what a record must meet before a locked run may spend the lock on
    it, since a run that ends in a record verify() rejects wastes the lock (Ralph R12-6)."""
    problem = calibration_problem(record)
    if problem:
        return False, problem
    return verify_files(record, root)


def build_policy(record: dict) -> DecisionPolicy:
    bands = tuple(
        BandRule(b["lower_words"], b["upper_words"], b["human_max"], b["ai_min"])
        for b in record["calibration"]["bands"] if b["supported"]
    )
    min_words = min((b.lower_words for b in bands), default=record["calibration"].get("min_words", 50))
    return DecisionPolicy(min_word_count=min_words, bands=bands, version=record.get("policy_version", POLICY_VERSION))


def build_service(path: Optional[Path] = None, root: Optional[Path] = None,
                  fallback: Optional[AuthorshipService] = None) -> Optional[AuthorshipService]:
    """Service for the registered model, or fallback when no calibrated model is available."""
    return service_from_record(load(path), root, fallback)


def main_only(record: dict) -> dict:
    """The record without its short-text specialist: the main model's own thresholds in every band."""
    out = {k: v for k, v in record.items() if k != "short_model"}
    main_bands = record["calibration"].get("main_bands") or record["calibration"]["bands"]
    out["calibration"] = {**record["calibration"], "bands": main_bands}
    return out


def service_from_record(record: Optional[dict], root: Optional[Path] = None,
                        fallback: Optional[AuthorshipService] = None,
                        validated: Optional[bool] = None) -> Optional[AuthorshipService]:
    """validated overrides the registry's verdict (the locked benchmark tests an unvalidated candidate)."""
    if not record or not weights_file(artifact_dir(record, root)).exists():
        logger.info("registered authorship model unavailable; using fallback")
        return fallback
    from .detector import AuthorshipModelDetector, LengthRoutedDetector, OodGate

    verdict, reason = verify(record, root)
    validated = verdict if validated is None else validated
    view = record.get("text_normalization")

    def model(directory: Path, platt, ood: dict, bank: Path, max_tokens: int, artifact: dict,
              entry: dict) -> AuthorshipModelDetector:
        gate = OodGate(np.load(bank), ood["k"], ood["threshold"]) if bank.exists() else None
        return AuthorshipModelDetector(directory, tuple(platt), gate, max_tokens=max_tokens,
                                       text_view=view,
                                       expected_files_sha256=artifact.get("files_sha256"),
                                       precision=entry.get("precision", LEGACY_PRECISION),
                                       half_weights="precision" in entry)

    detector = model(artifact_dir(record, root), record["calibration"]["platt"], record["calibration"]["ood"],
                     bank_path(record, root), record.get("max_tokens", 256), record["artifact"], record)
    parts = short_parts(record, root)
    short_problem = _check_short(record, root) if parts else None
    if parts and short_problem is None:
        short = record["short_model"]
        detector = LengthRoutedDetector(
            detector, model(parts[0], short["platt"], short["ood"], parts[1], short.get("max_tokens", 160),
                            short["artifact"], short),
            short["ranges"])
    elif parts:
        # The specialist's files are missing or changed. Never pair the main model with the
        # specialist's thresholds (Ralph R07-4): fall back to the main model's own bands.
        logger.warning("short-text model unavailable (%s); using the main model's thresholds", short_problem)
        record = main_only(record)
    logger.info("authorship model %s: %s", record["model_id"], reason)
    return AuthorshipService(detector, policy=build_policy(record), model_validated=validated,
                             detector_version=record["model_id"], require_english=True)
