"""Where corpus artifacts live. Text stays outside the repo; only locks and records are committed."""

import os
from pathlib import Path

# v2 (Increment 4b) added assistant-style and 2026-model sources; v3 (Increment 4c) pairs Dolly and
# OpenAssistant with local model answers and tests on pre-ChatGPT Stack Exchange; v4 (Increment 4d) adds
# MAGA, DACTYL, Arena and OpenTuringBench, fits Stack Exchange, and tests only on rows no earlier model
# fitted or was tested on. Earlier records stay.
VERSION = "v4"


def previous_versions(version: str) -> tuple:
    """Every corpus version before this one, whose locked tests it must not reuse."""
    return tuple(f"v{i}" for i in range(1, int(version.lstrip("v"))))


PREVIOUS_VERSIONS = previous_versions(VERSION)
DEFAULT_CORPUS_DIR = "D:/llm-authorship-corpus"
REPO_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmark"


def corpus_dir() -> Path:
    return Path(os.getenv("CORPUS_DIR", DEFAULT_CORPUS_DIR))


def lock_path(version: str = VERSION) -> Path:
    return REPO_BENCHMARK_DIR / version / "test.lock.json"


def record_path(version: str = VERSION) -> Path:
    return REPO_BENCHMARK_DIR / version / "build_record.json"
