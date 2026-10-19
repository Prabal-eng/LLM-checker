"""Speed and memory limits of the system under test (pre-registration; build brief Revision 3).

    python -m src.authorship.speed --device cuda    # the GPU limits
    python -m src.authorship.speed --device cpu     # the CPU limits (hides the GPU with CUDA_VISIBLE_DEVICES=-1)

Runs the registered service over validation texts: 30 single texts for the p95 time, and a 500 x 6
sheet scored the way the app scores one (QAPipeline.prefetch_sheet): every cell, then each row's cells
joined, as rater comments, in batched calls, starting from a fresh service so model loading counts (Plan C
final review I3; build brief Revision 4). On the CPU a 20-row sample is timed and scaled to 500 rows, and
the CPU sheet figure is reported, not gated. The plain batched figure the v1 bake-off used is recorded
beside it. On the GPU, peak inference
memory is recorded too. Results merge into data/benchmark/<version>/model/speed.json by device. Reads
only validation rows.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from ..corpus.paths import REPO_BENCHMARK_DIR, VERSION
from ..detection import TextContext
from . import registry

logger = logging.getLogger(__name__)

# Build brief Revision 4 (GPU-first): the CPU is a slow fallback, held to 20 s a text; its sheet time is
# reported, not gated.
# Amendment 16: the GPU sheet limit is 900 s (was 120 s), a user decision after the 566 s measurement.
LIMITS = {"cuda": {"peak_vram_gb": 3.5, "p95_ms": 500.0, "sheet_seconds": 900.0},
          "cpu": {"p95_ms": 20_000.0}}
SHEET_ROWS, CELLS_PER_ROW, SINGLES, CPU_SHEET_ROWS = 500, 6, 30, 20


def time_sheet(service, rows: list) -> float:
    """Seconds to score rows as the app does (QAPipeline.prefetch_sheet): every cell, then each row's cells
    joined when it has more than one, as rater comments, in batched calls. Timed on a fresh service, the
    first batch includes loading the models, as the build brief's end-to-end sheet limit does."""
    import pandas as pd

    from ..config import PipelineConfig
    from ..pipeline import QAPipeline

    cols = [f"c{i}" for i in range(max(len(r) for r in rows))]
    df = pd.DataFrame([list(r) + [""] * (len(cols) - len(r)) for r in rows], columns=cols)
    pipeline = QAPipeline(PipelineConfig(), authorship_service=service)
    start = time.perf_counter()
    pipeline.prefetch_sheet(df, cols)
    return time.perf_counter() - start


def measure(service, singles: list, rows: list, sheet_rows: int = SHEET_ROWS) -> dict:
    elapsed = time_sheet(service, rows)  # first, so it includes loading the models
    times = []
    for text in singles:
        start = time.perf_counter()
        service.check_batch([text], TextContext.GENERAL)
        times.append(1000 * (time.perf_counter() - start))
    cells = [c for row in rows for c in row]
    start = time.perf_counter()
    service.check_batch(cells, TextContext.RATER_COMMENT)
    batched = time.perf_counter() - start
    return {"p95_ms": round(float(np.percentile(times, 95)), 1),
            "sheet_seconds": round(elapsed * sheet_rows / len(rows), 1), "sheet_rows_timed": len(rows),
            "sheet_seconds_batched": round(batched * sheet_rows * CELLS_PER_ROW / len(cells), 1)}


def within(result: dict, device: str) -> dict:
    limits = LIMITS[device]
    checks = {k: result.get(k) is not None and result[k] <= v for k, v in limits.items()}
    return {"limits": limits, "checks": checks, "passed": all(checks.values())}


def report_file() -> Path:
    return REPO_BENCHMARK_DIR / VERSION / "model" / "speed.json"


def run(registry_path: Optional[Path] = None, out: Optional[Path] = None, device: Optional[str] = None) -> dict:
    """device forces the pass: "cpu" refuses to run while CUDA is still visible, since the GPU would be timed and
    recorded as the CPU (Ralph R13-8: an empty CUDA_VISIBLE_DEVICES doesn't hide it on Windows)."""
    import torch

    from .calibrate import load_split

    visible = torch.cuda.is_available()
    if device == "cpu" and visible:
        raise SystemExit("the GPU is still visible; run with --device cpu (CUDA_VISIBLE_DEVICES=-1)")
    if device == "cuda" and not visible:
        raise SystemExit("no GPU is visible for the GPU pass")
    device = device or ("cuda" if visible else "cpu")
    record = registry.load(registry_path)
    service = registry.service_from_record(record, validated=True)
    val = load_split("val")
    cells = val.sample(n=SHEET_ROWS * CELLS_PER_ROW, random_state=0)["text"].tolist()
    rows = [cells[i:i + CELLS_PER_ROW] for i in range(0, len(cells), CELLS_PER_ROW)]
    singles = val.sample(n=SINGLES, random_state=1)["text"].tolist()
    if device == "cpu":
        rows = rows[:CPU_SHEET_ROWS]
    else:
        torch.cuda.reset_peak_memory_stats()
    result = measure(service, singles, rows)
    if device == "cuda":
        result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    result.update(within(result, device), measured_at=datetime.now(timezone.utc).isoformat())
    out = out or report_file()
    saved = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    saved.update({"model_id": record["model_id"], device: result, "locked_split_reads": 0})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return saved


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    args = parser.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # before torch first touches CUDA
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(device=args.device), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
