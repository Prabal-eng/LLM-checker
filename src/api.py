"""FastAPI server and Web UI for file upload, audit execution, and report download."""
import os
import shutil
import uuid
import logging
import threading
from functools import lru_cache
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import PipelineConfig
from .detectors import get_detector
from .pipeline import QAPipeline
from .detection import (
    AuthorshipService,
    CheckTextRequest,
    TextCheckResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Excel AI & Structural QA Engine", version="1.0.0")

_v2_metrics_lock = threading.Lock()
_v2_metrics = {
    "text_checks_total": 0,
    "likely_ai_generated_total": 0,
    "likely_human_written_total": 0,
    "inconclusive_total": 0,
}

# CORS: configure allowed origins via ALLOWED_ORIGINS env var (comma-separated).
# Defaults to localhost only; set to * only for fully local/dev environments.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Maximum upload file size (bytes). Configurable via MAX_UPLOAD_MB env var.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024
logger.info("API configured — CORS origins: %s | max upload: %dMB",
            ALLOWED_ORIGINS, MAX_UPLOAD_BYTES // (1024 * 1024))

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
STATIC_INDEX = os.path.join(STATIC_DIR, "index.html")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/v2/health")
async def v2_health():
    """Health probe for the versioned authorship-check capability."""
    return {
        "status": "ok",
        "service": "llm-authorship-checker",
        "contract_version": "v2",
    }


@app.get("/api/v2/model")
def v2_model():
    """The model the local engine is serving, so the dashboard can name it."""
    service = _get_v2_authorship_service()
    return {"model_id": service.detector_version, "validated": service.model_validated}


@app.get("/api/v2/metrics")
async def v2_metrics():
    """Minimal decision counters for the walking skeleton."""
    with _v2_metrics_lock:
        return dict(_v2_metrics)


@lru_cache(maxsize=1)
def _get_v2_authorship_service() -> AuthorshipService:
    """The registered model (4d on this branch) with verdicts on when its files are present; otherwise the
    unvalidated legacy model. Its closed benchmark gate is set aside by the user's decision (2026-10-08)."""
    from .authorship.registry import load, service_from_record

    legacy = AuthorshipService(
        get_detector(provider="trained", threshold=0.80),
        model_validated=False,
        detector_version="trained-indomain-legacy-unvalidated",
    )
    return service_from_record(load(), fallback=legacy, validated=True)


@app.post("/api/v2/check-text", response_model=TextCheckResponse)
async def v2_check_text(request: CheckTextRequest) -> TextCheckResponse:
    """Check standalone text through the shared selective decision policy."""
    word_count = (
        _get_v2_authorship_service().policy.word_count(request.text)
        if isinstance(request.text, str) else 0
    )
    result = _get_v2_authorship_service().check_text(request.text, context=request.context)
    with _v2_metrics_lock:
        _v2_metrics["text_checks_total"] += 1
        _v2_metrics[f"{result.decision.value}_total"] += 1

    logger.info("v2 text check: words=%d decision=%s", word_count, result.decision.value)
    return result


@app.get("/", response_class=FileResponse)
async def serve_index():
    """Modern interactive dashboard for QA pipeline."""
    if os.path.exists(STATIC_INDEX):
        # Never cached: after a restart the browser kept showing the old engine list.
        return FileResponse(STATIC_INDEX, media_type="text/html", headers={"Cache-Control": "no-store"})
    return HTMLResponse("<h1>Index file not found</h1>", status_code=404)


@app.post("/api/inspect-columns")
async def inspect_columns_endpoint(file: UploadFile = File(...), locale: Optional[str] = Form(None),
                                   sheet: Optional[str] = Form(None)):
    """Extracts column headers from an uploaded file and recommends text rationale columns. Also lists the
    workbook's locales; with locale, inspects that locale's tab or rows (src/qa/locales.py)."""
    import pandas as pd
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".xlsx", ".xls", ".csv"):
        raise HTTPException(status_code=400, detail="Only .xlsx, .xls, and .csv files are supported.")

    # Read file content and enforce size limit
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) // (1024*1024)}MB). Maximum allowed is {MAX_UPLOAD_BYTES // (1024*1024)}MB."
        )

    file_id = str(uuid.uuid4())[:8]
    temp_path = os.path.join(UPLOAD_DIR, f"inspect_{file_id}_{file.filename}")
    with open(temp_path, "wb") as buffer:
        buffer.write(content)

    logger.info("inspect-columns: file=%s size=%dKB", file.filename, len(content) // 1024)
    try:
        from .qa.locales import LocaleError, list_locales, resolve, rows_for

        pipeline = QAPipeline()
        locales = None if locale else list_locales(temp_path, sheet or None)  # a locale pick re-inspects; the page has the list
        sheets = None
        if os.path.splitext(file.filename)[1].lower() != ".csv":
            with pd.ExcelFile(temp_path) as xl:  # the page offers every tab; the choice is the user's
                sheets = [{"name": n, "rows": len(pd.read_excel(xl, sheet_name=n))} for n in xl.sheet_names]
        sheet_name = sheet or None
        if sheet_name and sheets is not None and sheet_name not in [t["name"] for t in sheets]:
            raise HTTPException(status_code=400, detail=f"No tab named '{sheet_name}'.")
        locale_col = locale_name = None
        if locale:
            try:
                sheet_name, locale_col = resolve(temp_path, locale, sheet_name)
            except LocaleError as e:
                raise HTTPException(status_code=400, detail=str(e))
        df, detected_sheet = pipeline.load_dataset(temp_path, sheet_name=sheet_name)
        if locale:
            if locale_col:
                df = rows_for(df, locale_col, locale)
                locale_name = str(df[locale_col].iloc[0]).strip() if len(df) else locale
            else:
                locale_name = detected_sheet.strip()
        cols = list(df.columns)
        
        # Identify recommended columns using intelligent ColumnDiscovery
        from .qa.discovery import ColumnDiscovery
        discovered = ColumnDiscovery.discover(df)
        recommended = [c for c in discovered.text_cols if c in cols]
        if not recommended:
            recommended = [discovered.text_col] if discovered.text_col in cols else []

        logger.info("inspect-columns: detected sheet='%s', %d cols, %d recommended",
                    detected_sheet, len(cols), len(recommended))
        return {
            "columns": cols,
            "recommended": recommended,
            "detected_sheet": detected_sheet,
            "sheets": sheets,
            "locales": locales,
            "locale": locale_name,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("inspect-columns failed for %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/api/audit")
async def run_audit_endpoint(
    file: UploadFile = File(...),
    provider: str = Form("trained"),
    threshold: float = Form(0.70),  # legacy detectors only; the registered model uses its calibrated bars
    only_assigned: bool = Form(True),
    text_cols: Optional[str] = Form(None),
    locale: Optional[str] = Form(None),
    sheet: Optional[str] = Form(None),
    explain_triggers: bool = Form(True),
):
    """Handles file upload, column configuration, and returns QA audit metrics and output link. locale audits that
    locale only; the other locales can be run later, one at a time."""
    import json
    import re

    from .qa.locales import LocaleError
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".xlsx", ".xls", ".csv"):
        raise HTTPException(status_code=400, detail="Only .xlsx, .xls, and .csv files are supported.")

    # Enforce upload size limit before writing to disk
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) // (1024*1024)}MB). Maximum allowed is {MAX_UPLOAD_BYTES // (1024*1024)}MB."
        )

    file_id = str(uuid.uuid4())[:8]
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    locale_suffix = f"_{re.sub(r'[^A-Za-z0-9_-]', '', locale or sheet)}" if (locale or sheet) else ""
    output_filename = f"audited_{file_id}_{os.path.splitext(file.filename)[0]}{locale_suffix}.xlsx"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(input_path, "wb") as buffer:
        buffer.write(content)

    logger.info("audit: file=%s size=%dKB provider=%s threshold=%.0f%%",
                file.filename, len(content) // 1024, provider, threshold * 100)

    config = PipelineConfig()
    config.detector.provider = provider
    config.detector.ai_threshold = threshold
    config.structural.skip_unassigned = only_assigned
    config.triggers.enabled = explain_triggers

    if text_cols:
        try:
            parsed = json.loads(text_cols)
            if isinstance(parsed, list) and len(parsed) > 0:
                config.structural.text_columns = parsed
            elif isinstance(parsed, str):
                config.structural.text_columns = [parsed]
        except Exception:
            config.structural.text_columns = [c.strip() for c in text_cols.split(",") if c.strip()]

    try:
        pipeline = QAPipeline(config=config)
        try:
            res = pipeline.run(input_file=input_path, output_file=output_path, sheet_name=sheet or None, locale=locale or None)
        except LocaleError as e:
            raise HTTPException(status_code=400, detail=str(e))

        logger.info("audit complete: %d rows, %d AI flagged, %d errors",
                    res.total_rows, res.ai_flagged_count, res.structural_errors_count)
        return {
            "status": "success",
            "output_filename": output_filename,
            "total_rows": res.total_rows,
            "ai_flagged_count": res.ai_flagged_count,
            "likely_human_count": res.likely_human_count,
            "inconclusive_count": res.inconclusive_count,
            "structural_errors_count": res.structural_errors_count,
            "structural_warnings_count": res.structural_warnings_count,
            "avg_quality_score": res.avg_quality_score,
            "summary": res.summary
        }
    except HTTPException:
        raise
    except MemoryError:
        logger.error("audit OOM for file %s", file.filename)
        raise HTTPException(status_code=500, detail="File is too large to process in memory. Try a smaller file or contact support.")
    except Exception as e:
        logger.error("audit failed for %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/combine")
async def combine_audits_endpoint(body: dict):
    """Merges audited workbooks from earlier locale audits ({"filenames": [...]}) into one combined workbook."""
    from .report.combine import CombineError, combine_workbooks

    names = body.get("filenames")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise HTTPException(status_code=400, detail="filenames must be a list of audited workbook names.")
    paths = [os.path.join(OUTPUT_DIR, os.path.basename(n)) for n in names]
    missing = [os.path.basename(p) for p in paths if not os.path.exists(p)]
    if missing:
        raise HTTPException(status_code=404, detail=f"Audited workbook not found: {', '.join(missing)}")
    output_filename = f"combined_{str(uuid.uuid4())[:8]}.xlsx"
    try:
        res = combine_workbooks(paths, os.path.join(OUTPUT_DIR, output_filename))
    except CombineError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("combine failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "success", "output_filename": output_filename, "locales": res["locales"], "total_rows": res["rows"]}


@app.get("/api/download/{filename}")
async def download_report(filename: str):
    """Allows downloading generated multi-tab Excel reports."""
    safe_name = os.path.basename(filename)
    path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Requested report not found.")
    return FileResponse(
        path=path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=safe_name
    )
