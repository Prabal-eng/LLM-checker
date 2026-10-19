import os
import json
import logging
import time
import hashlib
from typing import Optional, Dict, Any, Callable, List
import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from .config import PipelineConfig
from .detectors import get_detector, BaseDetector
from .detection import (
    AuthorshipService,
    DetectionDecision,
    TextCheckResponse,
    TextContext,
    combine_column_results,
)
from .qa.structural import run_structural_qa, StructuralQAResult
from .qa.proxies import compute_rater_uniformity, evaluate_row_proxies
from .qa.quality import evaluate_justification_quality
from .qa.discovery import ColumnDiscovery, DiscoveredColumns, norm_name
from .qa.locales import auto_sheet, resolve as resolve_locale, rows_for as rows_for_locale
from .report.summary import compute_audit_summary
from .report.excel_writer import AuditedWorkbookBuilder
from .report.rows import RowReport, agent_table, decided_cells, tier_counts, trigger_word_table
from .report.tier_rates import load_rates
from .report.tiers import EXPLAINED, NOT_CHECKED, TierRules, bar_for
from .report.triggers import JOIN, TriggerFinder, TriggerResult


LOCAL_PROVIDERS = ("trained", "indomain", "in_domain", "ml", "heuristic", "mock", "offline", "local", "model")
# Opt-in (user decision 2026-10-01): the registered model gives verdicts even though its locked benchmark failed,
# and every output says so. The default engine ("trained") stays behind the gate.
PREVIEW_PROVIDER = "preview-unvalidated"   # the registered model (the 4d model on this branch)
PREVIEW_V5 = "preview-v5"                  # the previous model, frozen as the head-to-head incumbent
PREVIEW_PROVIDERS = (PREVIEW_PROVIDER, PREVIEW_V5)
TRIGGER_WORDS_IN_SUMMARY = 50  # the run summary (and the API) carries the top of the table; the workbook all of it


class PipelineRunResult(BaseModel):
    input_file: str
    output_file: str
    total_rows: int
    ai_flagged_count: int
    likely_human_count: int
    inconclusive_count: int
    structural_errors_count: int
    structural_warnings_count: int
    avg_quality_score: float
    summary: Dict[str, Any]


class QAPipeline:
    """End-to-end execution pipeline for Excel QA."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        authorship_service: Optional[AuthorshipService] = None,
    ):
        """Initialises pipeline with config and loads the configured detector."""
        self.config = config or PipelineConfig()
        self.detector: BaseDetector = get_detector(
            provider=self.config.detector.provider,
            threshold=self.config.detector.ai_threshold,
            api_key=self.config.detector.api_key
        )
        # Configured external or legacy detectors stay unvalidated. The local default uses the
        # registered model, whose registry entry opens the gate only after the locked benchmark.
        legacy = AuthorshipService(
            self.detector,
            model_validated=False,
            detector_version=f"{self.config.detector.provider}-legacy-unvalidated",
        )
        self.unvalidated_preview: Optional[Dict[str, Any]] = None
        provider = self.config.detector.provider.lower().strip()
        if authorship_service is None and provider in PREVIEW_PROVIDERS:
            authorship_service = self._preview_service(previous=provider == PREVIEW_V5)
        elif authorship_service is None and provider == "trained":
            # The registered model (4d on this branch) gives verdicts despite its closed gate: the user's decision
            # on 2026-10-08 ("this model works for us for now"). Without its files, the legacy model stands in.
            from .authorship.registry import load, service_from_record

            authorship_service = service_from_record(load(), fallback=legacy, validated=True)
        elif authorship_service is None and provider in LOCAL_PROVIDERS:
            from .authorship.registry import build_service

            authorship_service = build_service(fallback=legacy)
        self.authorship_service = authorship_service or legacy
        # Verdict tiers rank rows in the workbook below the calibrated bar; they never change a decision.
        self.tier_rules = TierRules.from_config(self.config.tiers)
        trig = self.config.triggers
        detector = getattr(self.authorship_service, "detector", None)  # stand-in services (speed timing) have none
        self.trigger_finder: Optional[TriggerFinder] = (
            TriggerFinder(detector, min_share=trig.min_share, max_sentences=trig.max_sentences, top=trig.top,
                          words=trig.words, max_words=trig.max_words, top_words=trig.top_words,
                          min_word_share=trig.min_word_share)
            if trig.enabled and detector is not None else None)
        self.last_reports: List[RowReport] = []
        self._cache: Dict[str, TextCheckResponse] = {}
        self._prefetched: set = set()
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        logger.info("QAPipeline initialised — detector: %s, threshold: %.0f%%",
                    self.config.detector.provider, self.config.detector.ai_threshold * 100)

    def _cache_key(self, clean_text: str, context: TextContext) -> str:
        key_raw = (
            f"{self.authorship_service.detector_version}:"
            f"{self.authorship_service.policy.version}:{TextContext(context).value}:{clean_text}"
        )
        return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

    def prefetch_sheet(self, df: pd.DataFrame, text_cols: List[str], chunk: int = 256,
                       context: TextContext = TextContext.RATER_COMMENT) -> None:
        """Score every cell and each row's joined cells, as the row loop will ask for them, in batched calls,
        so a sheet costs a few detector calls rather than one per cell (build brief Revision 4: the 4 GB GPU
        holds a sheet to 120 s with a large model only when it batches). The loop then reads the cache."""
        cols = [c for c in text_cols if c in df.columns]
        pending: Dict[str, str] = {}

        def want(text: str) -> None:
            key = self._cache_key(text, context)
            if key not in self._cache:
                pending.setdefault(key, text)

        for idx in range(len(df)):
            cells = ["" if pd.isna(df.loc[idx, c]) else str(df.loc[idx, c]).strip() for c in cols]
            for text in cells + [""] * (len(text_cols) - len(cols)):
                want(text)
            filled = [t for t in cells if t]
            if len(filled) > 1:
                want("\n\n".join(filled))
        keys, texts = list(pending), list(pending.values())
        for start in range(0, len(texts), chunk):
            for key, det in zip(keys[start:start + chunk],
                                self.authorship_service.check_batch(texts[start:start + chunk], context)):
                self._cache[key] = det
                self._prefetched.add(key)
        self._cache_misses += len(texts)

    def _cached_detect(self, text: str, context: TextContext = TextContext.RATER_COMMENT) -> TextCheckResponse:
        """Evaluates text using SHA256 caching to avoid redundant API or model calls."""
        clean_text = "" if text is None else str(text).strip()
        cache_key = self._cache_key(clean_text, context)

        if cache_key in self._cache:
            if cache_key in self._prefetched:
                self._prefetched.discard(cache_key)  # its detector call was counted when prefetched
            else:
                self._cache_hits += 1
            return self._cache[cache_key].model_copy()

        det = self.authorship_service.check_text(clean_text, context=context)
        self._cache[cache_key] = det
        self._cache_misses += 1
        return det

    def _preview_service(self, previous: bool = False) -> AuthorshipService:
        """The registered model (or, with previous, the frozen previous model) with verdicts on despite its closed
        gate. Since 2026-10-08 it carries no unvalidated label (user decision); the slices it failed are logged."""
        from .authorship import headtohead, registry

        record = registry.load(headtohead.INCUMBENT_PATH if previous else None)
        service = registry.service_from_record(record, validated=True) if record else None
        if service is None:
            raise RuntimeError("this engine needs a registered model with its files present")
        failures = self._locked_test_failures(record, headtohead)
        if failures:
            logger.info("%s failed its latest locked test (%s); served with verdicts on by the user's decision",
                        record["model_id"], "; ".join(failures))
        return service

    @staticmethod
    def _locked_test_failures(record: Dict[str, Any], headtohead) -> List[str]:
        """The failing slices of the model's latest locked test: its own failed benchmark, else (a record frozen at an
        earlier pass, like the previous model) its side of the latest head-to-head report."""
        benchmark = record.get("benchmark") or {}
        if not benchmark.get("passed"):
            return list(benchmark.get("failures") or [])
        report = headtohead.report_file()
        if not os.path.exists(report):
            return []
        models = json.loads(open(report, encoding="utf-8").read()).get("models", {})
        return next((list(m.get("failures") or []) for m in models.values() if m.get("model_id") == record["model_id"]), [])

    def load_dataset(self, file_path: str, sheet_name: Optional[str] = None) -> tuple[pd.DataFrame, str]:
        """Loads dataset from .xlsx, .xls, or .csv, automatically finding evaluation sheet if needed."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".xlsx", ".xls"):
            # Closed after reading: on Windows an open handle keeps an uploaded copy from being deleted.
            with pd.ExcelFile(file_path) as excel_file:
                target_sheet = sheet_name or self.config.structural.sheet_name
                if not target_sheet:
                    # The only sheet, else the one with the most rows (README-like sheets skipped)
                    target_sheet = auto_sheet(excel_file)
                df = pd.read_excel(excel_file, sheet_name=target_sheet)
            return df, target_sheet
        elif ext == ".csv":
            return pd.read_csv(file_path), "CSV"
        else:
            raise ValueError(f"Unsupported file format: {ext}. Must be .xlsx, .xls, or .csv")

    def _discover_columns(self, df: pd.DataFrame) -> dict:
        """Discovers rater, justification / text columns, verdict, and id columns via ColumnDiscovery."""
        discovered = ColumnDiscovery.discover(df, self.config.structural)
        return discovered.to_dict()

    def _agent_column(self, rater_col: Optional[str], text_cols: List[str]) -> Optional[str]:
        """The discovered rater column when it is one: set in config, or named like a rater column. Discovery
        falls back to the sheet's first column when no column is named like one; that column (an ID, a prompt, a
        rationale) is not an agent column (final review, Important 2)."""
        if not rater_col or rater_col in text_cols:
            return None
        if rater_col == self.config.structural.rater_column:
            return rater_col
        name = norm_name(rater_col)
        return rater_col if any(k in name for k in ColumnDiscovery.RATER_KEYWORDS) else None

    @staticmethod
    def _cell(df: pd.DataFrame, idx: int, column: Optional[str]) -> Optional[str]:
        if not column or column not in df.columns or pd.isna(df.loc[idx, column]):
            return None
        return str(df.loc[idx, column]).strip() or None

    def _find_triggers(self, reports: List[RowReport], progress_callback=None) -> Dict[str, Any]:
        """Trigger sentences for rows in the explained tiers (spec 2026-10-01). A failure on one row is noted
        on that row and never fails the audit."""
        if self.trigger_finder is None:
            return {"enabled": False, "rows": 0, "seconds": 0.0}
        wanted = [r for r in reports if r.tier in EXPLAINED and r.decided]
        started = time.perf_counter()
        for n, report in enumerate(wanted):
            if progress_callback and n % 10 == 0:
                progress_callback(85 + int(5 * n / max(1, len(wanted))), 100,
                                  f"Finding trigger sentences and words {n + 1}/{len(wanted)}...")
            try:
                report.trigger = self.trigger_finder.explain(report.decided)
            except Exception as exc:  # e.g. CUDA out of memory: note it on the row, keep auditing
                logger.warning("trigger sentences failed on row %d: %s", report.master_row, exc.__class__.__name__)
                report.trigger = TriggerResult("error", note="trigger not found (error)")
        return {"enabled": True, "rows": len(wanted), "seconds": round(time.perf_counter() - started, 2)}

    def run(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        sheet_name: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        locale: Optional[str] = None,
    ) -> PipelineRunResult:
        """Executes full audit pipeline on an input spreadsheet. locale audits one locale only: its tab when the
        workbook has one, else the rows of the sheet's locale column that carry it (src/qa/locales.py)."""
        t_start = time.perf_counter()
        logger.info("Pipeline run started — file: %s", input_file)
        if progress_callback:
            progress_callback(0, 100, "Loading spreadsheet...")
        locale_col = locale_name = None
        if locale:
            sheet_name, locale_col = resolve_locale(input_file, locale, sheet_name or self.config.structural.sheet_name)
        df, detected_sheet = self.load_dataset(input_file, sheet_name=sheet_name)
        if locale:
            if locale_col:
                df = rows_for_locale(df, locale_col, locale)
                locale_name = str(df[locale_col].iloc[0]).strip() if len(df) else locale
            else:
                locale_name = detected_sheet.strip()
            logger.info("Locale %s: %d rows from sheet '%s'", locale_name, len(df), detected_sheet)
        logger.info("Loaded sheet '%s': %d rows, %d columns", detected_sheet, len(df), len(df.columns))

        col_map = self._discover_columns(df)
        rater_col = col_map["rater_col"]
        text_cols = col_map["text_cols"]
        primary_text_col = col_map["text_col"]
        verdict_col = col_map["verdict_col"]
        row_id_col = col_map["row_id_col"]

        # Adapt structural config to discovered columns
        struct_cfg = self.config.structural.model_copy()
        struct_cfg.justification_column = primary_text_col
        struct_cfg.text_columns = text_cols
        struct_cfg.verdict_column = verdict_col or "verdict"

        # Optional filter for active / assigned evaluation rows
        if struct_cfg.skip_unassigned and rater_col in df.columns:
            mask = df[rater_col].notna() & (df[rater_col].astype(str).str.strip() != "") & (df[rater_col].astype(str).str.lower() != "nan")
            df = df[mask].reset_index(drop=True)
        
        # Only require columns that actually make sense for this dataset
        req = [rater_col] + [c for c in text_cols if c in df.columns]
        if row_id_col and row_id_col in df.columns:
            req.append(row_id_col)
        if verdict_col and verdict_col in df.columns:
            req.append(verdict_col)
        struct_cfg.required_columns = list(dict.fromkeys(req))

        # Duplicate check subset
        if not struct_cfg.duplicate_subset:
            if rater_col and row_id_col and rater_col in df.columns and row_id_col in df.columns:
                struct_cfg.duplicate_subset = [rater_col, row_id_col]

        # 1. Structural QA
        if progress_callback:
            progress_callback(10, 100, "Running structural QA rules...")
        logger.info("[1/4] Structural QA — %d rows, checking %d columns",
                    len(df), len(struct_cfg.required_columns))
        structural_result = run_structural_qa(df, struct_cfg)
        logger.info("Structural QA complete — %d errors, %d warnings",
                    structural_result.error_count, structural_result.warning_count)

        # 2. Inter-rater Uniformity Proxy (using primary text column)
        if progress_callback:
            progress_callback(25, 100, "Evaluating rater uniformity...")
        uniformity_map = compute_rater_uniformity(df, rater_col=rater_col, text_col=primary_text_col)

        # 3. Row-by-row AI Detection across selected columns, Proxies, and Quality
        logger.info("[3/4] AI detection + proxy signals — %d rows × %d text columns (detector: %s)",
                    len(df), len(text_cols), self.config.detector.provider)
        detection_results = []
        proxy_results = []
        quality_results = []
        reports: List[RowReport] = []
        agent_col = self._agent_column(rater_col, text_cols)
        # A fallback ID column that isn't an agent column is still the task id when discovery skipped it as one.
        task_col = row_id_col or (rater_col if agent_col is None and rater_col not in text_cols and any(
            k in norm_name(rater_col) for k in ColumnDiscovery.ROW_ID_KEYWORDS) else None)

        total_scanned_chars = 0
        total_scanned_words = 0

        category_col = "category" if "category" in df.columns else ("locale" if "locale" in df.columns else "")
        total_rows = len(df)
        if progress_callback:
            progress_callback(30, 100, "Scoring the sheet's text...")
        self.prefetch_sheet(df, text_cols)

        for idx in range(total_rows):
            if progress_callback and total_rows > 0:
                pct = 30 + int((idx / total_rows) * 55)
                progress_callback(pct, 100, f"Scanning row {idx + 1}/{total_rows} for AI & quality...")

            verdict = str(df.loc[idx, verdict_col]) if verdict_col and verdict_col in df.columns and not pd.isna(df.loc[idx, verdict_col]) else ""
            cat_val = str(df.loc[idx, category_col]) if category_col and category_col in df.columns else ""

            # Check all selected text columns
            col_results: Dict[str, TextCheckResponse] = {}

            for t_col in text_cols:
                raw_val = df.loc[idx, t_col] if t_col in df.columns else ""
                val_text = "" if pd.isna(raw_val) else str(raw_val).strip()

                total_scanned_chars += len(val_text)
                total_scanned_words += len(val_text.split())

                det = self._cached_detect(val_text)
                col_results[t_col] = det

            # The row's rationales checked together decide the row; cells stay as detail.
            cells = [(c, "" if pd.isna(df.loc[idx, c]) else str(df.loc[idx, c]).strip())
                     for c in text_cols if c in df.columns]
            row_texts = [t for _, t in cells if t]
            row_det = self._cached_detect(JOIN.join(row_texts)) if len(row_texts) > 1 else None
            row_result = combine_column_results(col_results, row_result=row_det)
            detection_results.append(row_result)

            # Proxies (evaluated on primary text column)
            primary_text = str(df.loc[idx, primary_text_col]) if primary_text_col in df.columns and not pd.isna(df.loc[idx, primary_text_col]) else ""
            context_tokens = [cat_val, verdict] if cat_val else ([verdict] if verdict else [])
            prox = evaluate_row_proxies(
                text=primary_text,
                rater_similarity=uniformity_map.get(idx, 0.0),
                context_tokens=context_tokens
            )
            proxy_results.append(prox)

            tier, reason = self.tier_rules.tier(row_result)
            decided = decided_cells(cells, row_result, combined_checked=row_det is not None)
            reports.append(RowReport(
                position=idx, task_id=self._cell(df, idx, task_col),
                agent=self._cell(df, idx, agent_col) or "(no agent)", result=row_result, tier=tier, reason=reason,
                bar=None if tier == NOT_CHECKED else bar_for(self.authorship_service.policy,
                                                             JOIN.join(t for _, t in decided)),
                cells=cells, decided=decided, combined=row_det, proxy_warnings=list(prox.warnings)))

            # Quality Assessment
            qual = evaluate_justification_quality(primary_text, verdict, self.config.quality)
            quality_results.append(qual)

        trigger_stats = self._find_triggers(reports, progress_callback)
        self.last_reports = reports
        ai_flagged_total = sum(
            1 for d in detection_results
            if d.decision is DetectionDecision.LIKELY_AI_GENERATED
        )
        inconclusive_total = sum(
            1 for d in detection_results
            if d.decision is DetectionDecision.INCONCLUSIVE
        )
        logger.info("[3/4] Detection complete — %d/%d rows flagged (%.1f%%) | cache hits: %d, misses: %d",
                    ai_flagged_total, len(df), ai_flagged_total / max(1, len(df)) * 100,
                    self._cache_hits, self._cache_misses)

        # 4. Compute Summary & Token/Cost Estimation
        if progress_callback:
            progress_callback(90, 100, "Generating audit workbook and analytics...")
        logger.info("[4/4] Building audited workbook...")
        summary = compute_audit_summary(
            df=df,
            detection_results=detection_results,
            quality_results=quality_results,
            structural_issues=structural_result.issues,
            rater_col=rater_col
        )
        summary["detected_sheet"] = detected_sheet
        summary["unvalidated_preview"] = self.unvalidated_preview
        summary["locale"] = locale_name
        summary["rater_column"] = rater_col
        summary["justification_column"] = primary_text_col
        summary["text_columns"] = text_cols
        summary["verdict_column"] = verdict_col
        summary["detector_version"] = self.authorship_service.detector_version
        summary["agent_column"] = agent_col
        summary["tier_counts"] = tier_counts(reports)
        summary["agents"] = agent_table(reports)
        summary["tier_rates"] = load_rates(summary["detector_version"].split(":")[0], rules=self.tier_rules)
        summary["triggers"] = trigger_stats
        words_table = trigger_word_table(reports)
        summary["trigger_words"] = words_table[:TRIGGER_WORDS_IN_SUMMARY]

        # Cost tracking per provider
        rates_per_1k_words = {
            "gptzero": 0.01,
            "originality": 0.10,
            "trained": 0.0,
            "heuristic": 0.0
        }
        provider_key = self.config.detector.provider.lower()
        rate = rates_per_1k_words.get(provider_key, 0.0)
        est_tokens = int(total_scanned_words * 1.3)
        est_cost_usd = round((total_scanned_words / 1000.0) * rate, 4)

        summary["token_usage"] = {
            "total_chars": total_scanned_chars,
            "total_words": total_scanned_words,
            "estimated_tokens": est_tokens,
            "estimated_cost_usd": est_cost_usd,
            "rate_per_1k_words": rate
        }
        summary["cache_stats"] = {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_ratio": round(self._cache_hits / max(1, self._cache_hits + self._cache_misses), 3)
        }

        # 5. Build Audited Workbook
        if not output_file:
            base_dir = os.path.dirname(input_file) or "."
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(base_dir, f"{base_name}_audited.xlsx")

        wb_builder = AuditedWorkbookBuilder(self.tier_rules)
        wb_builder.add_summary_sheet(summary, rates=summary["tier_rates"],
                                     warning=(self.unvalidated_preview or {}).get("note"))
        wb_builder.add_agents_sheet(summary["agents"])
        if trigger_stats["enabled"]:
            wb_builder.add_trigger_words_sheet(words_table)
        wb_builder.add_ai_review_sheet(reports)
        wb_builder.add_annotated_master_sheet(df, reports, quality_results, summary["row_issues_count"], text_cols)
        wb_builder.add_technical_sheet(reports, self.authorship_service.policy)
        if structural_result.issues:
            wb_builder.add_structural_issues_sheet(structural_result.issues)
        wb_builder.save(output_file)

        elapsed = time.perf_counter() - t_start
        logger.info("Pipeline run complete in %.2fs — output: %s", elapsed, output_file)

        return PipelineRunResult(
            input_file=input_file,
            output_file=output_file,
            total_rows=summary["total_rows"],
            ai_flagged_count=summary["ai_flagged_count"],
            likely_human_count=summary["likely_human_count"],
            inconclusive_count=summary["inconclusive_count"],
            structural_errors_count=summary["structural_errors_count"],
            structural_warnings_count=summary["structural_warnings_count"],
            avg_quality_score=summary["avg_quality_score"],
            summary=summary
        )
