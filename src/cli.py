"""Command Line Interface for the Excel AI QA Pipeline."""
import argparse
import sys
import json
import logging
import os
from .config import PipelineConfig, DetectorConfig
from .pipeline import QAPipeline


def _setup_logging() -> None:
    """Configures the root logger for the pipeline. Level set by LOG_LEVEL env var."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers unless debugging
    if level > logging.DEBUG:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    """Entry point for the Excel AI QA Pipeline CLI."""
    _setup_logging()
    parser = argparse.ArgumentParser(
        description="Excel AI Text Detection and Structural QA Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Audit subcommand
    audit_parser = subparsers.add_parser("audit", help="Run end-to-end audit on an Excel or CSV file")
    audit_parser.add_argument("input_file", help="Path to input .xlsx or .csv file")
    audit_parser.add_argument("-o", "--output", help="Path to save audited .xlsx report", default=None)
    audit_parser.add_argument("-t", "--threshold", type=float, default=0.70, help="AI detection flag threshold (0.0 to 1.0)")
    audit_parser.add_argument("-p", "--provider", choices=["trained", "preview-unvalidated", "preview-v5", "heuristic", "gptzero", "originality"], default="trained",
                              help="AI detection engine (trained: the registered 4d model; preview-v5: the previous model)")
    audit_parser.add_argument("-s", "--sheet", help="Target sheet name in Excel file (auto-detected if omitted)", default=None)
    audit_parser.add_argument("--locale", help="Audit one locale only (e.g. hi_IN): its tab if the workbook has one, "
                                               "else the rows of the sheet's locale column that carry it", default=None)
    audit_parser.add_argument("--rater-col", help="Column name containing rater IDs / emails", default=None)
    audit_parser.add_argument("--text-col", help="Column name containing justification / rationale (alias for --text-cols)", default=None)
    audit_parser.add_argument("--text-cols", nargs="*", help="One or more column names containing justification / rationale to scan for AI", default=None)
    audit_parser.add_argument("--only-assigned", action="store_true", help="Audit only rows that have an assigned rater")
    audit_parser.add_argument("--no-triggers", action="store_true",
                              help="Skip finding trigger sentences (the sentences behind a high AI score)")
    audit_parser.add_argument("--api-key", help="API key for commercial detector (or set via environment variable)", default=None)
    audit_parser.add_argument("--json", action="store_true", help="Output summary in JSON format")

    # Combine subcommand
    combine_parser = subparsers.add_parser(
        "combine", help="Merge audited workbooks (one per locale) into one combined workbook")
    combine_parser.add_argument("audited_files", nargs="+", help="Two or more audited .xlsx workbooks")
    combine_parser.add_argument("-o", "--output", default="combined_audit.xlsx", help="Path to save the combined .xlsx")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "combine":
        from .report.combine import CombineError, combine_workbooks
        try:
            result = combine_workbooks(args.audited_files, args.output)
        except (CombineError, OSError) as e:
            print(f"\n[ERROR] Combine failed: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"[OK] Combined {len(result['locales'])} locales ({result['rows']} rows): {result['output']}")
        return

    if args.command == "audit":
        config = PipelineConfig()
        config.detector.provider = args.provider
        config.detector.ai_threshold = args.threshold
        if args.sheet:
            config.structural.sheet_name = args.sheet
        if args.rater_col:
            config.structural.rater_column = args.rater_col

        # Handle text column(s) selection
        selected_cols = []
        if args.text_cols:
            for item in args.text_cols:
                # Support comma-separated items within args e.g. --text-cols "col1,col2"
                for sub in item.split(","):
                    if sub.strip():
                        selected_cols.append(sub.strip())
        elif args.text_col:
            for sub in args.text_col.split(","):
                if sub.strip():
                    selected_cols.append(sub.strip())

        if selected_cols:
            config.structural.text_columns = selected_cols
            config.structural.justification_column = selected_cols[0]

        if args.only_assigned:
            config.structural.skip_unassigned = True
        if args.no_triggers:
            config.triggers.enabled = False
        if args.api_key:
            config.detector.api_key = args.api_key

        pipeline = QAPipeline(config=config)

        if not args.json:
            print(f"\n[INFO] Running QA Pipeline on: {args.input_file}")
            print(f"   Provider: {args.provider} | AI Threshold: {args.threshold * 100:.0f}%")
            if selected_cols:
                print(f"   Target Columns for AI Detection: {', '.join(selected_cols)}")
            if args.locale:
                print(f"   Locale: {args.locale}")

        pbar = None
        if not args.json:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=100, desc="Starting audit", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}% [{elapsed}<{remaining}] {desc}")
            except ImportError:
                pbar = None

        def _cli_progress(current: int, total: int, stage: str):
            if pbar:
                pbar.set_description(stage[:40])
                pbar.n = min(100, max(0, current))
                pbar.refresh()

        try:
            result = pipeline.run(
                args.input_file,
                args.output,
                sheet_name=args.sheet,
                progress_callback=_cli_progress if pbar else None,
                locale=args.locale,
            )
            if pbar:
                pbar.n = 100
                pbar.set_description("Audit complete")
                pbar.close()

            if args.json:
                print(json.dumps(result.model_dump(), indent=2))
            else:
                summary = result.summary
                print("\n" + "=" * 55)
                print("           AUDIT SUMMARY RESULTS")
                print("=" * 55)
                print(f" Detected Sheet             : {summary.get('detected_sheet')}")
                if summary.get("locale"):
                    print(f" Locale                     : {summary['locale']}")
                print(f" Rater Column               : {summary.get('rater_column')}")
                print(f" AI Target Columns          : {', '.join(summary.get('text_columns', [summary.get('justification_column')]))}")
                print(f" Total Rows Evaluated       : {result.total_rows}")
                print(f" AI Flagged Rows            : {result.ai_flagged_count} ({(result.ai_flagged_count / max(1, result.total_rows)) * 100:.1f}%)")
                print(f" Likely Human Rows          : {result.likely_human_count}")
                print(f" Inconclusive Rows          : {result.inconclusive_count}")
                print(f" Structural Errors          : {result.structural_errors_count}")
                print(f" Structural Warnings        : {result.structural_warnings_count}")
                print(f" Avg Justification Quality  : {result.avg_quality_score * 100:.1f}%")
                if "token_usage" in summary:
                    t_info = summary["token_usage"]
                    print(f" Est. Tokens / Cost         : {t_info['estimated_tokens']} tokens (~${t_info['estimated_cost_usd']:.4f})")
                if "cache_stats" in summary:
                    c_info = summary["cache_stats"]
                    print(f" Cache Hit Ratio            : {c_info['hit_ratio'] * 100:.1f}% ({c_info['cache_hits']} hits, {c_info['cache_misses']} misses)")
                print("-" * 55)
                print(f"[OK] Generated Audited Excel Report: {result.output_file}\n")

        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
