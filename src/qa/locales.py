"""Locales in a workbook, so an audit can run on one locale at a time.

A locale is a tab named like a locale code (en_US, pt-BR, es-419) or, for locales without a tab, a value of a
`locale` column in the sheet the audit would read. A tab wins: in the Personality rater workbooks each locale tab
holds the agents' work (assignee, rationale), while the combined sheet holds only the source rows. Locales finish at
different times, so the counts show how far along each one is."""

import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .discovery import ColumnDiscovery

LOCALE_CODE = re.compile(  # optional variant: hi_in_bhojpuri; or a dialect alone after the language: ja-Kansai
    # region: 2 letters, 3 digits, or 3 capitals (ar-MSA); optional trailing rater/batch number: ar-EG-1
    r"^[a-z]{2,3}[_-](?:(?:[a-z]{2}|\d{3}|(?-i:[A-Z]{3}))(?:[_-][a-z]{3,})?|[a-z]{4,})(?:[_-]\d{1,2})?$",
    re.IGNORECASE)
LOCALE_COLUMNS = ("locale", "dialect")
SKIP_SHEETS = ("readme", "instructions", "guidelines")
PREVIEW_ROWS = 2000


class LocaleError(ValueError):
    """The requested locale isn't in the workbook."""


def locale_key(locale) -> str:
    """Case and separator don't matter: EN-us and en_US are one locale."""
    return str(locale).strip().lower().replace("-", "_")


def is_locale(value) -> bool:
    return isinstance(value, str) and bool(LOCALE_CODE.match(value.strip()))


def auto_sheet(excel_file: pd.ExcelFile) -> str:
    """The sheet an audit reads when none is named: the only one, else the one with the most rows, skipping
    README-like sheets."""
    names = excel_file.sheet_names
    if len(names) == 1:
        return names[0]
    best, most = names[0], -1
    for name in names:
        if name.lower() in SKIP_SHEETS:
            continue
        rows = len(pd.read_excel(excel_file, sheet_name=name, nrows=PREVIEW_ROWS))
        if rows > most:
            best, most = name, rows
    return best


def _locale_column(df: pd.DataFrame) -> Optional[str]:
    return next((c for c in df.columns if str(c).strip().lower() in LOCALE_COLUMNS), None)


def _filled(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return series.notna() & (text != "") & (text.str.lower() != "nan")


def _task_rows(df: pd.DataFrame) -> pd.Series:
    """Rows holding a task: the task-id column is filled (tabs pad with empty rows), else any cell is."""
    row_id = ColumnDiscovery.discover(df).row_id_col
    if row_id and row_id in df.columns:
        return _filled(df[row_id])
    return df.notna().any(axis=1)


def _assigned(df: pd.DataFrame, tasks: pd.Series) -> Optional[int]:
    rater = ColumnDiscovery.discover(df).rater_col
    if not rater or rater not in df.columns:
        return None
    return int((_filled(df[rater]) & tasks).sum())


def _column_locales(df: pd.DataFrame, sheet: str, taken: set) -> List[dict]:
    column = _locale_column(df)
    if column is None:
        return []
    counts = Counter(v.strip() for v in df[column] if is_locale(v))
    return [{"locale": loc, "source": "column", "sheet": sheet, "rows": n, "assigned": None}
            for loc, n in counts.items() if locale_key(loc) not in taken]


def list_locales(path, sheet_name: Optional[str] = None) -> List[dict]:
    """Every locale an audit could run on: {locale, source ("tab" or "column"), sheet, rows, assigned}. Tabs come
    first in workbook order; column locales come from the sheet the audit would read (sheet_name, else auto-detected),
    so a dashboard tab's own Locale column isn't mistaken for task rows."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return _column_locales(pd.read_csv(path), "CSV", set())
    entries, taken = [], set()
    with pd.ExcelFile(path) as excel_file:  # closed, so Windows can delete an uploaded copy afterwards
        for name in excel_file.sheet_names:
            if is_locale(name):
                df = pd.read_excel(excel_file, sheet_name=name)
                tasks = _task_rows(df)
                entries.append({"locale": name.strip(), "source": "tab", "sheet": name, "rows": int(tasks.sum()),
                                "assigned": _assigned(df, tasks)})
                taken.add(locale_key(name))
        base = sheet_name or auto_sheet(excel_file)
        entries += _column_locales(pd.read_excel(excel_file, sheet_name=base), base, taken)
    return entries


def resolve(path, locale: str, sheet_name: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """(sheet, locale column or None) for one locale: its tab if there is one, else the rows of the audited sheet's
    locale column that carry it."""
    want = locale_key(locale)
    path = Path(path)
    if path.suffix.lower() == ".csv":
        base, df = "CSV", pd.read_csv(path)
    else:
        with pd.ExcelFile(path) as excel_file:
            for name in excel_file.sheet_names:
                if is_locale(name) and locale_key(name) == want:
                    return name, None
            base = sheet_name or auto_sheet(excel_file)
            df = pd.read_excel(excel_file, sheet_name=base)
    column = _locale_column(df)
    if column is not None and df[column].map(lambda v: is_locale(v) and locale_key(v) == want).any():
        return base, column
    available = sorted({e["locale"] for e in list_locales(path, sheet_name)})
    raise LocaleError(f"locale {locale!r} not found; the workbook has {', '.join(available) or 'no locales'}")


def rows_for(df: pd.DataFrame, column: str, locale: str) -> pd.DataFrame:
    """The rows of df whose locale column carries locale."""
    want = locale_key(locale)
    return df[df[column].map(lambda v: is_locale(v) and locale_key(v) == want)].reset_index(drop=True)
