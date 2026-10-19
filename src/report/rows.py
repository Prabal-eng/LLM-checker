"""One record per audited row (spec 2026-10-01, section 3): the workbook, the summary and the API read these
instead of parallel lists."""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..detection import RowAuthorshipResult, TextCheckResponse
from .tiers import HIGH, LIKELY_AI, MODERATE, NOT_CHECKED, TIER_ORDER
from .triggers import TriggerResult


@dataclass
class RowReport:
    position: int                       # 0-based row of the audited sheet (as written to Annotated Master)
    task_id: Optional[str]
    agent: str
    result: RowAuthorshipResult
    tier: str
    reason: Optional[str]               # why the row was not checked
    bar: Optional[float]                # the score its decided text needed for Likely AI
    cells: List[Tuple[str, str]]        # every text column's (stripped) cell, in text-column order
    decided: List[Tuple[str, str]]      # the cells whose text decided the row
    combined: Optional[TextCheckResponse] = None   # the row's cells checked together, when there were several
    trigger: Optional[TriggerResult] = None
    proxy_warnings: List[str] = field(default_factory=list)

    @property
    def score(self) -> Optional[float]:
        return self.result.score

    @property
    def master_row(self) -> int:
        return self.position + 2

    @property
    def decided_on(self) -> str:
        names = [c for c, _ in self.decided]
        return "all columns together" if len(names) > 1 else (names[0] if names else "")


def decided_cells(cells: List[Tuple[str, str]], row_result, combined_checked: bool) -> List[Tuple[str, str]]:
    """The cells whose text decided the row: every filled cell when the row was decided on its combined text,
    else the one cell whose result became the row's (a cell that flagged the row on its own, or the only one)."""
    filled = [(c, t) for c, t in cells if t]
    if not filled:
        return []
    flagged_alone = any(getattr(e, "code", None) == "CELL_FLAGGED_AI" for e in row_result.evidence)
    if combined_checked and not flagged_alone:
        return filled
    pool = [(c, t) for c, t in filled if c in row_result.likely_ai_columns] or filled
    score = lambda ct: row_result.column_scores.get(ct[0]) if row_result.column_scores.get(ct[0]) is not None else -1.0  # noqa: E731
    return [max(pool, key=score)]


def tier_counts(reports: List[RowReport]) -> Dict[str, int]:
    return {tier: sum(r.tier == tier for r in reports) for tier in TIER_ORDER}


def agent_table(reports: List[RowReport]) -> List[dict]:
    """One row per agent, most flagged first: Likely AI, then High AI score, then highest score."""
    table: Dict[str, dict] = {}
    for r in reports:
        a = table.setdefault(r.agent, {"agent": r.agent, "rows": 0, "likely_ai": 0, "high": 0, "moderate": 0,
                                       "not_checked": 0, "highest_score": None, "share_likely_ai": 0.0})
        a["rows"] += 1
        a["likely_ai"] += r.tier == LIKELY_AI
        a["high"] += r.tier == HIGH
        a["moderate"] += r.tier == MODERATE
        a["not_checked"] += r.tier == NOT_CHECKED
        if r.score is not None and (a["highest_score"] is None or r.score > a["highest_score"]):
            a["highest_score"] = round(float(r.score), 4)
    for a in table.values():
        a["share_likely_ai"] = round(a["likely_ai"] / a["rows"], 4)
    return sorted(table.values(), key=lambda a: (-a["likely_ai"], -a["high"],
                                                 -(a["highest_score"] if a["highest_score"] is not None else -1.0),
                                                 a["agent"]))


def trigger_word_table(reports: List[RowReport]) -> List[dict]:
    """One line per trigger word or phrase across the sheet, most often triggering first. rows_containing counts
    every audited row whose text holds it, triggering or not, so wording most rows share (such as rubric terms)
    stands out: a phrase that triggers often and sits in most rows points at the checker, not at the agents."""
    table: Dict[str, dict] = {}
    for r in reports:
        seen = set()
        for w in (r.trigger.words if r.trigger is not None else []):
            key = " ".join(w.text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            e = table.setdefault(key, {"rows": 0, "agents": set(), "shares": []})
            e["rows"] += 1
            e["agents"].add(r.agent)
            e["shares"].append(min(max(w.share, 0.0), 1.0))
    texts = [" ".join(t for _, t in r.cells if t).lower() for r in reports]
    out = []
    for key, e in table.items():
        pattern = re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, key.split())) + r"(?!\w)")
        containing = sum(1 for t in texts if pattern.search(t))
        out.append({"phrase": key, "trigger_rows": e["rows"], "agents": len(e["agents"]),
                    "agent_names": sorted(e["agents"]), "average_share": round(sum(e["shares"]) / len(e["shares"]), 4),
                    "rows_containing": containing, "share_of_rows": round(containing / len(reports), 4)})
    return sorted(out, key=lambda e: (-e["trigger_rows"], -e["average_share"], e["phrase"]))
