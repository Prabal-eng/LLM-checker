"""Adapter between existing detectors and the versioned authorship policy."""

import logging
from typing import Any, Dict, List, Optional

from ..detectors.base import BaseDetector, DetectionResult
from .contracts import (
    DecisionEvidence,
    DetectionDecision,
    DetectorStatus,
    RowAuthorshipResult,
    TextCheckResponse,
    TextContext,
)
from .obfuscation import deobfuscate
from .policy import DecisionPolicy

logger = logging.getLogger(__name__)


class AuthorshipService:
    """Run one detector and apply a selective three-way decision policy."""

    def __init__(
        self,
        detector: BaseDetector,
        *,
        policy: Optional[DecisionPolicy] = None,
        model_validated: bool = False,
        detector_version: Optional[str] = None,
        require_english: bool = False,
    ) -> None:
        self.detector = detector
        self.policy = policy or DecisionPolicy()
        self.model_validated = model_validated
        self.detector_version = detector_version or detector.__class__.__name__
        self.require_english = require_english

    def check_text(self, text: Any, context: TextContext = TextContext.GENERAL) -> TextCheckResponse:
        """Return a decision while converting detector failures into abstentions.

        Rater comments are flag-only and skip the domain gate: their false-AI rate is measured
        on in-house rater text, which is outside the public-corpus distribution by design.
        """
        return self.check_batch([text], context)[0]

    def check_batch(self, texts: List[Any], context: TextContext = TextContext.GENERAL) -> List[TextCheckResponse]:
        """check_text for many texts with one batched detector call; identical decisions.

        Character-level disguise (invisible characters, lookalike letters) is undone first, so
        every gate and the detector see the text a reader sees.
        """
        context = TextContext(context)
        cleaned = [deobfuscate(t) if isinstance(t, str) else None for t in texts]
        texts = [c.text if c is not None else t for c, t in zip(cleaned, texts)]
        results: List[Optional[TextCheckResponse]] = [
            self._precheck(text) or (self._disguised(text) if clean is not None and clean.abstain else None)
            for text, clean in zip(texts, cleaned)]
        pending = [i for i, r in enumerate(results) if r is None]
        if pending:
            try:
                raws = self.detector.detect_batch([texts[i] for i in pending])
                if len(raws) != len(pending):
                    raise ValueError("detector returned a different number of results")
            except Exception as exc:
                logger.warning("detector raised %s", exc.__class__.__name__)
                raws = [exc] * len(pending)
            for i, raw in zip(pending, raws):
                results[i] = self._from_detector(texts[i], raw, context is TextContext.RATER_COMMENT)
        for result, clean in zip(results, cleaned):
            if clean is not None and clean.changed:
                result.evidence.append(DecisionEvidence(
                    code="OBFUSCATION_NORMALIZED",
                    description="Invisible characters were removed and lookalike letters from other scripts "
                                "were read as Latin before checking.",
                    value={"format_characters_removed": clean.format_removed,
                           "lookalike_letters_mapped": clean.letters_mapped,
                           "mixed_script_words": clean.mixed_script_words,
                           "unusual_letter_words": clean.unusual_letter_words,
                           "odd_symbols": clean.odd_symbols},
                ))
        return results  # type: ignore[return-value]

    def _disguised(self, text: str) -> TextCheckResponse:
        """Disguise that normalization can't read back: abstain rather than score it (Ralph R10-1)."""
        return self.policy.decide(
            text=text,
            detector_version=self.detector_version,
            detector_status=DetectorStatus.AVAILABLE,
            score=None,
            model_validated=self.model_validated,
            unsupported_reason="DISGUISE_SUSPECTED",
        )

    def _precheck(self, text: Any) -> Optional[TextCheckResponse]:
        """Abstentions decided before any detector call: sufficiency, validation, language."""
        if not self.policy.has_sufficient_text(text):
            return self.policy.decide(
                text=text,
                detector_version=self.detector_version,
                detector_status=DetectorStatus.NOT_EVALUATED,
                score=None,
                model_validated=self.model_validated,
            )

        if not self.model_validated:
            return self.policy.decide(
                text=text,
                detector_version=self.detector_version,
                detector_status=DetectorStatus.AVAILABLE,
                score=None,
                model_validated=False,
            )

        if self.require_english:
            from ..authorship.language import check as check_language

            language = check_language(text)
            if not language.is_english:
                return self.policy.decide(
                    text=text,
                    detector_version=self.detector_version,
                    detector_status=DetectorStatus.AVAILABLE,
                    score=None,
                    model_validated=True,
                    unsupported_reason="UNSUPPORTED_LANGUAGE",
                    observed_evidence=[DecisionEvidence(
                        code="LANGUAGE",
                        description="Only English text is supported; this text does not read as English.",
                        value={"latin_share": language.latin_share,
                               "function_word_share": language.function_word_share},
                    )],
                )
        return None

    def _from_detector(self, text: str, raw: Any, flag_only: bool) -> TextCheckResponse:
        """Turn one detector outcome (a result or the exception it raised) into a decision."""
        def unavailable(reason: str) -> TextCheckResponse:
            return self.policy.decide(
                text=text,
                detector_version=self.detector_version,
                detector_status=DetectorStatus.UNAVAILABLE,
                score=None,
                model_validated=True,
                failure_reason=reason,
            )

        if isinstance(raw, Exception):
            return unavailable("DETECTOR_ERROR")
        if not isinstance(raw, DetectionResult):
            return unavailable("DETECTOR_INVALID_RESPONSE")
        if raw.analysis.get("error"):
            return unavailable("DETECTOR_REPORTED_ERROR")
        try:
            observed = [
                DecisionEvidence(code="OBSERVED_INDICATOR", description=str(item))
                for item in raw.analysis.get("evidence", [])
            ]
        except Exception:
            return unavailable("DETECTOR_INVALID_EVIDENCE")

        windows = raw.analysis.get("windows")
        if windows and len(windows) > 1:
            return self._from_windows(text, windows, observed, flag_only)

        unsupported = None
        if raw.analysis.get("in_distribution") is False and not flag_only:
            unsupported = "UNSUPPORTED_DOMAIN"
            observed.append(DecisionEvidence(
                code="OUT_OF_DISTRIBUTION",
                description="The text is unlike the writing the detector was calibrated on.",
                value=raw.analysis.get("ood_distance"),
            ))
        return self.policy.decide(
            text=text,
            detector_version=self.detector_version,
            detector_status=DetectorStatus.AVAILABLE,
            score=raw.ai_score,
            model_validated=True,
            observed_evidence=observed,
            unsupported_reason=unsupported,
            flag_only=flag_only,
        )

    def _from_windows(self, text: str, windows: list, observed: list, flag_only: bool) -> TextCheckResponse:
        """Decide a text too long for one model window from all of its windows (Ralph R07-2).

        The opening window decides, as it was calibrated to: thresholds come from texts read from
        their start. Every later window is checked too, and one that decisively points the other
        way makes the result Inconclusive, so text the opening never saw can veto a label but
        never create one. (A majority vote over windows was tried on validation and rejected:
        later windows of AI text score lower, and it tripled missed AI on long texts.)
        """
        if not flag_only and any(w.get("in_distribution") is False for w in windows):
            observed.append(DecisionEvidence(
                code="OUT_OF_DISTRIBUTION",
                description="Part of the text is unlike the writing the detector was calibrated on.",
                value=max((w.get("ood_distance") or 0.0) for w in windows),
            ))
            return self.policy.decide(text=text, detector_version=self.detector_version,
                                      detector_status=DetectorStatus.AVAILABLE, score=None, model_validated=True,
                                      observed_evidence=observed, unsupported_reason="UNSUPPORTED_DOMAIN")
        # Windows are judged without the flag-only rule so that a likely-human window can still
        # veto; the rule is applied to the final result. The opening is the whole text (the model
        # reads its start), judged in the band of the text's full length, as in calibration.
        parts = [self.policy.decide(text=text if i == 0 else w["text"], detector_version=self.detector_version,
                                    detector_status=DetectorStatus.AVAILABLE, score=w["score"],
                                    model_validated=True) for i, w in enumerate(windows)]
        ai = [p for p in parts if p.decision is DetectionDecision.LIKELY_AI_GENERATED]
        human = [p for p in parts if p.decision is DetectionDecision.LIKELY_HUMAN_WRITTEN]
        observed.append(DecisionEvidence(
            code="WINDOWED_SCORING",
            description=f"The text is longer than one model window, so all {len(parts)} windows were scored; "
                        "the opening decides and any window pointing the other way vetoes.",
            value={"windows": len(parts), "likely_ai": len(ai), "likely_human": len(human),
                   "scores": [round(p.score, 4) if p.score is not None else None for p in parts]},
        ))
        opening = parts[0]
        opposite = {DetectionDecision.LIKELY_AI_GENERATED: human, DetectionDecision.LIKELY_HUMAN_WRITTEN: ai}
        if opposite.get(opening.decision):
            observed.append(DecisionEvidence(
                code="MIXED_AUTHORSHIP_SIGNALS",
                description="Later parts of the text point the other way from its opening.",
                value={"likely_ai": len(ai), "likely_human": len(human)},
            ))
            decision, reason = DetectionDecision.INCONCLUSIVE, "MIXED_AUTHORSHIP_SIGNALS"
        elif flag_only and opening.decision is not DetectionDecision.LIKELY_AI_GENERATED:
            decision, reason = DetectionDecision.INCONCLUSIVE, "BELOW_AI_THRESHOLD_FLAG_ONLY"
        else:
            decision, reason = opening.decision, opening.reason_codes[0]
        return TextCheckResponse(
            decision=decision, score=opening.score, score_valid=opening.score is not None,
            detector_status=DetectorStatus.AVAILABLE, reason_codes=[reason], evidence=observed,
            detector_version=self.detector_version, policy_version=self.policy.version,
        )


def combine_column_results(
    column_results: Dict[str, TextCheckResponse],
    row_result: Optional[TextCheckResponse] = None,
) -> RowAuthorshipResult:
    """Combine a row's cell results into one row decision.

    With row_result (the row's cells checked as one combined text), the row decision is that
    result and the cells are supporting detail: short cells rarely support a decision alone,
    while a row's combined rationales usually do. A cell that reads as Likely AI on its own
    still flags the row (Ralph R07-5): the combined text must not hide a calibrated AI decision.
    Without row_result, cells combine conservatively.
    """
    if not column_results:
        raise ValueError("at least one column result is required")
    if row_result is not None:
        ai_columns = [
            name for name, result in column_results.items()
            if result.decision is DetectionDecision.LIKELY_AI_GENERATED
        ]
        basis, inherited = row_result, []
        if ai_columns and row_result.decision is not DetectionDecision.LIKELY_AI_GENERATED:
            basis = max((column_results[name] for name in ai_columns),
                        key=lambda result: result.score if result.score is not None else -1.0)
            inherited = [DecisionEvidence(
                code="CELL_FLAGGED_AI",
                description=f"{len(ai_columns)} cell(s) read as Likely AI on their own, so the row is flagged "
                            f"although the combined text was {row_result.decision.value}.",
                supports=DetectionDecision.LIKELY_AI_GENERATED,
                value=ai_columns,
            )]
        return RowAuthorshipResult(
            **basis.model_dump(exclude={"evidence"}),
            evidence=[DecisionEvidence(
                code="ROW_COMBINED_TEXT",
                description=f"Decision made on the row's {len(column_results)} text cells checked together.",
                value=len(column_results),
            ), *inherited, *basis.evidence],
            column_decisions={name: result.decision for name, result in column_results.items()},
            column_scores={name: result.score for name, result in column_results.items()},
            column_results={name: result.model_copy(deep=True) for name, result in column_results.items()},
            likely_ai_columns=ai_columns,
        )

    values = list(column_results.values())
    ai_columns = [
        name for name, result in column_results.items()
        if result.decision is DetectionDecision.LIKELY_AI_GENERATED
    ]
    valid = [result for result in values if result.score_valid]

    if ai_columns:
        decision = DetectionDecision.LIKELY_AI_GENERATED
        representative = max(
            (column_results[name] for name in ai_columns),
            key=lambda result: result.score if result.score is not None else -1.0,
        )
    elif all(
        result.decision is DetectionDecision.LIKELY_HUMAN_WRITTEN
        for result in values
    ):
        decision = DetectionDecision.LIKELY_HUMAN_WRITTEN
        representative = max(
            values, key=lambda result: result.score if result.score is not None else -1.0
        )
    else:
        decision = DetectionDecision.INCONCLUSIVE
        representative = max(
            valid or values,
            key=lambda result: result.score if result.score is not None else -1.0,
        )

    if decision is DetectionDecision.INCONCLUSIVE:
        score = representative.score if representative.score_valid else None
        status = (
            DetectorStatus.AVAILABLE
            if any(r.detector_status is DetectorStatus.AVAILABLE for r in values)
            else DetectorStatus.UNAVAILABLE
            if any(r.detector_status is DetectorStatus.UNAVAILABLE for r in values)
            else DetectorStatus.NOT_EVALUATED
        )
    else:
        score = representative.score
        status = DetectorStatus.AVAILABLE

    return RowAuthorshipResult(
        decision=decision,
        score=score,
        score_valid=score is not None,
        detector_status=status,
        reason_codes=list(dict.fromkeys(
            code for result in values for code in result.reason_codes
        )),
        evidence=[item for result in values for item in result.evidence],
        detector_version=representative.detector_version,
        policy_version=representative.policy_version,
        column_decisions={name: result.decision for name, result in column_results.items()},
        column_scores={name: result.score for name, result in column_results.items()},
        column_results={name: result.model_copy(deep=True) for name, result in column_results.items()},
        likely_ai_columns=ai_columns,
    )
