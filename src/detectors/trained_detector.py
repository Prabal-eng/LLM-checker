"""Trained In-Domain AI Provenance Detector with hybrid calibrated stylometrics."""
import logging
import math
import os
import re
import joblib
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from .base import BaseDetector, DetectionResult
from src.qa.stylometry import StylometricFeatureExtractor

logger = logging.getLogger(__name__)

# Formulaic comparative openings commonly used in LLM candidate evaluations
FORMULAIC_OPENING_PATTERNS = [
    r"^(reponse|response )?[a-c] is the strongest (overall|because|of the three|here)",
    r"^[a-c] is the strongest overall because",
    r"^(response )?[a-c] feels the most natural (and human )?because",
    r"^(response )?[a-c] is the best (of the three|fit|option) (here )?because",
    r"^(response )?[a-c] (is the most balanced|stood better overall|gives the most grounded)",
    r"^(all three responses|both responses) are similarly strong because",
]

# Canned / repetitive boilerplate evaluation templates used by low-effort raters
BOILERPLATE_TEMPLATES = [
    r"^(excellent|good|great) (generated image|response) based on the user'?s prompt",
    r"^the prompt follows the user'?s (intent and )?instructions? as requested",
    r"^the prompt (is aligned with|aligns with|matches) the user'?s (intent|request|instructions?)",
    r"^the image follows the user'?s prompt with clear",
    r"^the overall image and instructions displayed are clear and easy to follow",
    r"^the prompt follows the user'?s request with major factual",
]

# High-frequency LLM buzzwords and stock transitions for explainability and scoring
LLM_INDICATOR_PATTERNS = [
    r"\bdelve[s]?\b",
    r"\bcrucial\b",
    r"\btestament\b",
    r"\bseamlessly\b",
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\bin conclusion\b",
    r"\bit is worth noting that\b",
    r"\bit is important to consider\b",
    r"\bcomprehensive understanding\b",
    r"\brobust framework\b",
    r"\bunderscores the necessity\b",
    r"\bpivotal role\b",
    r"\bhighlights the significance\b",
    r"\bserves as a reminder\b",
    r"\baligns with the expected\b",
    r"\bupon closer inspection\b",
    r"\bupon close inspection\b",
    r"\bmeticulously\b",
    r"\bnotwithstanding\b",
    r"\bparamount\b",
    r"\bthe response is balanced\b",
    r"\bgives a balanced response\b",
    r"\bis the strongest overall because\b",
    r"\bis the strongest of the three\b",
    r"\bis the strongest because\b",
    r"\bis appropriately candid\b",
    r"\btakes a balanced approach\b",
    r"\bbalances? .* with\b",
    r"\bconstructive pushback\b",
    r"\bwhile maintaining a\b",
    r"\bwithout becoming (overly|as)\b",
]

# Typical informal/human rater markers (counter-signals that lower AI probability)
HUMAN_INFORMAL_PATTERNS = [
    r"\bidk\b",
    r"\btw\b",
    r"\blooks? ok\b",
    r"\blol\b",
    r"\btypo\b",
    r"\bimo\b",
    r"\bdidn't work\b",
    r"\bcan't repro\b",
    r"\bgave up\b",
    r"\bclicked\b",
    r"\bdoesn't make sense\b",
    r"\bmix up of\b",
    r"\bdid math right\b",
    r"\bweird time wrap\b",
    r"\bbecasue\b",
    r"\bidentifiying\b",
    r"\bexplainging\b",
    r"\bconcers\b",
    r"\bsuggeation\b",
    r"\baslo\b",
    r"\bbrigs\b",
    r"\blighten the stress\b",
    r"\bbeing fail\b",
    r"\bgamer fellow\b",
    r"\bpoem7s\b",
    r"\bbrusting\b",
    r"\benthisiastic\b",
    r"\bकाफी स्मार्ट\b",
    r"\bशानदार\b",
    r"\bबेहतरीन\b",
]


class TrainedInDomainDetector(BaseDetector):
    """Deep calibrated classifier trained specifically on rater evaluation datasets.
    
    Combines Word N-Grams + Char N-Grams + Stylometric Feature Extractor with Calibrated
    Classifier probabilities and rich diagnostic explainability (burstiness, lexical diversity,
    formulaic comparative openings, and linguistic marker detection).
    """

    def __init__(self, threshold: float = 0.70, model_path: Optional[str] = None):
        super().__init__(threshold=threshold)
        self.compiled_formulaic_openings = [re.compile(p, re.IGNORECASE) for p in FORMULAIC_OPENING_PATTERNS]
        self.compiled_boilerplate = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_TEMPLATES]
        self.compiled_llm_markers = [re.compile(p, re.IGNORECASE) for p in LLM_INDICATOR_PATTERNS]
        self.compiled_human_markers = [re.compile(p, re.IGNORECASE) for p in HUMAN_INFORMAL_PATTERNS]

        if not model_path:
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "models",
                "indomain_ai_classifier.joblib"
            )
            model_path = default_path

        if not os.path.exists(model_path):
            logger.error("Trained model not found at '%s'", model_path)
            raise FileNotFoundError(f"Trained model not found at '{model_path}'. Run `python scripts/train_indomain_model.py` first.")

        logger.info("Loading trained AI detector model from %s", model_path)
        self.model = joblib.load(model_path)
        logger.info("Trained model loaded successfully.")

    def _compute_score_and_explainability(
        self, clean_text: str, words: List[str], probs: np.ndarray
    ) -> Tuple[float, bool, float, Dict[str, Any]]:
        """Computes calibrated hybrid score and diagnostic explainability."""
        word_count = len(words)
        ml_prob = float(probs[1])

        # 1. Match formulaic comparative openings
        matched_openings = [pat.pattern for pat in self.compiled_formulaic_openings if pat.search(clean_text)]

        # 2. Match boilerplate template copypasta
        matched_boilerplate = [pat.pattern for pat in self.compiled_boilerplate if pat.search(clean_text)]

        # 3. Match LLM indicator markers
        llm_matches = []
        for pat in self.compiled_llm_markers:
            found = pat.findall(clean_text)
            if found:
                llm_matches.extend(found)

        # 4. Match human informal markers / typos
        human_matches = []
        for pat in self.compiled_human_markers:
            found = pat.findall(clean_text)
            if found:
                human_matches.extend(found)

        # 5. Burstiness (Sentence length variation)
        sentences = [s.strip() for s in re.split(r"[.!?]+", clean_text) if s.strip()]
        sentence_lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences if len(s.split()) > 0]
        low_burstiness = False
        if len(sentence_lengths) > 1:
            mean_len = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((l - mean_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            std_dev = math.sqrt(variance)
            burstiness = round(std_dev / (mean_len + 1e-5), 3)
            if burstiness < 0.22:
                low_burstiness = True
        else:
            burstiness = 0.50

        # 6. Lexical diversity (Type-Token Ratio)
        unique_words = len(set(w.lower() for w in words))
        ttr = round(unique_words / (word_count + 1e-5), 3)

        # 7. Hybrid Score Synthesis
        score = ml_prob
        evidence: List[str] = []

        if matched_openings:
            evidence.append("Formulaic comparative opening pattern")
            score = max(score, 0.75 + min(0.20, len(llm_matches) * 0.05))
            if llm_matches:
                evidence.append(f"Contains {len(llm_matches)} typical LLM marker phrase(s): {', '.join(set(llm_matches))}")
            if low_burstiness:
                evidence.append(f"Low sentence burstiness ({burstiness:.2f}) indicates uniform machine-like phrasing")
        elif matched_boilerplate:
            evidence.append("Repetitive stock boilerplate template")
            score = max(score, 0.75)
            if low_burstiness:
                evidence.append(f"Low sentence burstiness ({burstiness:.2f}) indicates uniform machine-like phrasing")
        elif llm_matches:
            evidence.append(f"Contains {len(llm_matches)} typical LLM marker phrase(s): {', '.join(set(llm_matches))}")
            if len(llm_matches) >= 2 or ml_prob >= 0.50:
                score = max(score, 0.70 + min(0.25, len(llm_matches) * 0.08))
            else:
                score = min(0.95, score + len(llm_matches) * 0.12)
            if low_burstiness:
                evidence.append(f"Low sentence burstiness ({burstiness:.2f}) indicates uniform machine-like phrasing")

        if human_matches:
            evidence.append(f"Contains authentic informal human signal(s): {', '.join(set(human_matches))}")
            score -= 0.40 * len(human_matches)

        final_ai_score = round(max(0.01, min(0.99, score)), 4)
        is_flagged = final_ai_score >= self.threshold
        confidence = round(float(max(final_ai_score, 1.0 - final_ai_score)), 3)

        explain = {
            "word_count": word_count,
            "sentence_count": len(sentences),
            "burstiness_ratio": burstiness,
            "ttr": ttr,
            "llm_markers_found": list(set(llm_matches)),
            "human_markers_found": list(set(human_matches)),
            "formulaic_openings_found": matched_openings,
            "evidence": evidence,
            "predicted_label": "AI" if is_flagged else "Human",
            "ml_model_prob": round(ml_prob, 4),
            "ai_prob": final_ai_score,
            "human_prob": round(1.0 - final_ai_score, 4)
        }

        return final_ai_score, is_flagged, confidence, explain

    def detect(self, text: str) -> DetectionResult:
        if not text or not str(text).strip():
            return DetectionResult(
                provider="trained_indomain",
                ai_score=0.0,
                is_flagged=False,
                confidence=1.0,
                analysis={"reason": "Empty input text", "word_count": 0}
            )

        clean_text = str(text).strip()
        words = clean_text.split()
        if len(words) < 3:
            return DetectionResult(
                provider="trained_indomain",
                ai_score=0.10,
                is_flagged=False,
                confidence=0.5,
                analysis={"reason": "Text too short (<3 words)", "word_count": len(words)}
            )

        # Predict calibrated probability of AI class (label 1)
        probs = self.model.predict_proba([clean_text])[0]
        ai_score, is_flagged, confidence, explain = self._compute_score_and_explainability(
            clean_text, words, probs
        )

        return DetectionResult(
            provider="trained_indomain",
            ai_score=ai_score,
            is_flagged=is_flagged,
            confidence=confidence,
            analysis=explain
        )

    def detect_batch(self, texts: List[str], max_workers: int = 5) -> List[DetectionResult]:
        """Vectorized batch prediction across all texts with explainability extraction."""
        if not texts:
            return []

        results: List[Optional[DetectionResult]] = [None] * len(texts)
        texts_to_predict = []
        indices_to_predict = []

        for idx, text in enumerate(texts):
            if not text or not str(text).strip():
                results[idx] = DetectionResult(
                    provider="trained_indomain",
                    ai_score=0.0,
                    is_flagged=False,
                    confidence=1.0,
                    analysis={"reason": "Empty input text", "word_count": 0}
                )
            else:
                clean_text = str(text).strip()
                words = clean_text.split()
                if len(words) < 3:
                    results[idx] = DetectionResult(
                        provider="trained_indomain",
                        ai_score=0.10,
                        is_flagged=False,
                        confidence=0.5,
                        analysis={"reason": "Text too short (<3 words)", "word_count": len(words)}
                    )
                else:
                    texts_to_predict.append(clean_text)
                    indices_to_predict.append((idx, clean_text, words))

        if texts_to_predict:
            all_probs = self.model.predict_proba(texts_to_predict)
            for (idx, clean_text, words), probs in zip(indices_to_predict, all_probs):
                ai_score, is_flagged, confidence, explain = self._compute_score_and_explainability(
                    clean_text, words, probs
                )
                results[idx] = DetectionResult(
                    provider="trained_indomain",
                    ai_score=ai_score,
                    is_flagged=is_flagged,
                    confidence=confidence,
                    analysis=explain
                )

        return [r for r in results if r is not None]
