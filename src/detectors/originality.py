"""Originality.ai API Detector connector."""
import os
import time
import logging
from typing import Optional
import requests
from .base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds


class OriginalityDetector(BaseDetector):
    """Detector client for Originality.ai API."""

    BASE_URL = "https://api.originality.ai/api/v1/scan/ai"

    def __init__(self, api_key: Optional[str] = None, threshold: float = 0.70):
        super().__init__(threshold=threshold)
        self.api_key = api_key or os.getenv("ORIGINALITY_API_KEY")

    def detect(self, text: str) -> DetectionResult:
        """Sends text to Originality.ai API and returns a DetectionResult."""
        if not self.api_key:
            raise ValueError("Originality API Key is missing. Set ORIGINALITY_API_KEY in environment or pass in config.")

        if not text or not text.strip():
            return DetectionResult(
                provider="originality",
                ai_score=0.0,
                is_flagged=False,
                analysis={"error": "Empty text provided"}
            )

        headers = {
            "X-OAI-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "content": text,
            "title": "Row Justification QA Scan"
        }

        last_error: str = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.debug("Originality.ai API call (attempt %d/%d)", attempt, MAX_RETRIES)
                response = requests.post(self.BASE_URL, json=payload, headers=headers, timeout=15)

                if response.status_code == 429:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning("Originality.ai rate limited (429). Waiting %.1fs before retry %d/%d",
                                   wait, attempt, MAX_RETRIES)
                    time.sleep(wait)
                    last_error = "rate_limited_429"
                    continue

                response.raise_for_status()
                data = response.json()

                score_data = data.get("score")
                if not isinstance(score_data, dict) or score_data.get("ai") is None:
                    return DetectionResult(
                        provider="originality",
                        ai_score=0.0,
                        is_flagged=False,
                        analysis={"error": "Malformed response: missing AI score"},
                        raw_response=data,
                    )
                if isinstance(score_data["ai"], bool):
                    return DetectionResult(
                        provider="originality",
                        ai_score=0.0,
                        is_flagged=False,
                        analysis={"error": "Malformed response: AI score must be numeric"},
                        raw_response=data,
                    )
                ai_score = float(score_data["ai"])
                ai_score = round(ai_score, 3)
                is_flagged = ai_score >= self.threshold

                logger.debug("Originality.ai result: score=%.3f flagged=%s", ai_score, is_flagged)
                return DetectionResult(
                    provider="originality",
                    ai_score=ai_score,
                    is_flagged=is_flagged,
                    confidence=float(score_data.get("confidence", 0.85)),
                    analysis={
                        "originality_score": score_data.get("original"),
                        "ai_score": score_data.get("ai"),
                    },
                    raw_response=data
                )
            except requests.exceptions.Timeout:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning("Originality.ai timeout on attempt %d/%d. Retrying in %.1fs",
                               attempt, MAX_RETRIES, wait)
                last_error = "timeout"
                time.sleep(wait)
            except Exception as e:
                last_error = str(e)
                logger.error("Originality.ai error on attempt %d/%d: %s", attempt, MAX_RETRIES, e)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_BASE ** attempt)

        logger.error("Originality.ai failed after %d attempts. Last error: %s. Returning score=0.0",
                     MAX_RETRIES, last_error)
        return DetectionResult(
            provider="originality",
            ai_score=0.0,
            is_flagged=False,
            analysis={"error": f"All {MAX_RETRIES} attempts failed: {last_error}"}
        )
