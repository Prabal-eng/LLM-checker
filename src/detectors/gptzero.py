"""GPTZero API Detector connector."""
import os
import time
import logging
from typing import Optional
import requests
from .base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds


class GPTZeroDetector(BaseDetector):
    """Detector client for GPTZero API v2."""

    BASE_URL = "https://api.gptzero.me/v2/predict/text"

    def __init__(self, api_key: Optional[str] = None, threshold: float = 0.70):
        super().__init__(threshold=threshold)
        self.api_key = api_key or os.getenv("GPTZERO_API_KEY")

    def detect(self, text: str) -> DetectionResult:
        """Sends text to GPTZero v2 API and returns a DetectionResult."""
        if not self.api_key:
            raise ValueError("GPTZero API Key is missing. Set GPTZERO_API_KEY in environment or pass in config.")

        if not text or not text.strip():
            return DetectionResult(
                provider="gptzero",
                ai_score=0.0,
                is_flagged=False,
                analysis={"error": "Empty text provided"}
            )

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "document": text,
            "version": "2024-01-09"
        }

        last_error: str = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.debug("GPTZero API call (attempt %d/%d)", attempt, MAX_RETRIES)
                response = requests.post(self.BASE_URL, json=payload, headers=headers, timeout=15)

                if response.status_code == 429:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning("GPTZero rate limited (429). Waiting %.1fs before retry %d/%d",
                                   wait, attempt, MAX_RETRIES)
                    time.sleep(wait)
                    last_error = "rate_limited_429"
                    continue

                response.raise_for_status()
                data = response.json()

                documents_list = data.get("documents")
                if not isinstance(documents_list, list) or not documents_list or not isinstance(documents_list[0], dict):
                    return DetectionResult(
                        provider="gptzero",
                        ai_score=0.0,
                        is_flagged=False,
                        analysis={"error": "Malformed response: missing documents"},
                        raw_response=data,
                    )
                documents = documents_list[0]
                ai_score = documents.get("completely_generated_prob")
                if ai_score is None:
                    ai_score = documents.get("average_generated_prob")
                if ai_score is None:
                    return DetectionResult(
                        provider="gptzero",
                        ai_score=0.0,
                        is_flagged=False,
                        analysis={"error": "Malformed response: missing AI score"},
                        raw_response=data,
                    )
                if isinstance(ai_score, bool):
                    return DetectionResult(
                        provider="gptzero",
                        ai_score=0.0,
                        is_flagged=False,
                        analysis={"error": "Malformed response: AI score must be numeric"},
                        raw_response=data,
                    )
                ai_score = round(float(ai_score), 3)

                confidence = documents.get("confidence_category")
                is_flagged = ai_score >= self.threshold

                logger.debug("GPTZero result: score=%.3f flagged=%s", ai_score, is_flagged)
                return DetectionResult(
                    provider="gptzero",
                    ai_score=ai_score,
                    is_flagged=is_flagged,
                    confidence=0.9 if confidence == "high" else 0.7,
                    analysis={
                        "predicted_class": documents.get("predicted_class"),
                        "burstiness": documents.get("burstiness"),
                        "perplexity": documents.get("overall_burstiness"),
                    },
                    raw_response=data
                )
            except requests.exceptions.Timeout:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning("GPTZero timeout on attempt %d/%d. Retrying in %.1fs", attempt, MAX_RETRIES, wait)
                last_error = "timeout"
                time.sleep(wait)
            except Exception as e:
                last_error = str(e)
                logger.error("GPTZero API error on attempt %d/%d: %s", attempt, MAX_RETRIES, e)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_BASE ** attempt)

        logger.error("GPTZero failed after %d attempts. Last error: %s. Returning score=0.0",
                     MAX_RETRIES, last_error)
        return DetectionResult(
            provider="gptzero",
            ai_score=0.0,
            is_flagged=False,
            analysis={"error": f"All {MAX_RETRIES} attempts failed: {last_error}"}
        )
