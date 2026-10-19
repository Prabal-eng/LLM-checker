"""Airtable Sync Integration for multi-reviewer collaboration."""
import os
import time
import logging
from typing import List, Dict, Any, Optional
import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds


class AirtableConfig(BaseModel):
    api_token: str = Field(default_factory=lambda: os.getenv("AIRTABLE_API_TOKEN", ""))
    base_id: str = Field(default_factory=lambda: os.getenv("AIRTABLE_BASE_ID", ""))
    table_name: str = Field(default_factory=lambda: os.getenv("AIRTABLE_TABLE_NAME", "AI Review Queue"))


class AirtableSyncClient:
    """Syncs flagged evaluation records to Airtable for team review."""

    def __init__(self, config: Optional[AirtableConfig] = None):
        self.config = config or AirtableConfig()

    def sync_flagged_rows(self, flagged_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batches and posts records to Airtable with retry on transient errors."""
        if not self.config.api_token or not self.config.base_id:
            logger.warning("Airtable sync skipped: credentials not configured.")
            return {
                "status": "skipped",
                "message": "Airtable credentials (AIRTABLE_API_TOKEN, AIRTABLE_BASE_ID) not configured."
            }

        url = f"https://api.airtable.com/v0/{self.config.base_id}/{self.config.table_name}"
        headers = {
            "Authorization": f"Bearer {self.config.api_token}",
            "Content-Type": "application/json"
        }

        created_count = 0
        batch_size = 10
        errors = []

        logger.info("Airtable sync: %d records in batches of %d", len(flagged_records), batch_size)
        for i in range(0, len(flagged_records), batch_size):
            batch = flagged_records[i:i + batch_size]
            payload = {"records": [{"fields": rec} for rec in batch]}

            last_error = ""
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    res = requests.post(url, json=payload, headers=headers, timeout=15)
                    if res.status_code == 429:
                        wait = RETRY_BACKOFF_BASE ** attempt
                        logger.warning("Airtable rate limited (429). Waiting %.1fs", wait)
                        time.sleep(wait)
                        last_error = "rate_limited_429"
                        continue
                    res.raise_for_status()
                    created_count += len(batch)
                    logger.debug("Airtable batch %d/%d synced (%d records)",
                                 i // batch_size + 1, -(-len(flagged_records) // batch_size), len(batch))
                    break
                except Exception as e:
                    last_error = str(e)
                    logger.error("Airtable batch error (attempt %d/%d): %s", attempt, MAX_RETRIES, e)
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BACKOFF_BASE ** attempt)
            else:
                errors.append(f"Batch {i // batch_size + 1}: {last_error}")

        logger.info("Airtable sync complete: %d synced, %d errors", created_count, len(errors))
        return {
            "status": "completed" if not errors else "partial_error",
            "synced_count": created_count,
            "errors": errors
        }
