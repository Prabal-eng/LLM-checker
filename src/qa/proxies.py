"""Proxy signals module to corroborate or counterbalance raw AI detection scores."""
import re
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProxySignals(BaseModel):
    is_uniform_phrasing: bool = False
    rater_similarity_score: float = 0.0
    is_suspiciously_polished: bool = False
    lacks_row_anchors: bool = False
    proxy_flags_count: int = 0
    warnings: List[str] = Field(default_factory=list)


def _tokenize_ngrams(text: str, n: int = 3) -> set:
    words = re.findall(r"\b\w+\b", str(text).lower())
    if len(words) < n:
        return set(words)
    return set(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def compute_rater_uniformity(
    df: pd.DataFrame,
    rater_col: str = "rater_id",
    text_col: str = "justification",
    max_peer_comparisons: int = 30
) -> Dict[int, float]:
    """Calculates average n-gram overlap of each row's justification against other justifications by the same rater.
    
    Optimized to O(N * min(N, max_peer_comparisons)) to prevent O(N^2) bottlenecks on large datasets.
    """
    if rater_col not in df.columns or text_col not in df.columns:
        logger.debug("compute_rater_uniformity: missing rater_col '%s' or text_col '%s'", rater_col, text_col)
        return {idx: 0.0 for idx in df.index}

    uniformity_scores: Dict[int, float] = {}
    grouped = df.groupby(rater_col)

    for rater_val, group in grouped:
        indices = group.index.tolist()
        if len(indices) <= 1:
            for idx in indices:
                uniformity_scores[idx] = 0.0
            continue

        ngrams_map = {idx: _tokenize_ngrams(str(df.loc[idx, text_col])) for idx in indices}
        use_sampling = len(indices) > max_peer_comparisons

        for idx in indices:
            cur_ngrams = ngrams_map[idx]
            if not cur_ngrams:
                uniformity_scores[idx] = 0.0
                continue

            # If large group, compare against evenly spaced sample of peers to cap O(n^2)
            if use_sampling:
                step = max(1, (len(indices) - 1) // max_peer_comparisons)
                peer_indices = [oidx for i, oidx in enumerate(indices) if oidx != idx and i % step == 0][:max_peer_comparisons]
            else:
                peer_indices = [oidx for oidx in indices if oidx != idx]

            other_scores = [
                _jaccard_similarity(cur_ngrams, ngrams_map[oidx])
                for oidx in peer_indices
            ]
            avg_sim = sum(other_scores) / len(other_scores) if other_scores else 0.0
            uniformity_scores[idx] = round(avg_sim, 3)

    return uniformity_scores


def evaluate_row_proxies(
    text: str,
    rater_similarity: float,
    context_tokens: Optional[List[str]] = None
) -> ProxySignals:
    """Evaluates all proxy signals for a single row's justification."""
    warnings = []
    text_str = str(text or "").strip()
    words = re.findall(r"\b\w+\b", text_str)
    word_count = len(words)

    # 1. Uniform phrasing across rater
    is_uniform = rater_similarity >= 0.45
    if is_uniform:
        warnings.append(f"High inter-row phrasing repetition for rater (overlap: {rater_similarity * 100:.1f}%)")

    # 2. Suspiciously polished syntax
    # Complex punctuation (semicolons, em-dashes), formal adverbial openers, high average word length
    has_semicolon = ";" in text_str or "—" in text_str or "--" in text_str
    formal_intro = bool(re.match(r"^(Upon|Furthermore|Consequently|In regard to|It should be noted),", text_str, re.I))
    avg_word_len = sum(len(w) for w in words) / (word_count + 1e-5)
    
    is_polished = (word_count >= 15 and avg_word_len >= 5.5 and (has_semicolon or formal_intro))
    if is_polished:
        warnings.append("Suspiciously formal/polished prose structure relative to peer ratings")

    # 3. Contextual detachment (lacks row-specific references)
    lacks_anchors = False
    if context_tokens and word_count >= 10:
        found_tokens = [tok.lower() for tok in context_tokens if tok and tok.lower() in text_str.lower()]
        if not found_tokens:
            lacks_anchors = True
            warnings.append("Justification is generic: lacks specific category or row context keywords")

    return ProxySignals(
        is_uniform_phrasing=is_uniform,
        rater_similarity_score=rater_similarity,
        is_suspiciously_polished=is_polished,
        lacks_row_anchors=lacks_anchors,
        proxy_flags_count=len(warnings),
        warnings=warnings
    )
