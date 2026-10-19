"""Generate paired AI answers locally with Ollama (Increments 4c and 4d).

    python -m src.corpus.generate [--sources dolly,oasst1,stackexchange] [--workers 4]
    python -m src.corpus.generate --sources stackexchange,dolly,oasst1 --variant short4d
    python -m src.corpus.generate --sources stackexchange --variant train4d --humanize

Increment 4d variants write to CORPUS_DIR/raw/<source>/generated_<variant>/ and draw a style per item
(persona, plain-text instruction, temperature) and, for short variants, a 25-99-word target; the
default variant ("") keeps 4c's settings so its records stay reproducible. --humanize rewrites 10% of a
variant's answers with a different training generator into generated_<variant>_humanized/.

Each prompt gets one answer from one generator, picked by a hash of its id, at about the length
of the paired human answer. Answers are appended to CORPUS_DIR/raw/<source>/generated/<model>.jsonl
with the model's digest and licence, the options, the seed and the target length. A rerun skips
answers already written, so an interrupted run resumes where it stopped. Nothing leaves the
machine: Ollama runs locally.
"""

import argparse
import hashlib
import os
import json
import logging
import re
import shutil
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

from .paths import corpus_dir
from .trims import trim_preamble

logger = logging.getLogger(__name__)

OLLAMA = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
# Amendment 2: qwen3.5:4b did not fit the 4 GB GPU (25/33 layers, 13 tok/s), so Qwen runs at 2B;
# gemma3:4b fits one request at a time (34/35 layers, 33 tok/s) and runs that way.
TRAINING_GENERATORS = ("llama3.2:3b", "qwen3.5:2b", "gemma3:4b", "ministral-3:3b")
# Increment 4d adds Falcon 3 (tii) as a third family no model trains on.
HELD_OUT_GENERATORS = ("phi4-mini", "granite4:3b", "falcon3:3b")
# The default variant keeps v3's pools, so a resumed default run picks the generators v3 recorded (final
# review I4): pick() hashes over the pool, and Falcon 3 joined only for Increment 4d.
DEFAULT_POOLS = {"dolly": TRAINING_GENERATORS, "oasst1": TRAINING_GENERATORS,
                 "stackexchange": TRAINING_GENERATORS + ("phi4-mini", "granite4:3b")}
SYSTEM_PROMPT = "You are a helpful assistant."
# 2048 tokens covers a 450-word prompt plus an 860-token answer, and leaves GPU memory for batching
# (the first 1,201 Llama answers used 3072; each record keeps its own options).
TEMPERATURE, TOP_P, NUM_CTX = 0.8, 0.95, 2048
MIN_TARGET, MAX_TARGET = 30, 400
MAX_PROMPT_WORDS = 450
# Pre-registration Amendments 1 and 3: prompts answered per source, chosen by a hash of the item id,
# so generation fits the laptop's measured speed. Every human answer is kept.
GENERATION_LIMITS = {"dolly": 1500, "oasst1": 1500, "stackexchange": 1200}
# Pre-registration Amendment 4: artifacts of our prompt, not of AI writing. An answer cut off at the
# token cap ends mid-sentence, and one that restates the requested length at its start ("aiming for
# around 117 words") or end ("(Word count: ~180)") echoes the instruction. Either is a shortcut.
_LENGTH_NOTE = re.compile(r"\b(\d+|~\s*\d+)[\s-]*words?\b|word count|word limit", re.I)
# Amendment 13 (Ralph R12-13): refusals, a word count noted inside the answer and a switch into Chinese,
# Japanese or Korean script got past the Amendment 4 filters. Refusals are looked for where they appear,
# in the opening, and a model speaking of itself anywhere; a bracketed word count anywhere.
_REFUSAL_OPENING = re.compile(r"\bi (?:can(?:no|')t|am unable to|'m unable to|'m not able to) (?:help|assist|provide|"
                              r"answer|give|offer|recommend|advise)\b|\bi(?:'m| am) sorry,? but\b", re.I)
_SELF_AS_MODEL = re.compile(r"\bas an ai\b|\bas a(?:n ai)? language model\b", re.I)
_LENGTH_NOTE_INSIDE = re.compile(r"\(\s*(?:about|around|approx\.?|approximately|~)?\s*\d+\s*words?\s*\)|\bword count\b",
                                 re.I)
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")

# Increment 4d pre-registration: styles drawn per item from a hash of the variant and item id.
PERSONAS = {
    "neutral": SYSTEM_PROMPT,
    "casual": "You are chatting casually with a friend. Write the way people type, not like a textbook.",
    "forum": "You are a regular user answering a question on a Q&A forum.",
    "terse": "You answer briefly and directly.",
}
PLAIN_SHARE = 0.30
PLAIN_NOTE = "Write plain text: no headings, lists, bold or markdown."
TEMPERATURE_RANGE = (0.6, 1.1)
SHORT_RANGE = (25, 99)
HUMANIZE_SHARE = 0.10
HUMANIZE_PROMPT = ("Rewrite the text below so it sounds casual and natural, like a person typed it. Keep the "
                   "meaning and roughly the same length. Reply with the rewrite only.\n\n")
# Generators and item counts per (source, variant): held-out generators answer only the held-out sites.
POOLS_4D = {
    ("stackexchange", "train4d"): TRAINING_GENERATORS,
    ("stackexchange", "short4d"): TRAINING_GENERATORS,
    ("dolly", "short4d"): TRAINING_GENERATORS,
    ("oasst1", "short4d"): TRAINING_GENERATORS,
    ("stackexchange_heldout", "test4d"): TRAINING_GENERATORS + HELD_OUT_GENERATORS,
    ("stackexchange_heldout", "short4d"): TRAINING_GENERATORS + HELD_OUT_GENERATORS,
}
GENERATION_LIMITS_4D = {
    ("stackexchange", "train4d"): 2500, ("stackexchange", "short4d"): 2500,
    ("dolly", "short4d"): 1000, ("oasst1", "short4d"): 1000,
    ("stackexchange_heldout", "test4d"): 2100, ("stackexchange_heldout", "short4d"): 2100,
}
# Every variant folder a loader reads; the humanized rewrites carry their own transform.
LOADED_VARIANTS = ("", "train4d", "short4d", "test4d", "train4d_humanized")


def slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def _hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _salted(kind: str, item_id: str, variant: str) -> str:
    """The default variant keeps 4c's hash inputs, so its generator and seed choices are unchanged."""
    return f"{kind}:{variant}:{item_id}" if variant else f"{kind}:{item_id}"


def pick(item_id: str, pool: tuple, variant: str = "") -> str:
    return pool[_hash(_salted("generator", item_id, variant)) % len(pool)]


def seed_of(item_id: str, variant: str = "") -> int:
    return _hash(_salted("seed", item_id, variant)) % (2**31)


def style_of(item_id: str, variant: str) -> dict:
    """Persona, plain-text flag and temperature for one item of a 4d variant, from its hash."""
    h = hashlib.sha256(f"style:{variant}:{item_id}".encode("utf-8")).digest()
    names = list(PERSONAS)
    lo, hi = TEMPERATURE_RANGE
    return {"persona": names[h[0] % len(names)], "plain": h[1] / 256 < PLAIN_SHARE,
            "temperature": round(lo + (hi - lo) * int.from_bytes(h[2:4], "big") / 65535, 3)}


def short_target(item_id: str, variant: str) -> int:
    lo, hi = SHORT_RANGE
    return lo + _hash(f"short:{variant}:{item_id}") % (hi - lo + 1)


def target_words(human_text: str) -> int:
    return max(MIN_TARGET, min(MAX_TARGET, len(human_text.split())))


def user_message(prompt: str, words: int, plain: bool = False) -> str:
    prompt = " ".join(prompt.split()[:MAX_PROMPT_WORDS]) if len(prompt.split()) > MAX_PROMPT_WORDS else prompt
    return f"{prompt}\n\nAnswer in about {words} words." + (f" {PLAIN_NOTE}" if plain else "")


# Amendment 8: the casual persona opens answers with a greeting ("Hey," "Hi there!") that real answers
# almost never have (Stack Exchange humans 0.03%, against 4-8% of generated answers), which would teach the
# model a shortcut. Loading trims it (and, since Amendment 9, a leading "Okay,"; see trim_opener).
_GREETING = re.compile(r"^\s*(?i:(?:hey|hi|hello|howdy|yo)(?:\s+(?:hey|hi|hello))*)"
                       r"(?:\s+(?:(?i:there|friend|buddy|mate|man|dude|bro/sis|bro|sis|guys|folks|all|everyone"
                       r"|what'?s up|what up)|[A-Z][a-z]+))?"  # bro, sis, what up: Amendment 9
                       r"\s*[,!.:;—-]+\s*")


def _trim(pattern: re.Pattern, text: str) -> str:
    match = pattern.match(text)
    if not match or match.end() >= len(text):
        return text
    rest = text[match.end():]
    return rest[:1].upper() + rest[1:]


def trim_greeting(text: str) -> str:
    """text without a leading greeting, its next word capitalised; unchanged if nothing would be left."""
    return _trim(_GREETING, text)


# Amendment 9: Gemma opens 34% of its answers with "Okay," (0-1% for the other generators), against 0.21% of
# human texts, so a leading "Okay"/"OK" followed by punctuation is trimmed too. Other fillers ("Sure,",
# "Honestly,") stay: humans open with them at comparable rates.
_OKAY = re.compile(r"^\s*(?i:okay|ok)\b(?:\s+[A-Z][a-z]+)?\s*[,!.:;—-]+\s*")


def trim_opener(text: str) -> str:
    """text without leading greetings, "Okay," openers or "Here's a casual version:" announcements, in any
    order (Amendments 8, 9 and 13)."""
    for _ in range(4):
        trimmed = trim_preamble(_trim(_OKAY, _trim(_GREETING, text)))
        if trimmed == text:
            break
        text = trimmed
    return text


def artifact(text: str, done_reason: Optional[str]) -> Optional[str]:
    """Why an answer is unusable under Amendments 4 and 13, or None."""
    if done_reason == "length":
        return "cut_off"
    if _LENGTH_NOTE.search(text.strip().split("\n")[0][:120]):
        return "length_note_opening"
    if _LENGTH_NOTE.search(text[-80:]):
        return "length_note_ending"
    if _REFUSAL_OPENING.search(text[:200]) or _SELF_AS_MODEL.search(text):
        return "refusal"
    if _LENGTH_NOTE_INSIDE.search(text):
        return "length_note_inside"
    if _CJK.search(text):
        return "code_switch"
    return None


def licence_name(text: Optional[str]) -> str:
    """The line naming the licence, not a preamble (phi4-mini's MIT text opens "Microsoft.")."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    named = next((line for line in lines if re.search(r"licen[cs]e|terms", line, re.I)), None)
    return (named or (lines[0] if lines else "unknown"))[:200]


def model_info(model: str) -> dict:
    r = requests.post(f"{OLLAMA}/api/show", json={"model": model}, timeout=60)
    r.raise_for_status()
    info = r.json()
    # /api/show carries no digest; /api/tags lists it (the sha256 of the model's manifest).
    tags = requests.get(f"{OLLAMA}/api/tags", timeout=30)
    tags.raise_for_status()
    digest = next((m.get("digest") for m in tags.json()["models"] if m.get("name") in (model, f"{model}:latest")), None)
    return {"digest": digest, "licence": licence_name(info.get("license")), "modified_at": info.get("modified_at"),
            "parameter_size": info.get("details", {}).get("parameter_size"),
            "quantization": info.get("details", {}).get("quantization_level")}


def chat(model: str, message: str, seed: int, max_tokens: int, *, system: str = SYSTEM_PROMPT,
         temperature: float = TEMPERATURE) -> dict:
    body = {"model": model, "stream": False, "think": False, "keep_alive": "30m",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": message}],
            "options": {"temperature": temperature, "top_p": TOP_P, "seed": seed, "num_predict": max_tokens,
                        "num_ctx": NUM_CTX}}
    r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=900)
    if r.status_code == 400 and "think" in r.text:  # a model without a thinking switch
        body.pop("think")
        r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=900)
    r.raise_for_status()
    return r.json()


def chosen(items: pd.DataFrame, limit: Optional[int], variant: str = "") -> pd.DataFrame:
    """The first `limit` items in a fixed hash order: the same subset on every run."""
    order = items["item_id"].map(lambda i: hashlib.sha256(_salted("subset", i, variant).encode("utf-8")).hexdigest())
    ordered = items.assign(_order=order).sort_values("_order").drop(columns="_order")
    return ordered if limit is None else ordered.head(limit)


def available_models() -> set:
    r = requests.get(f"{OLLAMA}/api/tags", timeout=30)
    r.raise_for_status()
    names = {m["name"] for m in r.json()["models"]}
    return names | {n.removesuffix(":latest") for n in names}  # Ollama lists "phi4-mini" as "phi4-mini:latest"


def done_ids(out_dir: Path) -> set:
    ids = set()
    for path in out_dir.glob("*.jsonl"):
        with path.open(encoding="utf-8") as fh:
            ids.update(json.loads(line)["item_id"] for line in fh if line.strip())
    return ids


def generated_dir(raw: Path, variant: str = "") -> Path:
    return raw / ("generated" if not variant else f"generated_{variant}")


def generate(source: str, items: pd.DataFrame, pool: tuple, *, root: Optional[Path] = None,
             workers: int = 8, limit: Optional[int] = None, models: Optional[set] = None, variant: str = "",
             chat_fn: Callable = chat, info_fn: Callable = model_info) -> dict:
    """items has item_id, prompt, human_text. Answers the chosen subset with the generators in
    `models` (default: all of `pool`); returns counts written per model this run. A non-empty variant
    writes to its own folder and draws a style (and, for short variants, a length) per item."""
    out_dir = generated_dir((root or corpus_dir()) / "raw" / source, variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = chosen(items, limit, variant)
    todo = todo[~todo["item_id"].isin(done_ids(out_dir))].copy()
    todo["generator"] = todo["item_id"].map(lambda i: pick(i, pool, variant))
    if models is not None:
        todo = todo[todo["generator"].isin(models)]
    written: dict = {}
    for model, part in todo.groupby("generator", sort=True):
        info = info_fn(model)
        lock = threading.Lock()
        path = out_dir / f"{slug(model)}.jsonl"
        logger.info("%s: %d answers with %s", source, len(part), model)

        def one(row) -> None:
            style = style_of(row.item_id, variant) if variant else None
            words = short_target(row.item_id, variant) if variant.startswith("short") else target_words(row.human_text)
            seed, max_tokens = seed_of(row.item_id, variant), 2 * words + 60
            system = PERSONAS[style["persona"]] if style else SYSTEM_PROMPT
            temperature = style["temperature"] if style else TEMPERATURE
            message = user_message(row.prompt, words, plain=bool(style and style["plain"]))
            try:
                reply = chat_fn(model, message, seed, max_tokens, system=system, temperature=temperature)
            except Exception as exc:  # recorded as missing; the next run retries it
                logger.warning("%s %s failed: %s", model, row.item_id, exc.__class__.__name__)
                return
            record = {"item_id": row.item_id, "source": source, "generator": model, "model_digest": info["digest"],
                      "model_licence": info["licence"], "quantization": info.get("quantization"),
                      "options": {"temperature": temperature, "top_p": TOP_P, "num_ctx": NUM_CTX,
                                  "num_predict": max_tokens, "seed": seed, "system": system},
                      "variant": variant, "seed_salt": variant, **({"style": style} if style else {}),
                      "target_words": words, "prompt_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
                      "text": (reply.get("message") or {}).get("content", ""), "done_reason": reply.get("done_reason"),
                      "eval_count": reply.get("eval_count"), "created_at": datetime.now(timezone.utc).isoformat()}
            with lock:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written[model] = written.get(model, 0) + 1
                if written[model] % 100 == 0:
                    logger.info("%s %s: %d/%d", source, model, written[model], len(part))

        with ThreadPoolExecutor(max_workers=workers) as pool_exec:
            list(pool_exec.map(one, part.itertuples()))
    return written


def load_generated(raw: Path, variant: str = "") -> pd.DataFrame:
    """Every usable generated answer of one variant for a source, newest record per item. Amendment 4
    artifacts are dropped here; the raw files keep them."""
    rows = []
    for path in sorted(generated_dir(raw, variant).glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    if not rows:
        return pd.DataFrame(columns=["item_id", "generator", "text", "model_licence"])
    df = pd.DataFrame(rows).drop_duplicates("item_id", keep="last")
    done = df["done_reason"] if "done_reason" in df else pd.Series(None, index=df.index, dtype=object)
    reasons = pd.Series([artifact(str(t), d) for t, d in zip(df["text"], done)], index=df.index)
    if reasons.notna().any():
        logger.info("%s: Amendment 4 drops %s", raw.name, dict(Counter(reasons.dropna())))
    kept = df[reasons.isna()]
    trimmed = kept["text"].map(lambda t: trim_opener(str(t)))
    changed = int((trimmed != kept["text"]).sum())
    if changed:
        logger.info("%s: Amendments 8 and 9 trim %d leading greetings or \"Okay,\" openers", raw.name, changed)
    return kept.assign(text=trimmed)


def load_all_generated(raw: Path) -> pd.DataFrame:
    """Every usable generated answer of every variant a loader reads, with its variant and transform
    (humanized rewrites are llm_rewrite_casual)."""
    frames = []
    for variant in LOADED_VARIANTS:
        df = load_generated(raw, variant)
        if len(df):
            frames.append(df.assign(variant=variant,
                                    transform="llm_rewrite_casual" if variant.endswith("_humanized") else "none"))
    if not frames:
        return pd.DataFrame(columns=["item_id", "generator", "text", "model_licence", "variant", "transform"])
    return pd.concat(frames, ignore_index=True)


def humanize(source: str, variant: str, rewriter_pool: tuple, *, root: Optional[Path] = None,
             share: float = HUMANIZE_SHARE, workers: int = 8, models: Optional[set] = None,
             chat_fn: Callable = chat, info_fn: Callable = model_info) -> int:
    """Rewrite `share` of a variant's answers, chosen by hash, with the next generator in rewriter_pool
    after the one that wrote each answer; returns how many rewrites this run wrote."""
    raw = (root or corpus_dir()) / "raw" / source
    target = f"{variant}_humanized"
    out_dir = generated_dir(raw, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    answers = load_generated(raw, variant)
    answers = answers[answers["item_id"].map(lambda i: _hash(f"humanize:{variant}:{i}") % 10_000 < share * 10_000)]
    answers = answers[~answers["item_id"].isin(done_ids(out_dir))].copy()

    def rewriter(original: str) -> str:
        position = rewriter_pool.index(original) if original in rewriter_pool else -1
        return rewriter_pool[(position + 1) % len(rewriter_pool)]

    answers["rewriter"] = answers["generator"].map(rewriter)
    if models is not None:
        answers = answers[answers["rewriter"].isin(models)]
    written = 0
    for model, part in answers.groupby("rewriter", sort=True):
        info = info_fn(model)
        lock = threading.Lock()
        path = out_dir / f"{slug(model)}.jsonl"

        def one(row) -> None:
            nonlocal written
            words = len(str(row.text).split())
            seed, max_tokens = seed_of(row.item_id, target), 2 * words + 60
            message = HUMANIZE_PROMPT + str(row.text)
            try:
                reply = chat_fn(model, message, seed, max_tokens, system=SYSTEM_PROMPT, temperature=TEMPERATURE)
            except Exception as exc:  # recorded as missing; the next run retries it
                logger.warning("%s rewrite of %s failed: %s", model, row.item_id, exc.__class__.__name__)
                return
            record = {"item_id": row.item_id, "source": source, "generator": model, "rewritten_from": row.generator,
                      "model_digest": info["digest"], "model_licence": info["licence"],
                      "quantization": info.get("quantization"), "variant": target, "seed_salt": target,
                      "options": {"temperature": TEMPERATURE, "top_p": TOP_P, "num_ctx": NUM_CTX,
                                  "num_predict": max_tokens, "seed": seed, "system": SYSTEM_PROMPT},
                      "source_text_sha256": hashlib.sha256(str(row.text).encode("utf-8")).hexdigest(),
                      "text": (reply.get("message") or {}).get("content", ""), "done_reason": reply.get("done_reason"),
                      "eval_count": reply.get("eval_count"), "created_at": datetime.now(timezone.utc).isoformat()}
            with lock:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

        with ThreadPoolExecutor(max_workers=workers) as pool_exec:
            list(pool_exec.map(one, part.itertuples()))
    return written


def backfill_provenance(out_dir: Path, info: dict, *, note: str) -> dict:
    """Fill each record's missing digest, and a licence line taken before model_info read the whole
    licence text, from `info` (generator -> model_info). A model modified after its first answer is
    refused, since its digest then need not be the one that wrote them. Originals go to out_dir/backup."""
    files = {}
    for path in sorted(out_dir.glob("*.jsonl")):
        records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        todo = [r for r in records if r["generator"] in info and
                (not r.get("model_digest") or r.get("model_licence") != info[r["generator"]]["licence"])]
        if todo:
            files[path] = (records, todo)
    for records, todo in files.values():
        for model in {r["generator"] for r in todo}:
            first = min(pd.Timestamp(r["created_at"]) for r in records if r["generator"] == model)
            if pd.Timestamp(info[model]["modified_at"]) >= first:
                raise ValueError(f"{model} was modified at {info[model]['modified_at']}, after its first "
                                 f"answer at {first}; its current digest may not be the one that wrote them")
    changed: dict = {}
    for path, (records, todo) in files.items():
        backup = out_dir / "backup" / path.name
        if not backup.exists():
            backup.parent.mkdir(exist_ok=True)
            shutil.copy2(path, backup)
        for r in todo:
            meta = info[r["generator"]]
            r.update(model_digest=r.get("model_digest") or meta["digest"], model_licence=meta["licence"],
                     provenance_backfill=note)
            changed[r["generator"]] = changed.get(r["generator"], 0) + 1
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        tmp.replace(path)
    return changed


def main(argv: Optional[list[str]] = None) -> int:
    from .sources import dolly, oasst1, stackexchange, stackexchange_heldout

    modules = {"dolly": dolly, "oasst1": oasst1, "stackexchange": stackexchange,
               "stackexchange_heldout": stackexchange_heldout}
    # The default variant keeps 4c's pools and limits.
    plans = DEFAULT_POOLS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", default="dolly,oasst1,stackexchange")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--models", default=None, help="comma-separated generators to run this pass")
    parser.add_argument("--variant", default="", help="Increment 4d variant (train4d, short4d, test4d)")
    parser.add_argument("--humanize", action="store_true",
                        help="rewrite 10%% of the variant's answers with a different training generator")
    parser.add_argument("--backfill-provenance", action="store_true",
                        help="fill missing digests and licence lines in existing answers (Amendment 4), then exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from .sources._common import download

    names = args.sources.split(",")
    if args.backfill_provenance:
        note = (f"Amendment 4, {datetime.now(timezone.utc).date()}: digest from /api/tags and licence line from "
                "the full licence text, read after generation from a local model unchanged since before it")
        for name in names:
            out_dir = corpus_dir() / "raw" / name / "generated"
            models = {json.loads(line)["generator"] for path in out_dir.glob("*.jsonl")
                      for line in path.open(encoding="utf-8") if line.strip()}
            print(name, backfill_provenance(out_dir, {m: model_info(m) for m in sorted(models)}, note=note))
        return 0
    variant = args.variant
    if variant:
        missing = [n for n in names if (n, variant) not in POOLS_4D]
        if missing:
            parser.error(f"no pre-registered {variant!r} plan for {missing}")
        pools = {n: POOLS_4D[(n, variant)] for n in names}
        limits = {n: GENERATION_LIMITS_4D[(n, variant)] for n in names}
    else:
        pools = {n: plans[n] for n in names}
        limits = {n: GENERATION_LIMITS.get(n) for n in names}
    ready = available_models()
    wanted = set(args.models.split(",")) if args.models else None
    order = [m for m in TRAINING_GENERATORS + HELD_OUT_GENERATORS
             if (wanted is None or m in wanted) and any(m in pools[n] for n in names)]
    skipped = [m for m in order if m not in ready]
    for model in skipped:
        logger.warning("%s is not pulled yet; skipping it this run", model)
    order = [m for m in order if m in ready]
    if args.humanize:
        # Rewrites run one rewriter at a time too.
        for model in order:
            for name in names:
                print(name, f"{variant}_humanized", model,
                      humanize(name, variant, TRAINING_GENERATORS, workers=args.workers, models={model}))
        return 0
    items = {}
    for name in names:
        module = modules[name]
        raw = corpus_dir() / "raw" / name
        for url, rel in module.SPEC.files:
            download(url, raw / rel)
        items[name] = module.pairs(raw)
    # One model at a time across all sources, so the server never swaps models mid-run.
    for model in order:
        for name in names:
            if model in pools[name]:
                print(name, variant or "default", model,
                      generate(name, items[name], pools[name], workers=args.workers, limit=limits[name],
                               models={model}, variant=variant))
    return 0


if __name__ == "__main__":
    sys.exit(main())
