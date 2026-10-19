# Increment 4d, Plan A: corpus v4 implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Build and audit corpus v4. It adds the new public sources and the `format3` view, puts Stack
Exchange into training, and has a fresh v4 locked test that no earlier model has seen.

**Architecture:**
- Each new source is a loader module in `src/corpus/sources/` that exposes `SPEC` and
  `load(raw, cap)`, like the existing ones.
- Retirement widens from "earlier locked tests" to "every row an earlier model fitted or was tested
  on", checked by id, group and near-duplicate cluster in both the build and the independent audit.
- Generation gains variants: native short answers, style settings, a humanized rewrite pass, and
  held-out answers for the new Stack Exchange sites. Each variant writes its own folder and records
  its settings.

**Tech stack:** Python 3.12, pandas 3, pydantic, pytest, Ollama (local, 4-bit), the existing
`src/corpus` pipeline.

**Spec:** `docs/superpowers/specs/2026-09-28-increment-4d-detection-design.md`

**Later plans:**
- Plan B: training, detector precision, perplexity stack and style thresholds. It follows the screening
  results.
- Plan C: head-to-head locked run and report.

## Global constraints

- **Where to work:** all code changes happen in the worktree `..\excel-ai-qa-pipeline-4d` (branch
  `increment-4d`). The main checkout keeps serving the sealed v5 model; editing its sealed files would
  close today's gate.
- **Existing views keep their behaviour.** Every existing view name keeps its exact behaviour, and
  `_FINGERPRINT_CASES` must not change: the v5 registry checks `format2`'s fingerprint.
- **Licences:** every sample records a licence, and the licence class is one of permissive, copyleft,
  non_commercial, internal or unknown.
- **No training on in-house sheets.** The `legacy_inhouse` rows stay in the `inhouse` split (report only).
- **One model at a time:** one Ollama model loaded at a time, and one GPU job at a time.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Amendments before results:** the pre-registration (Task 8) is committed before any v4 generation
  or build runs.
- **The test suite passes on a machine with no GPU and no model files.**

## Review focus

1. **An earlier version's training row reaches the v4 test through a near-duplicate** (a quoted or
   lightly edited copy). Expected: the build keeps it out of evaluation, and the audit fails if one gets
   through. Pinned in Task 3, Step 1 (`test_near_duplicate_of_an_earlier_fitted_row_is_never_evaluated`).
2. **Non-English or text-less rows in WasItAI** (lang ≠ en, or `text` null because it isn't
   redistributed). Expected: the loader skips them, and never re-fetches from `source_url`. Pinned in
   Task 5.
3. **Applying `format3` twice changes the text again** (`**1.** Step`). Expected: one application gives
   the final result. Pinned in Task 1.
4. **An AI-only source that is trivially separable by layout** (Arena markdown survives the view).
   Expected: the build record's AI-only probe AUC is above 0.85, and the source is excluded before
   training. Pinned in Task 7.
5. **In-house and stress rows dropped by the wider retirement.** Expected: they are report-only, so
   earlier versions' in-house and stress rows stay in their splits. Pinned in Task 2.

---

### Task 1: `format3` text view (closes D14 and D15)

**Files:**
- Modify: `src/corpus/text.py` (add the view; `MODEL_TEXT_VIEW` becomes `format3+typography+collapse_whitespace`)
- Test: `tests/test_text_views.py` (new)

**Interfaces:**
- Produces:
  - `text_view("format3+typography+collapse_whitespace") -> Callable[[str], str]`
  - `MODEL_TEXT_VIEW == "format3+typography+collapse_whitespace"`
  - `text_view_fingerprint(name)`, unchanged for the four earlier views

- [ ] **Step 1: Write the failing tests**

```python
"""format3 (Increment 4d, debts D14 and D15): every bullet glyph and (1) marker goes, dashes follow one
rule, and the view is idempotent. Earlier views keep their exact behaviour."""

import pytest

from src.corpus.text import MODEL_TEXT_VIEW, text_view, text_view_fingerprint

F3 = text_view("format3+typography+collapse_whitespace")
F2 = text_view("format2+typography+collapse_whitespace")

# Fingerprints the v5 registry was sealed with must not move (read from models/registry.json).
FORMAT2_FINGERPRINT = "f3289b1f8878b99eae20375b518b4e5bb6986540e86268047f58918fcc2e8eaa"


def test_model_view_is_format3():
    assert MODEL_TEXT_VIEW == "format3+typography+collapse_whitespace"


def test_earlier_view_fingerprints_are_unchanged():
    assert text_view_fingerprint("format2+typography+collapse_whitespace") == FORMAT2_FINGERPRINT


@pytest.mark.parametrize("glyph", ["\u00b7", "\u2219", "\u25cf", "\u25cb", "\u25a0", "\u25a1", "\u25ab",
                                   "\u2027", "\u25b8", "\u25ba", "\u2713", "\u2714", "\u27a4", "\u2022"])
def test_every_bullet_glyph_is_dropped(glyph):
    assert F3(f"{glyph} salt {glyph} pepper") == "salt pepper"


def test_parenthesised_and_bare_numbered_markers_are_dropped():
    assert F3("Do this: (1) mix (2) bake 3) serve") == "Do this: mix bake serve"
    assert F3("version (2024) stays") == "version (2024) stays"  # four digits are not a marker


def test_view_is_idempotent_on_bold_numbers():
    once = F3("**1.** Step one\n**2.** Step two")
    assert once == "Step one Step two"
    assert F3(once) == once


@pytest.mark.parametrize("text", ["tea\u2014then bed", "tea \u2014 then bed", "tea -- then bed",
                                  "tea\u2013then bed", "tea - then bed"])
def test_spaced_and_unspaced_dashes_follow_one_rule(text):
    assert F3(text) == "tea then bed"


def test_word_hyphens_stay():
    assert F3("a well-known trick") == "a well-known trick"


def test_format2_behaviour_is_untouched():
    assert F2("tea\u2014then bed") == "tea-then bed"  # the D15 behaviour stays under its old name
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `python -m pytest tests/test_text_views.py -q`
Expected: FAIL with `ValueError: unknown text view 'format3+typography+collapse_whitespace'`.

- [ ] **Step 3: Implement the view in `src/corpus/text.py`**

Add the following after `strip_formatting_inline`:

```python
# Increment 4d (debts D14, D15): other bullet glyphs and (1)/1) markers survived format2, a bold
# number ("**1.**") needed two passes, and an unspaced em dash became a word hyphen while a spaced one
# was dropped. format3 drops every bullet glyph and short numbered marker, treats every en or em dash
# (and a free-standing "--") as a space before typography runs, and repeats until nothing changes, so
# one application is final.
_BULLETS3 = re.compile(r"[•‣⁃▪◦·∙●○■□▫‧"
                       r"▸►✓✔➤]")
_NUMBERED3 = re.compile(r"(?<!\S)(?:\(\d{1,2}\)|\d{1,2}\))(?!\S)")
_DASHES3 = re.compile(r"[–—―]|(?<![-\w])-{2,3}(?![-\w])")


def strip_formatting3(text: str) -> str:
    for _ in range(4):
        cleaned = _NUMBERED3.sub(" ", _BULLETS3.sub(" ", strip_formatting_inline(text)))
        if cleaned == text:
            break
        text = cleaned
    return text
```

Add the view to `_TEXT_VIEWS`. The dashes go first, because typography would otherwise turn en and em
dashes into "-":

```python
    "format3+typography+collapse_whitespace":
        lambda text: collapse_whitespace(strip_formatting3(normalize_typography(_DASHES3.sub(" ", text)))),
```

Set `MODEL_TEXT_VIEW = "format3+typography+collapse_whitespace"`. Leave `_FINGERPRINT_CASES` unchanged, so
earlier views keep their fingerprints. Add cases that only views from format3 on are fingerprinted with,
so a change to the new rules is caught:

```python
# Views added from Increment 4d on also fingerprint the cases their new rules cover.
_EXTRA_FINGERPRINT_CASES = {
    "format3+typography+collapse_whitespace": ("\u00b7 dot (1) one 2) two tea\u2014then", "**1.** Bold\n**2.** Two"),
}
```

In `text_view_fingerprint`, fingerprint `_FINGERPRINT_CASES + _EXTRA_FINGERPRINT_CASES.get(name, ())`.
Add this test:

```python
def test_format3_fingerprint_covers_its_new_rules():
    assert text_view_fingerprint("format3+typography+collapse_whitespace") != FORMAT2_FINGERPRINT
```

- [ ] **Step 4: Run the tests to see them pass, then run the full suite**

Run: `python -m pytest tests/test_text_views.py -q`, then `python -m pytest -q`.
Expected: all tests pass. If a test asserts `MODEL_TEXT_VIEW == "format2..."` (such as
`test_train_keeps_best_epoch_and_writes_a_text_free_record`), update it to format3: the training record
reads `MODEL_TEXT_VIEW`.

- [ ] **Step 5: Commit**

```bash
git add src/corpus/text.py tests/test_text_views.py tests/test_authorship_train.py
git commit -m "feat: format3 view drops every bullet glyph and numbered marker and is idempotent (D14, D15)"
```

---

### Task 2: v4 version and wider retirement

**Files:**
- Modify: `src/corpus/paths.py` (`VERSION = "v4"`)
- Modify: `src/corpus/build.py` (`retirement` reads fitted and tested rows)
- Modify: `src/corpus/audit.py` (`earlier_test_groups` becomes `earlier_seen_groups` with the same split set)
- Test: `tests/test_corpus_v4.py` (new)

**Interfaces:**
- Produces:
  - `build.SEEN_SPLITS = ("train", "val", "test", "test_unseen_generator")`
  - `build.retirement(root, versions, retired=None, retired_groups=None) -> (ids, groups)`, which now
    reads `SEEN_SPLITS`
  - `audit.earlier_seen_groups(base, version) -> frozenset`

- [ ] **Step 1: Write the failing tests**

```python
"""Corpus v4 retirement: nothing an earlier model fitted or was tested on may be evaluated again."""

import pandas as pd

from src.corpus import build as B
from src.corpus.audit import earlier_seen_groups
from src.corpus.paths import VERSION


def _manifest(root, version, rows):
    (root / version).mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(root / version / "manifest.parquet", index=False)


def test_version_is_v4():
    assert VERSION == "v4"


def test_retirement_covers_fitted_and_tested_rows_but_not_report_only_ones(tmp_path):
    for v, rows in (("v1", [{"sample_id": "a", "group_key": "g:a", "split": "train"},
                            {"sample_id": "b", "group_key": "g:b", "split": "stress"}]),
                    ("v2", [{"sample_id": "c", "group_key": "g:c", "split": "val"},
                            {"sample_id": "d", "group_key": "g:d", "split": "inhouse"}]),
                    ("v3", [{"sample_id": "e", "group_key": "g:e", "split": "test"},
                            {"sample_id": "f", "group_key": "g:f", "split": "test_unseen_generator"}])):
        _manifest(tmp_path, v, rows)
    ids, groups = B.retirement(tmp_path, ("v1", "v2", "v3"))
    assert ids == {"a", "c", "e", "f"}
    assert groups == {"g:a", "g:c", "g:e", "g:f"}
    assert earlier_seen_groups(tmp_path, "v4") == groups
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `python -m pytest tests/test_corpus_v4.py -q`
Expected: FAIL (`VERSION` is `"v3"`, and `earlier_seen_groups` doesn't exist).

- [ ] **Step 3: Implement**

In `src/corpus/paths.py`, change the version and extend the comment:

```python
# ... v4 (Increment 4d) adds MAGA, DACTYL, Arena and OpenTuringBench, fits Stack Exchange, and tests on
# rows no earlier model fitted or was tested on.
VERSION = "v4"
```

In `src/corpus/build.py`, replace `_retired` so it reads every seen split:

```python
# Increment 4d: an earlier model fitted on train and validation rows and was judged on test rows, so
# none may be evaluated again. Stress and in-house rows are report-only and stay available.
SEEN_SPLITS = ("train", "val", "test", "test_unseen_generator")


def _retired(root: Path, versions, column: str) -> frozenset:
    values: set = set()
    for version in versions:
        manifest = root / version / "manifest.parquet"
        if not manifest.exists():
            raise BuildError(f"cannot retire {version}'s rows: {manifest} is missing")
        values |= set(pd.read_parquet(manifest, columns=[column], filters=[("split", "in", list(SEEN_SPLITS))])[column])
    return frozenset(values)
```

Rename `retired_test_ids` to `retired_seen_ids` and `retired_test_groups` to `retired_seen_groups`, and
update their callers (`retirement`, and `tests/test_corpus.py::test_retired_test_ids_read_the_previous_manifest`).

In `src/corpus/audit.py`, replace `earlier_test_groups` with the following, and use it in `main`:

```python
def earlier_seen_groups(base: Path, version: str) -> frozenset:
    """Group keys of every row an earlier version fitted or tested. A missing manifest is an error:
    skipping it would silently let a retired group be tested again (Ralph R08-10)."""
    from .build import SEEN_SPLITS

    retired: set = set()
    for earlier in previous_versions(version):
        manifest = base / earlier / "manifest.parquet"
        if not manifest.exists():
            raise FileNotFoundError(f"{manifest} is missing, so {earlier}'s retired groups can't be checked")
        retired |= set(pd.read_parquet(manifest, columns=["group_key"],
                                       filters=[("split", "in", list(SEEN_SPLITS))])["group_key"])
    return frozenset(retired)
```

- [ ] **Step 4: Run the tests to see them pass, then run the full suite**

Run: `python -m pytest tests/test_corpus_v4.py tests/test_corpus.py -q`, then `python -m pytest -q`.
Expected: all tests pass. Update the renamed functions in the old tests; their assertions stay.

- [ ] **Step 5: Commit** with message `feat: corpus v4 retires every row an earlier model fitted or was tested on`.

---

### Task 3: near-duplicate retirement against earlier fitted text

**Files:**
- Modify: `src/corpus/build.py` (`seen_by_cluster`, called before `settle_splits`)
- Modify: `src/corpus/audit.py` (`audit(..., earlier_texts=...)` check)
- Test: `tests/test_corpus_v4.py`

**Interfaces:**
- Produces:
  - `build.earlier_texts(root, versions) -> list[str]`: texts of `SEEN_SPLITS` rows from each earlier
    `corpus.parquet`
  - `build.seen_by_cluster(texts: list[str], earlier: list[str]) -> list[bool]`: True where a v4 text
    shares a containment cluster with an earlier text
  - `audit.audit(df, locked, retired_groups=..., earlier_texts=())`

- [ ] **Step 1: Write the failing tests**

```python
from src.corpus.build import seen_by_cluster
from src.corpus.audit import audit


LONG = ("The river bends twice before the old mill, where the path narrows and the reeds grow tall "
        "enough to hide a heron; walkers usually turn back at the stile unless the ground is dry.")


def test_near_duplicate_of_an_earlier_fitted_row_is_never_evaluated():
    fresh = "A completely different paragraph about sourdough starters and how often to feed them daily."
    flags = seen_by_cluster([LONG + " Quoted again.", fresh], earlier=[LONG])
    assert flags == [True, False]


def test_audit_fails_when_an_evaluation_row_matches_earlier_fitted_text(built):
    df, lock = built  # fixture from tests/test_corpus.py: a built toy corpus and its lock
    victim = df.index[df["split"] == "test"][0]
    report = audit(df, lock, earlier_texts=[df.loc[victim, "text"]])
    assert any("earlier version fitted or tested" in e for e in report.errors)
```

Move the `built` fixture from `tests/test_corpus.py` into `tests/conftest.py`, unchanged, so both files
can use it.

- [ ] **Step 2: Run the tests to see them fail**

Run: `python -m pytest tests/test_corpus_v4.py -q`
Expected: FAIL (ImportError for `seen_by_cluster`).

- [ ] **Step 3: Implement**

In `src/corpus/build.py`:

```python
def earlier_texts(root: Path, versions=PREVIOUS_VERSIONS) -> list:
    """Texts of every row an earlier version fitted or tested, for near-duplicate retirement."""
    texts: list = []
    for version in versions:
        path = root / version / "corpus.parquet"
        if not path.exists():
            raise BuildError(f"cannot check near-duplicates of {version}: {path} is missing")
        texts += pd.read_parquet(path, columns=["text"], filters=[("split", "in", list(SEEN_SPLITS))])["text"].tolist()
    return texts


def seen_by_cluster(texts: list, earlier: list) -> list:
    """True for each text that shares a containment cluster with an earlier version's fitted or tested
    text, so a quoted or lightly edited copy is retired like the original."""
    clusters = dedup.cluster_with_containment(list(texts) + list(earlier))
    earlier_clusters = set(clusters[len(texts):])
    return [c in earlier_clusters for c in clusters[:len(texts)]]
```

In `build()`, after `df["cluster_id"] = ...` and `retirement(...)`, cluster only rows that could still be
evaluated. Rows already retired by id or group can't be evaluated anyway, and clustering everything
would not fit this machine's RAM:

```python
    open_rows = ~(df["sample_id"].isin(retired) | df["group_key"].isin(retired_groups))
    near = pd.Series(False, index=df.index)
    near[open_rows] = seen_by_cluster(df.loc[open_rows, "text"].tolist(), earlier_texts(root))
    retired = frozenset(retired) | frozenset(df.loc[near, "sample_id"])
    logger.info("%d rows retired as near-duplicates of earlier fitted or tested text", int(near.sum()))
```

Also record `"near_duplicate_retired": int(near.sum())` in the build record.

In `src/corpus/audit.py`, add the parameter `earlier_texts: Sequence[str] = ()` and this check after the
`retired_groups` check:

```python
    if len(earlier_texts):
        evaluation = df["split"].isin([Split.TEST.value, Split.TEST_UNSEEN_GENERATOR.value])
        ev = df[evaluation]
        clusters = dedup.cluster_with_containment(ev["text"].tolist() + list(earlier_texts))
        old = set(clusters[len(ev):])
        hits = [sid for sid, c in zip(ev["sample_id"], clusters[:len(ev)]) if c in old]
        if hits:
            report.errors.append(f"{len(hits)} evaluation samples match text an earlier version fitted or "
                                 f"tested, e.g. {hits[0][:12]}")
```

In `audit.main` and in `build()`'s audit call, pass
`earlier_texts=earlier_texts(root)`. Import `earlier_texts` from `.build` inside `audit.main` to avoid a
circular import.

- [ ] **Step 4: Run the tests to see them pass, then run the full suite**

Run: `python -m pytest tests/test_corpus_v4.py tests/test_corpus.py -q`, then `python -m pytest -q`.
Expected: all tests pass. `build()` tests that use a temp root with no earlier versions pass
`versions=()`: give `build()` a `previous_versions` parameter (default `PREVIOUS_VERSIONS`), passed to
both `retirement` and `earlier_texts`.

- [ ] **Step 5: Commit** with message `feat: retire near-duplicates of earlier fitted or tested text`.

---

### Task 4: held-out families and sources for v4

**Files:**
- Modify: `src/corpus/split.py` (`HOLDOUT_FAMILIES`, `HOLDOUT_SOURCES`, `SOURCE_TEST_SHARE`)
- Modify: `src/corpus/sources/_common.py` (`_FAMILIES`: falcon → tii)
- Test: `tests/test_corpus_v4.py`

**Interfaces:**
- Produces:
  - `HOLDOUT_FAMILIES == frozenset({"bigscience", "microsoft", "ibm", "tii"})`
  - `HOLDOUT_SOURCES == frozenset({"stackexchange_heldout", "wasitai2026", "lede"})`
  - `generator_family("falcon3:3b") == "tii"`

- [ ] **Step 1: Write the failing test**

```python
from src.corpus.split import HOLDOUT_FAMILIES, HOLDOUT_SOURCES
from src.corpus.sources._common import generator_family


def test_v4_holdouts():
    # Cohere and Moonshot are fitted now (Arena has Command A and Kimi K2); Phi, Granite and Falcon
    # answer only the held-out Stack Exchange sites, and no training source has BLOOM.
    assert HOLDOUT_FAMILIES == {"bigscience", "microsoft", "ibm", "tii"}
    assert HOLDOUT_SOURCES == {"stackexchange_heldout", "wasitai2026", "lede"}
    assert generator_family("falcon3:3b") == "tii"
```

- [ ] **Step 2: Run it to see it fail.** Run `python -m pytest tests/test_corpus_v4.py::test_v4_holdouts -q`.

- [ ] **Step 3: Implement**

In `split.py`:

```python
# Increment 4d: Cohere and Moonshot are fitted (Arena supplies Command A and Kimi K2). Phi-4-mini,
# Granite 4 and Falcon 3 answer only the held-out Stack Exchange sites; no training source has BLOOM.
HOLDOUT_FAMILIES = frozenset({"bigscience", "microsoft", "ibm", "tii"})
# v4 tests on eight new pre-ChatGPT Stack Exchange sites and two frontier sets no model trains on;
# v3's eight Stack Exchange sites are fitted now that its test is spent.
HOLDOUT_SOURCES = frozenset({"stackexchange_heldout", "wasitai2026", "lede"})
```

Keep `SOURCE_TEST_SHARE`, and add `"maga": 0.20, "dactyl": 0.20` so their test slices clear 300
humans per domain.

In `_common.py`, add `(r"falcon", "tii"),` before the `gpt` pattern.

Update the tests in `tests/test_corpus.py` that construct frames with `cohere` as a held-out family:
pass `holdout_families=frozenset({"cohere"})` to `assign_splits` explicitly, which keeps their meaning.

- [ ] **Step 4: Run the full suite.** Run `python -m pytest -q`; all tests pass.

- [ ] **Step 5: Commit** with message `feat: v4 held-out families (bigscience, microsoft, ibm, tii) and sources`.

---

### Task 5: loaders for the test-only sources (WasItAI 2026-09, LEDE, new Stack Exchange sites)

**Files:**
- Create: `src/corpus/sources/wasitai2026.py`, `src/corpus/sources/lede.py`,
  `src/corpus/sources/stackexchange_heldout.py`
- Modify: `src/corpus/sources/__init__.py` (register them)
- Test: `tests/test_sources_v4.py` (new)

**Interfaces:**
- Consumes: `SourceSpec`, `make_sample`, `compact`, `take_per_stratum` from `_common.py`, and
  `stackexchange.to_text` and `stackexchange._best`.
- Produces: modules with `SPEC` and `load(raw, cap)`, plus `stackexchange_heldout.pairs(raw)` for
  generation.

- [ ] **Step 1: Confirm the columns** (the data is already downloaded)

Run:

```bash
python -c "import pandas as pd; w=pd.read_parquet('D:/llm-authorship-corpus/raw/wasitai2026/data/benchmark.parquet'); print(w.dtypes)"
python -c "import pandas as pd, glob; f=sorted(glob.glob('D:/llm-authorship-corpus/raw/lede/data/Eng/*.csv'))[0]; print(pd.read_csv(f, nrows=2).dtypes)"
```

WasItAI columns (confirmed 2026-09-28): `id, label (human|ai|humanized|ai_edited), group, lang, words,
generator, vendor, humanizer, humanizer_mode, source, source_dataset, source_url, text, text_available,
text_license`. The LEDE column names come from this step; write them into the loader's `_COLUMNS`
constant.

- [ ] **Step 2: Write the failing tests**

```python
"""Increment 4d test-only sources: WasItAIGenerated 2026-09, LEDE (English frontier), new Stack Exchange sites."""

import pandas as pd

from src.corpus.schema import Label, Role
from src.corpus.sources import lede, stackexchange_heldout, wasitai2026


def test_wasitai_keeps_english_full_text_rows_and_makes_humanized_rows_stress(tmp_path):
    (tmp_path / "data").mkdir()
    base = {"group": "g1", "words": 60, "vendor": "OpenAI", "humanizer": None, "humanizer_mode": None,
            "source": "fineweb", "source_dataset": "x", "source_url": "http://example.org", "text_available": True}
    pd.DataFrame([
        {**base, "id": "1", "label": "human", "lang": "en", "generator": None, "text": "A person wrote this " * 5,
         "text_license": "CC-BY-4.0"},
        {**base, "id": "2", "label": "ai", "lang": "en", "generator": "gpt-6", "text": "A model wrote this " * 5,
         "text_license": "CC-BY-4.0"},
        {**base, "id": "3", "label": "humanized", "lang": "en", "generator": "claude-opus-5.5",
         "humanizer": "undetectable", "text": "A model then a humanizer " * 5, "text_license": "CC-BY-4.0"},
        {**base, "id": "4", "label": "human", "lang": "de", "generator": None, "text": "Ein Mensch " * 5,
         "text_license": "CC-BY-4.0"},
        {**base, "id": "5", "label": "human", "lang": "en", "generator": None, "text": None,
         "text_license": "not redistributed (third-party copyright)", "text_available": False},
        {**base, "id": "6", "label": "ai_edited", "lang": "en", "generator": "gpt-6", "text": "Edited " * 5,
         "text_license": "MIT"},
    ]).to_parquet(tmp_path / "data" / "benchmark.parquet", index=False)
    got = list(wasitai2026.load(tmp_path, 100))
    assert sorted(s.source_record_id for s in got) == ["1", "2", "3"]
    stress = next(s for s in got if s.source_record_id == "3")
    assert stress.role is Role.STRESS and stress.label is Label.AI and stress.transform == "humanizer:undetectable"
    assert {s.licence for s in got} == {"CC-BY-4.0"}
    assert all(s.group_key.startswith("wasitai2026:") for s in got)


def test_stackexchange_heldout_uses_only_the_new_sites():
    assert set(stackexchange_heldout.SITES).isdisjoint(
        {"cooking", "gardening", "money", "parenting", "pets", "outdoors", "fitness", "interpersonal"})
    assert len(stackexchange_heldout.SITES) == 8
    assert stackexchange_heldout.SPEC.name == "stackexchange_heldout"
```

Add a LEDE test in the same style once Step 1 has named the columns. It builds a two-row CSV per model
file, and asserts:
- every sample is `Label.AI` with `Role.CORE`;
- the generator comes from the file name (`Claude_Opus_4.6_Eng.csv` gives `"Claude Opus 4.6"` in family
  `anthropic`);
- only the frontier files load: Claude 4.5–4.6, GPT-5.x and Gemini 3.1.

- [ ] **Step 3: Run the tests to see them fail.** Run `python -m pytest tests/test_sources_v4.py -q`.
Expected: ImportError.

- [ ] **Step 4: Implement `wasitai2026.py`**

```python
"""WasItAIGenerated AI-detector benchmark 2026-09: human text and answers from 2026 frontier models (GPT-6,
Claude Opus 5.5, Gemini 3.x, Grok 4.7, DeepSeek V4, Qwen 3.8) plus commercially humanized answers.

Held out whole (split.HOLDOUT_SOURCES) as v4's frontier test. Only English rows whose text is in the
file are used: rows the publisher could not redistribute are never re-fetched from their URLs, since
the page may have changed. Humanized rows are a report-only stress slice; the seven AI-edited rows
need a labelling policy and are skipped. Licences are recorded per row.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, compact, make_sample

_FILE = "data/benchmark.parquet"
SPEC = SourceSpec(
    name="wasitai2026",
    licence="mixed per row (dataset card: other)",
    licence_class=LicenceClass.UNKNOWN,
    files=((f"https://huggingface.co/datasets/wasitaigeneratedcom/ai-detector-benchmark-2026-09/resolve/main/{_FILE}",
            _FILE),),
)


def row_licence(name: str) -> tuple:
    lower = (name or "").lower()
    if "nc" in lower.split("-"):
        return name, LicenceClass.NON_COMMERCIAL
    if lower.startswith(("cc-by-4.0", "mit", "apache")):
        return name, LicenceClass.PERMISSIVE
    return name or "unknown", LicenceClass.UNKNOWN


def load(raw: Path, cap: int) -> Iterable[Sample]:
    df = pd.read_parquet(raw / _FILE)
    df = df[(df["lang"] == "en") & df["text"].notna() & df["label"].isin(["human", "ai", "humanized"])]
    for r in df.itertuples():
        label = Label.HUMAN if r.label == "human" else Label.AI
        role = Role.STRESS if r.label == "humanized" else Role.CORE
        yield from compact([make_sample(
            SPEC, text=r.text, label=label, role=role, record_id=str(r.id), domain=str(r.source),
            group_key=str(r.group), generator=r.generator if label is Label.AI else None,
            transform=f"humanizer:{r.humanizer}" if role is Role.STRESS else "none",
            verification="dataset label (publisher)", licence=row_licence(r.text_license))])
```

- [ ] **Step 5: Implement `stackexchange_heldout.py`**

```python
"""Eight Stack Exchange sites never used before (Increment 4d): the v4 locked test's unseen Q&A.

Same selection as the v3 loader (questions dated 2021 or earlier, best answer with a score of at least 2,
no code blocks, at most three links), paired with answers from generators no model trains on (Phi-4-mini,
Granite 4, Falcon 3). Held out whole (split.HOLDOUT_SOURCES).
"""

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..generate import load_generated
from ..schema import Label, LicenceClass, Role, Sample
from . import stackexchange as se
from ._common import SourceSpec, compact, generated_licence, make_sample, strip_markdown, take_per_stratum

SITES = ("travel", "diy", "workplace", "academia", "bicycles", "mechanics", "photo", "writers")
PER_SITE = 375
SPEC = SourceSpec(
    name="stackexchange_heldout",
    licence="CC-BY-SA-4.0",
    licence_class=LicenceClass.COPYLEFT,
    files=tuple((f"{se._BASE}{s}.stackexchange.com/train-00000-of-00001.parquet",
                 f"{s}.stackexchange.com/train-00000-of-00001.parquet") for s in SITES),
)


def pairs(raw: Path) -> pd.DataFrame:
    return se.select_pairs(raw, SITES, PER_SITE)


def load(raw: Path, cap: int) -> Iterable[Sample]:
    yield from se.samples(SPEC, raw, pairs(raw))
```

Refactor `stackexchange.py` so both loaders share the selection and sample code, without changing its
output:
- `select_pairs(raw, sites, per_site)` holds the current body of `pairs()`, using `sites` and `per_site`.
- `pairs(raw)` returns `select_pairs(raw, SITES, PER_SITE)`.
- `samples(spec, raw, items)` holds the current body of `load()`, parameterized by `spec`.
- `load(raw, cap)` returns `samples(SPEC, raw, pairs(raw))`.

Check the refactor preserves output by running the existing `tests/test_generate.py` and
`tests/test_corpus.py` unchanged.

- [ ] **Step 6: Implement `lede.py`**, using the columns from Step 1. Pattern:
  - `_FRONTIER = ("Claude_Opus_4.6", "Claude_Sonnet_4.5", "Claude_Sonnet_4.6", "GPT_5_", "GPT_5.4",
    "Gemini_3.1")`;
  - read each `data/Eng/<model>_Eng.csv` whose name starts with a frontier prefix;
  - generator = the file stem without `_Eng`, with underscores as spaces;
  - `Label.AI`, `Role.CORE`, domain `"news"`, group = the source-article id column;
  - licence `CC-BY-NC-4.0`, `LicenceClass.NON_COMMERCIAL`;
  - `take_per_stratum(df, ["generator"], cap, <id column>)`.

- [ ] **Step 7: Register the sources.** In `sources/__init__.py`, import `wasitai2026, lede, stackexchange_heldout`
and add them to `SOURCES`.

- [ ] **Step 8: Run the tests to see them pass, then run the full suite.** Run
`python -m pytest tests/test_sources_v4.py -q`, then `python -m pytest -q`.

- [ ] **Step 9: Commit** with message `feat: v4 test-only sources (WasItAI 2026-09, LEDE frontier, eight new Stack Exchange sites)`.

---

### Task 6: loaders for the new training sources (MAGA, DACTYL, Arena 140k, OpenTuringBench)

**Files:**
- Create: `src/corpus/sources/maga.py`, `src/corpus/sources/dactyl.py`,
  `src/corpus/sources/arena140k.py`, `src/corpus/sources/openturingbench.py`
- Modify: `src/corpus/sources/__init__.py`
- Test: `tests/test_sources_v4.py`

**Interfaces:**
- Produces: modules with `SPEC` and `load(raw, cap)`. `arena140k.load` and `openturingbench.load` yield
  AI samples only.

- [ ] **Step 1: Confirm the columns** once the downloads finish (`D:/llm-authorship-corpus/download-4d.log`
ends with `all downloads done`).

```bash
python -c "import json; f=open('D:/llm-authorship-corpus/raw/maga/train/MAGA_train.jsonl',encoding='utf-8'); r=json.loads(f.readline()); print({k:type(v).__name__ for k,v in r.items()})"
python -c "import pandas as pd; print(pd.read_parquet('D:/llm-authorship-corpus/raw/arena140k/data/train-00000-of-00007.parquet').dtypes)"
python -c "import json; d=json.load(open('D:/llm-authorship-corpus/raw/openturingbench/in-domain/val.json',encoding='utf-8')); print(type(d), (d[0].keys() if isinstance(d,list) else list(d)[:5]))"
```

Known columns:
- **DACTYL** (confirmed 2026-09-28): `id, text, domain (abstracts|reviews|student_essays|tweets|RedditWritingPrompts|news),
  model ("human" for human rows), category_info, generation_info, target (0 human, 1 AI), prompt, created_at`.
- **MAGA** AI records: `id, title, text, domain, human_source_id, prompt_id, system_prompt, user_prompt,
  model, temperature, top_p, top_k, repetition_penalty`.
- **MAGA human texts:** Step 1 decides where they come from. If `MAGA_train.jsonl` holds human rows
  (model `"human"` or a label field), use them. Otherwise MAGA is AI-only: its human side
  (`anyangsong/MAGA-human-source`) is the raw source corpora, tens of GB, and is not downloaded. In that
  case, record this in the pre-registration and treat MAGA like Arena.

- [ ] **Step 2: Write the failing tests** for each loader, following Task 5's fixture style. Each loader
gets:
- **DACTYL:** `RedditWritingPrompts` rows are dropped (MAGE has WritingPrompts). `target` sets the
  label, `model` sets the generator, and `domain` passes through. Human and AI rows that share a
  `prompt` share a `group_key`. The two `fine-tuned-Llama-3.2-1B` models map to family `meta`.
- **Arena 140k:**
  - Only English rows (the `language` column) are kept, and only the first assistant turn of each side.
  - Each side becomes one AI sample, with the generator from `model_a` / `model_b`.
  - Sides in a family `split.HOLDOUT_FAMILIES` names are dropped.
  - The group is the conversation id, so both sides share one group.
  - The licence is `"prompts CC-BY-4.0; outputs under model providers' terms"`, class unknown.
- **OpenTuringBench:** `in-domain/train.json`, `val.json` and `test.json` load as AI news samples, with
  the generator from the model field and the group from the headline or id field named in Step 1.
  The `in-domain-variations/` and `out-of-distribution/` files are not loaded.
- **MAGA:**
  - `train/` and `val/` records load.
  - The `extra/` configs load only when `maga.INCLUDE_EXTRA` is True (default False; see the spec's
    risks).
  - Records whose `domain` is wikiHow are dropped (DMCA).
  - Group = `human_source_id`.

- [ ] **Step 3: Run the tests to see them fail.** Run `python -m pytest tests/test_sources_v4.py -q`.

- [ ] **Step 4: Implement the four loaders.** Each module docstring states:
  - what the source is;
  - its licence, with the URL of the card that states it;
  - what is dropped and why;
  - its role in v4.

  Each loader caps with `take_per_stratum(df, [<domain>, <label or generator>], cap, <id>)`. Loader
  specifics:
  - **Arena:** read only the needed columns of each parquet shard (`pd.read_parquet(path, columns=[...])`),
    so RAM stays bounded.
  - **Arena first turn:** in `conversation_a` / `conversation_b`, the first message with role
    `assistant`. Its content may be a list of parts; join the text parts.

- [ ] **Step 5: Register, then run the tests to see them pass and run the full suite.**

- [ ] **Step 6: Commit** with message `feat: v4 training sources (MAGA, DACTYL, Arena 140k AI replies, OpenTuringBench AI news)`.

---

### Task 7: probe for AI-only sources

**Files:**
- Modify: `src/corpus/confound.py` (`ai_only_probe`)
- Modify: `src/corpus/build.py` (record it and exclude flagged sources before splitting)
- Test: `tests/test_corpus_v4.py`

**Interfaces:**
- Produces:
  - `confound.ai_only_probe(df, pools: dict[str, tuple[str, ...]]) -> dict[str, float]`: for each
    AI-only source, the shortcut-probe AUC of its AI rows against the pooled human rows of the named
    sources, after the model view.
  - `build.AI_ONLY_POOLS = {"arena140k": ("dolly", "oasst1", "no_robots"), "openturingbench": ("mage", "raid", "dactyl")}`,
    plus MAGA if Task 6 made it AI-only.

- [ ] **Step 1: Write the failing test**

```python
from src.corpus import confound


def test_ai_only_probe_flags_a_layout_giveaway():
    rows = [{"source_dataset": "arena140k", "label": "ai", "split": "val",
             "text": f"**Answer {i}**\n\n- point one\n- point two\n\n## Summary\nDone."} for i in range(200)]
    rows += [{"source_dataset": "dolly", "label": "human", "split": "val",
              "text": f"i think its fine honestly, number {i} worked for me last week"} for i in range(200)]
    import pandas as pd
    auc = confound.ai_only_probe(pd.DataFrame(rows), {"arena140k": ("dolly",)})
    assert auc["arena140k"] > 0.85
```

- [ ] **Step 2: Run it to see it fail.** Expected: `AttributeError: ai_only_probe`.

- [ ] **Step 3: Implement.** Reuse the feature extraction and cross-validated logistic probe already in
`confound.shortcut_probe` on the view `MODEL_TEXT_VIEW`. Refactor the shared part into
`_probe_auc(texts, labels) -> float` if `shortcut_probe` inlines it. In `build()`:
- compute `ai_only_probe` on the collected rows before `assign_splits`;
- drop any source over `confound.WARN_THRESHOLD` (0.85);
- record `{"ai_only_probe": {...}, "ai_only_excluded": [...]}` in the build record.

- [ ] **Step 4: Run the full suite.**

- [ ] **Step 5: Commit** with message `feat: probe AI-only sources against their human register and exclude giveaways`.

---

### Task 8: pre-registration

**Files:**
- Create: `docs/bob/llm-authorship-checker/increment-4d-preregistration.md`
- Modify: `docs/bob/llm-authorship-checker/build-brief.md` (Revision 3: CPU limits)

- [ ] **Step 1: Write the pre-registration.** Use the 4c pre-registration's section order:
  - Why;
  - Generated data;
  - Corpus v4;
  - Models and routing;
  - System under test;
  - Pass criteria (unchanged since v1);
  - The improvement rule and outcome table;
  - Reported, not gating;
  - Known risks.

  Copy the numbers verbatim from the spec. Name:
  - the held-out families (bigscience, microsoft, ibm, tii);
  - the held-out sources (stackexchange_heldout, wasitai2026, lede);
  - the eight new sites;
  - the shared-ground slices;
  - the three goals.

  Say that screening ran on v3 data before this document, and that it reads no locked split.

- [ ] **Step 2: Build brief Revision 3.**
  - In the non-functional requirements table, change the CPU rows' MVT to `≤10 s` (single text) and
    `≤1,800 s` (sheet).
  - Add a line under the approvals: "Revision 3: project owner — 2026-09-28 (CPU limits relaxed to
    10 s per text and 30 minutes per 500-row sheet so model size is set by the GPU; GPU limits
    unchanged)."

- [ ] **Step 3: Commit** with message `docs: Increment 4d pre-registration and build brief Revision 3`.
This commit must land before Task 9 starts.

---

### Task 9: generation variants

**Files:**
- Modify: `src/corpus/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Produces:
  - `generate(source, items, pool, *, variant: str = "", ...)`. It writes to
    `raw/<source>/generated` when `variant == ""`, or to `raw/<source>/generated_<variant>`.
  - `load_generated(raw, variant="")`.
  - `style_of(item_id, variant) -> dict` with keys `persona`, `plain`, `temperature` and `target_words`
    (for short variants).
  - `humanize(source, variant, rewriter_pool, share=0.10)`, which writes `generated_<variant>_humanized`.
  - `HELD_OUT_GENERATORS == ("phi4-mini", "granite4:3b", "falcon3:3b")`.

- [ ] **Step 1: Write the failing tests**

```python
def test_variant_answers_go_to_their_own_folder_with_their_style(tmp_path):
    calls = []
    G.generate("src", _items(), ("m1",), root=tmp_path, limit=4, variant="short4d",
               chat_fn=_fake_chat(calls), info_fn=INFO)
    out = tmp_path / "raw" / "src" / "generated_short4d"
    records = [json.loads(l) for p in out.glob("*.jsonl") for l in p.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert all(25 <= r["target_words"] <= 99 for r in records)
    assert all(set(r["style"]) == {"persona", "plain", "temperature"} for r in records)
    assert all(0.6 <= r["options"]["temperature"] <= 1.1 for r in records)
    assert not (tmp_path / "raw" / "src" / "generated").exists()


def test_style_is_deterministic_and_varied():
    styles = [G.style_of(f"s:{i}", "train4d") for i in range(200)]
    assert styles == [G.style_of(f"s:{i}", "train4d") for i in range(200)]
    assert {s["persona"] for s in styles} == set(G.PERSONAS)
    assert 0.2 < sum(s["plain"] for s in styles) / 200 < 0.5


def test_humanize_rewrites_a_tenth_with_a_different_model(tmp_path):
    calls = []
    G.generate("src", _items(40), ("m1", "m2"), root=tmp_path, variant="train4d",
               chat_fn=_fake_chat(calls), info_fn=INFO)
    n = G.humanize("src", "train4d", ("m1", "m2"), root=tmp_path, chat_fn=_fake_chat(calls), info_fn=INFO)
    rewritten = G.load_generated(tmp_path / "raw" / "src", variant="train4d_humanized")
    assert 1 <= len(rewritten) <= 8 and len(rewritten) == n
    source = G.load_generated(tmp_path / "raw" / "src", variant="train4d").set_index("item_id")
    assert all(r.generator != source.loc[r.item_id, "generator"] for r in rewritten.itertuples())
```

- [ ] **Step 2: Run them to see them fail.**

- [ ] **Step 3: Implement.**
  - `PERSONAS = ("neutral", "casual", "forum", "terse")`, each a system prompt:
    - "You are a helpful assistant.";
    - "You are chatting casually with a friend. Write the way people type, not like a textbook.";
    - "You are a regular user answering a question on a Q&A forum.";
    - "You answer briefly and directly.".
  - `plain` (30% of items) appends "Write plain text: no headings, lists, bold or markdown."
  - Temperature = `0.6 + 0.5 * u`, where `u` comes from the item hash.
  - Short variants (names starting `short`) set `target_words = 25 + (hash % 75)`.
  - Every choice comes from `sha256(f"{variant}:{item_id}")`, so reruns are identical.
  - Each record stores `style` and the effective `options.system`.
  - `pick()` and `seed_of()` take the variant as salt when it is non-empty, and keep today's values for
    `variant=""`, so v3 records stay reproducible.
  - `humanize` picks 10% of a variant's items by hash. It sends each answer to a different generator
    from `rewriter_pool` (the next one in pool order after the original) with: "Rewrite the text below
    so it sounds casual and natural, like a person typed it. Keep the meaning and roughly the same
    length. Reply with the rewrite only.\n\n<text>". Records carry `rewritten_from` (the original
    generator) and `item_id`.
  - `main()`: add `--variant` and `--humanize`, and extend `plans` with:
    - `"stackexchange": TRAINING_GENERATORS` (old sites, variants `train4d` and `short4d`);
    - `"dolly"` and `"oasst1"`: `short4d`;
    - `"stackexchange_heldout": HELD_OUT_GENERATORS` (variants `""` and `short4d`).

    Add `GENERATION_LIMITS_4D` with the pre-registered counts.

- [ ] **Step 4: Run the tests to see them pass, then run the full suite.** Also check the loaders that
call `load_generated` (dolly, oasst1, stackexchange): they read every variant folder. Change them to
concatenate `load_generated(raw, v)` for `v in ("", "train4d", "short4d", "train4d_humanized")`. Tag
humanized rows `transform="llm_rewrite_casual"`.

- [ ] **Step 5: Commit** with message `feat: generation variants (short answers, personas, plain text, temperature range, humanized rewrite)`.

---

### Task 10: generate, build v4, audit

- [ ] **Step 1: Pull the held-out generator.** Start Ollama with models on D:, then run
`ollama pull falcon3:3b`. Confirm it runs on the 4 GB GPU with a one-prompt test.
- [ ] **Step 2: Run generation detached** with `run-job.ps1`, one model at a time:

```
python -m src.corpus.generate --sources stackexchange,dolly,oasst1 --variant short4d
python -m src.corpus.generate --sources stackexchange --variant train4d
python -m src.corpus.generate --sources stackexchange_heldout --variant ""
python -m src.corpus.generate --sources stackexchange_heldout --variant short4d
python -m src.corpus.generate --sources stackexchange --variant train4d --humanize
```

- [ ] **Step 3: Build v4.** Run `python -m src.corpus.build` (all sources) detached. Expected: the audit
passes, and `data/benchmark/v4/build_record.json` and `test.lock.json` are written.
- [ ] **Step 4: Run the independent audit.** Run `python -m src.corpus.audit --version v4`; expected exit
0. Check the build record for:
  - the per-slice human counts, with every gated slice at 300 or more;
  - the shortcut probe, with no source above 0.85;
  - the AI-only probe;
  - the near-duplicate retired count.
- [ ] **Step 5: Commit** `data/benchmark/v4/build_record.json` and `data/benchmark/v4/test.lock.json`, with
message `data: corpus v4 build record and test lock`.
