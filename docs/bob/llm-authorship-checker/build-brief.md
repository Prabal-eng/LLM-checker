# LLM Authorship Checker Build Brief

**Revision 2 — 2026-09-26: scope widened from evaluator comments to general-purpose text.** Revision 1 targeted spreadsheet justifications and pasted responses from the same domain. The sponsor clarified that the checker must detect AI-generated text in general, not only short A/B comparison rationales. Requirements FR1–FR4 and Increment 1 are unchanged.

**Problem:** Reviewers need one reliable checker for any English text, whether pasted as a standalone response or held in spreadsheet cells. The current system returns a binary flag from a model trained mostly on repeated hand-written templates from two evaluation projects, so it can neither support a defensible authorship decision nor generalize beyond those projects. The new capability must return `Likely AI-generated`, `Likely human-written`, or `Inconclusive`, preserve the evidence behind the decision, and abstain whenever the input is outside the tested slices or the detector is unreliable.  
**Scale:** Medium (revised from Small; see note under Capacity)  
**Build type:** component-in-platform  
**Sponsor:** project owner  
**Hard deadlines:** none stated

**Inherit / Extend / Build:** CI/CD Extend: keep the CI added in the skeleton; add the dataset audit and GPU-free test path · Deployment Extend: keep the FastAPI/CLI runtime and non-root container; add an optional GPU runtime · Observability Extend: keep structured logs, health, and decision counters · Testing Extend: keep pytest and add contract, policy, dataset-audit, benchmark, and end-to-end tests · Security/auth Extend: preserve local-first operation and add input/retention controls before shared deployment · Data layer Build: create a provenance-tracked, licence-tagged, anonymized dataset from public and in-house sources with an immutable benchmark manifest · Compute Build: local GPU inference (4 GB VRAM) with CPU fallback; the host has no model-serving runtime · Conventions Inherit: Python, Pydantic, FastAPI, pandas, openpyxl

**Capacity:** `Available h/week = 1 × 40 × 0.8 × 0.85 × 0.75 = 20.4` · `Timeline (weeks) = 154 Est h ÷ 20.4 × 1.25 = 9.4`  
This is an engineering estimate. The 9.4-week timeline exceeds the Medium calendar band; the phases remain collapsed in this single brief for continuity with Revision 1, and every Phase 0–II gate item is still checked below.

## Scope decisions (sponsor, 2026-09-26)

| Decision | Choice | Future scope kept open |
|----------|--------|------------------------|
| Text coverage | General-purpose text; evaluator comments are one slice among many | — |
| Languages | English only at first release | Language is a slice attribute everywhere; adding a language means adding data, a slice, and calibration, never changing the contract |
| Inference location | Local only; no text leaves the machine at check time | External detectors as optional evidence sources (FR9) |
| Dataset licences | Any licence, including non-commercial, with the licence recorded per sample | A single configuration switch excludes non-commercial data if legal review requires it |
| Hardware | One local GPU with 4 GB VRAM; CPU fallback | Larger scoring models if hardware grows |

## Requirements

| ID | Requirement | Tier | Test |
|----|-------------|------|------|
| FR1 | Classify standalone text into exactly three outcomes with stable reason codes | MVT | API contract and policy unit tests |
| FR2 | Apply the same decision contract to every selected spreadsheet text cell | MVT | End-to-end workbook test covering all three outcomes |
| FR3 | Return `Inconclusive` for insufficient text, unsupported input, detector failure, or scores inside the abstention band | MVT | Boundary and failure-path tests |
| FR4 | Attach observed evidence, detector version, policy version, and score validity to every result | MVT | Response-schema assertions |
| FR5 | Produce leakage-free train, validation, and locked test manifests from public and in-house sources, where every sample records its source dataset, licence, verification method, generator (if AI), domain, and language | MVT | Dataset audit fails on duplicate or near-duplicate groups crossing splits, missing provenance or licence fields, or PII |
| FR6 | Select low/high thresholds from validation data and never from the locked test set | MVT | Reproducible calibration command and immutable test manifest |
| FR7 | Expose pasted-text checking in the web UI | Target | Browser/API end-to-end test |
| FR8 | Report coverage, false-positive bounds, false-human bounds, calibration, and slices by domain, language, length, generator family, and writer group (including non-native English) | Target | Generated evaluation report |
| FR9 | Support optional external detectors as evidence sources whose failure causes abstention | Target | Simulated outage and disagreement tests |
| FR10 | Detect drift and support reviewer feedback without contaminating the locked test set | Stretch | Drift report and feedback-store tests |
| FR11 | Return `Inconclusive` with a stable reason code for text outside the supported slices: unsupported language, or a domain the locked benchmark does not cover | MVT | Language and out-of-domain boundary tests |
| FR12 | Report robustness on stress sets: paraphrased AI text, human-edited AI text, generators absent from training, and non-native English human writing | Target | Stress-set section of the evaluation report |
| FR13 | Exclude samples by licence class through configuration without code changes | Target | Dataset build test with non-commercial exclusion enabled |

## Non-functional requirements

| Metric | MVT | Target | Stretch |
|--------|-----|--------|---------|
| Upper 95% confidence bound for human text labelled Likely AI, per supported major slice | ≤1.0% | ≤0.5% | ≤0.25% |
| Upper 95% confidence bound for AI text labelled Likely human, per supported major slice | ≤10% | ≤5% | ≤2.5% |
| Decisive coverage on supported text | ≥40% | ≥50% | ≥60% |
| API p95 latency, local detector, single text of ≤500 words, on the 4 GB GPU | ≤500 ms | ≤400 ms | ≤333 ms |
| Peak GPU memory during inference | ≤3.5 GB | ≤3.0 GB | ≤2.5 GB |
| API p95 latency, CPU only, single text of ≤500 words | ≤20 s | ≤10 s | ≤5 s |
| Spreadsheet of 500 rows × 6 text columns, end to end including model load and report write, on the 4 GB GPU | ≤120 s | ≤90 s | ≤60 s |
| Same spreadsheet, CPU only | reported, not gated (Revision 4) | ≤1,800 s | ≤900 s |
| Test suite passes on a machine with no GPU or CUDA installed | 100% | 100% | 100% |
| Unhandled detector failures presented as a decisive label | 0 | 0 | 0 |
| Exact or near-duplicate groups crossing data splits | 0 | 0 | 0 |
| Raw personal identifiers in model datasets | 0 | 0 | 0 |
| Samples without a recorded licence | 0 | 0 | 0 |

## Constraints

- Text alone does not provide proof of authorship; decisive labels are calibrated risk assessments.
- Short, edited, translated, mixed-authorship, non-English, and unseen-domain text may require abstention.
- Inference is local only and must fit 4 GB of VRAM; this excludes full-precision 7B-parameter scoring models.
- Colleagues without a GPU must be able to run the tests and check single texts on the CPU within the CPU target; a candidate model that misses it is rejected regardless of accuracy. Spreadsheets on the CPU are a slow fallback (Revision 4).
- Public datasets carry known confounds (human text older than AI text, topic mismatch, outdated generators); these must be measured, not assumed away.
- Existing spreadsheet loading, column discovery, report generation, FastAPI, CLI, and pytest are inherited unless they fail a requirement.
- The current model and benchmark remain available only as a legacy baseline; their published accuracy is not a release gate.

## Unknowns

| ID | Unknown | Impact | Route |
|----|---------|--------|-------|
| U1 | Can available data support the target false-positive bound per supported slice? | 5 | Increment 2: dataset audit and locked benchmark |
| U2 | Which model family generalizes best to unseen generators, domains, and writers within 4 GB VRAM? | 5 | Increment 3: baseline bake-off |
| U3 | What minimum length supports useful decisive coverage? | 3 | Increment 3: length-sliced evaluation |
| U4 | Can one calibrated policy serve all supported slices, or does each need its own thresholds? | 5 | Increment 4: compare shared versus per-slice calibration |
| U5 | Which evidence is stable enough to show users without overstating causality? | 3 | Increment 4: evidence ablation and review |
| U6 | What production deployment target will host the service? | 3 | Skeleton: sponsor selects target; deployment rehearsal closes it |
| U7 | Do public-dataset confounds (date, topic, source) let a model score well without detecting authorship? | 5 | Increment 2: confound check that trains a shortcut probe on metadata-only features; Increment 3 reports slice gaps |
| U8 | Where does verified human text for the evaluator-comment slice come from? | 3 | Increment 2: presumed-human rater text with an audited sample until a verified source is named |

## Design sketch

Both the standalone endpoint and spreadsheet pipeline call one versioned detection service. A support gate checks language and domain coverage before any scoring. A detector emits raw scores, availability, and observed indicators. A separate decision policy checks input sufficiency and support, then applies independently calibrated low/high thresholds. The API, CLI, workbook, and web UI render the resulting decision without recomputing it. Dataset ingestion, auditing, calibration, and locked evaluation are offline commands whose artifacts carry hashes, licences, and versions. A model registry records which detector version passed which benchmark and replaces the hard-coded validation gate.

## ADRs

- **ADR-001 — Three-way decision contract:** Use an enum plus reason codes, nullable score, detector status, detector version, and policy version. Revisit only if verified provenance signals justify a separate `Confirmed AI` outcome.
- **ADR-002 — Separate detector from decision policy:** The model produces evidence and a score; the policy owns abstention and thresholds. Revisit if a future model natively supports selective classification and passes the same gates.
- **ADR-003 — Reuse the host application:** Extend FastAPI, the existing pipeline, and Excel reporting. Revisit if load or isolation requirements exceed a single process.
- **ADR-004 — Superseded (Revision 2).** "No new model dependency yet" cannot hold for general-purpose detection: installed scikit-learn n-gram models memorize domain vocabulary. Replaced by ADR-009.
- **ADR-005 — Containerized local deployment:** Use the host's installed Docker runtime and a non-root image as the first production-like target. Revisit when a shared hosting environment is selected.
- **ADR-006 — Licence-tagged datasets (light):** Accept any dataset licence; record licence and licence class per sample; exclude a class by configuration (FR13). Revisit if legal review restricts use.
- **ADR-007 — Local-only inference (light):** No text is sent to external services at check time. External detectors stay behind FR9 and are disabled by default. Revisit when the sponsor approves an external provider.
- **ADR-008 — English first, language as a slice (light):** Detect language before scoring; non-English text returns `Inconclusive` with `UNSUPPORTED_LANGUAGE`. Adding a language adds data, a slice, and calibration without contract changes.
- **ADR-009 — Model family chosen by locked bake-off (one-way door, decided 2026-09-26: fine-tuned DistilRoBERTa, see [model-selection-2026-09-26.md](model-selection-2026-09-26.md)):** Candidates must fit 4 GB VRAM: (a) zero-shot likelihood-based scoring with small open language models, (b) a fine-tuned small transformer classifier, (c) the legacy n-gram model as baseline. The full weighted decision matrix is completed from Increment 3 validation results before any candidate is registered. Adding PyTorch and Hugging Face Transformers is accepted as the dependency cost for candidates (a) and (b): rungs 2–5 of the build ladder fail because no installed dependency runs neural language models.

- **ADR-010 — pyarrow for corpus storage (light):** Parquet is needed to read RAID and Beemo and to store a corpus of several hundred thousand samples with typed columns. Rungs 2–5 fail: pandas cannot read or write Parquet without an engine, and CSV loses types and is several times larger. Revisit if the corpus moves to a database.

## Phase 0–II gate

- [x] Problem, users, scale, inherited context, capacity, and timeline documented
- [x] Every requirement has a concrete test
- [x] Metrics have MVT, Target, and Stretch values
- [x] Every impact-5 unknown is mapped to an early increment
- [x] Components, interfaces, data flow, critical path, and significant decisions documented
- [x] Roadmap is dependency ordered and risk scored
- [x] Debt Ledger exists

Approval: project owner — 2026-09-26 (Revision 1: instruction to proceed and prioritize final reliability). Revision 2: project owner — 2026-09-26 (approved general-purpose scope, English first, local-only inference, any licence tagged, 4 GB GPU with CPU fallback, and the CPU and spreadsheet throughput targets). Revision 3: project owner — 2026-09-28 (CPU limits relaxed to 10 s per text and 30 minutes per 500-row sheet, so model size is set by the GPU; GPU limits unchanged; Increment 4d). Revision 4: project owner — 2026-09-28 (GPU-first, chosen after a CPU probe showed DeBERTa-v3-large at about 11 s per 256-token text and hours per sheet on the CPU: CPU single text ≤ 20 s; the CPU spreadsheet limit dropped to reported-only; GPU limits unchanged; the app scores sheets in batched calls so the GPU sheet limit holds; Increment 4d).
