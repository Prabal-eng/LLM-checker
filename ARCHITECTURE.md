# Architecture & Technical Methodology

This document outlines the engineering principles, architectural patterns, machine learning pipeline, and detection mechanics behind the **Excel AI Text Detection & Structural QA Engine**.

> **Status:** Section 2 describes the **legacy** detector. It is not validated and is gated off; the v2 authorship checker (`src/detection/`) returns `Inconclusive` until a model passes a verified, leakage-free benchmark. The current design is in [docs/bob/llm-authorship-checker/build-brief.md](docs/bob/llm-authorship-checker/build-brief.md).

---

## 1. System Design Goals

1. **High Domain Specificity**: Evaluation spreadsheets contain unique syntactical cadences (comparative winner rationales, criteria checklists, defect descriptions). Generic off-the-shelf detectors fail due to vocabulary domain shift.
2. **Minimal False Accusations**: Human raters who cite local terminology, use informal phrasing, or have natural typographical variation must rarely be labelled Likely AI. The target is a measured upper confidence bound per slice (see the build brief), not a claim of zero.
3. **Multi-Column Row Auditing**: Human evaluators often write rationale across multiple columns (e.g., candidate-specific comments + winner justifications). Every evaluation column must be audited independently, with the worst offender and flagged columns explicitly recorded.
4. **Local Execution & Fast Turnaround**: The engine must evaluate hundreds of rows in seconds on standard CPU cores without incurring API costs or sending sensitive evaluation data to third-party endpoints.
5. **Observability & Human-in-the-Loop Governance**: Automated detection is an aid to human QA teams. The pipeline must generate explainable evidence and provide human decision dropdowns directly inside the output Excel workbook.

---

## 2. Machine Learning & Detection Mechanics

### Feature Engineering Pipeline
The underlying model combines three orthogonal feature representations:

1. **Word-Level N-Grams (1-3)**: Captures key comparative phrases, transition keywords, hedging phrases, and stock LLM expressions (`is the strongest overall`, `takes a balanced approach`, `without becoming overly`).
2. **Character-Level N-Grams (3-5 with Word Boundaries)**: Captures subword stylistic cadences, morphological patterns, and punctuation spacing.
3. **Stylometric Feature Extractor**:
   - Word count and character count distributions.
   - Type-Token Ratio (lexical diversity / vocabulary richness).
   - Sentence length variance and burstiness ratio ($\sigma_{\text{sentence}} / \mu_{\text{sentence}}$).
   - Average syllable count per word (Flesch-Kincaid complexity proxy).
   - Question and punctuation density.

### Model Architecture & Calibration
- **Classifier**: Sparse Regularized Logistic Regression ($C=2.5$, `class_weight="balanced"`).
- **Probability Calibration**: 5-fold cross-validated sigmoid calibration (`CalibratedClassifierCV`) mapping raw classification margins into true probabilistic confidence scores.

### Hybrid Decision Function (legacy)
Rather than relying purely on TF-IDF or purely on heuristics, the engine uses a unified hybrid fusion function:

$$S_{\text{final}} = \text{clip}\Big(\max(P_{\text{ML}}, S_{\text{structural}}) - C_{\text{human}}, 0.01, 0.99\Big)$$

Where:
- **$P_{\text{ML}}$**: The calibrated probability from the multi-domain statistical classifier.
- **$S_{\text{structural}}$**:
  - **Formulaic Comparative Openings** ($+0.75$ baseline): Matches formulaic openings like `Response [A-C] is the strongest overall because`, `Response [A-C] feels the most natural because`, `Response [A-C] is the best of the three because`.
  - **Canned Stock Boilerplate** ($+0.75$ baseline): Matches copy-pasted low-effort evaluator templates like `Excellent generated image based on the user's prompt`, `The prompt follows the user's intent and instruction as requested`.
  - **LLM Evaluation Tropes** ($+0.10$ to $+0.25$ boost): Flags markers such as `takes a balanced approach`, `is appropriately candid`, `balances X with Y while maintaining Z`, `constructive pushback`, `delve`, `testament to`.
  - **Low Burstiness** ($+0.15$ boost): Evaluates if multiple sentences exhibit machine-like uniform length ($CV < 0.22$).
- **$C_{\text{human}}$**:
  - **Authentic Human Markers** ($-0.40$ per signal): Detects native language code-switching (Devanagari scripts), authentic evaluator typos (`did math right`, `weird time wrap`, `becasue`), and conversational informal slang (`idk`, `lol`, `didn't work`).

---

## 3. Intelligent Column Discovery

Spreadsheets from different clients and teams arrive with varying schemas. The `ColumnDiscovery` engine automatically determines roles:

- **Rater Column**: Discovered via keywords `["task assignee", "rater_id", "rater", "annotator", "evaluator"]`.
- **Text & Rationale Columns**: Discovered by inspecting column names for evaluation keywords (`"rationale"`, `"justification"`, `"specific comment"`), filtering out administrative/QC columns (`"qc"`, `"l2"`, `"for rater"`, `"reviewer"`), and verifying that columns contain substantial text (`mean length > 15`).
- **Verdict Column**: Discovered via verdict keywords (`["overall_assessment", "verdict", "score", "rating", "grade", "decision"]`).
- **Row / Item ID Column**: Discovered via `["prompt_id", "row_id", "task_id", "item_id"]`.

---

## 4. Multi-Tab Excel Reporting Format

Every audit run produces a 4-tab workbook styled with the team's visual guidelines:

1. **Summary Dashboard**:
   - Executive KPIs (Total Rows, AI Flagged Count, Structural Errors, Average Justification Quality).
   - Per-Rater Breakdown Table detailing each evaluator's row volume, flagged count, structural issue rate, and average AI score.
2. **AI Review**:
   - Focuses on flagged rows and rows with suspicious proxy signals.
   - Lists rater ID, likelihood score, specific flagged column(s), and justification excerpt.
   - Features an interactive **Data Validation Dropdown** for QA managers:
     - `Approved (Human)`
     - `Confirmed AI`
     - `False Positive`
     - `Needs Discussion`
3. **Structural Issues**:
   - Itemized log of every row violation with exact Excel row number, column name, issue type (`MISSING_REQUIRED_VALUE`, `INVALID_VERDICT`, `DUPLICATE_ROW`, `MISSING_ERROR_EXPLANATION`), and severity.
4. **Annotated Master**:
   - The original spreadsheet enriched with `QA_AI_Score`, `QA_AI_Flagged`, `QA_Quality_Score`, `QA_Quality_Flags`, and `QA_Structural_Issues`.

---

## 5. Reliability & Security Hardening

- **Upload Guard**: Rejects files exceeding `MAX_UPLOAD_MB` (default 50MB) with `HTTP 413` before memory allocation.
- **SHA256 Caching**: Hashes evaluation text and configuration parameters to return cached inference instantly on duplicate content.
- **Transient Error Retries**: External API clients implement 3-phase exponential backoff handling rate limits (HTTP 429) and network timeouts.
- **Strict Typing & Validation**: Domain models defined via Pydantic v2 and Python 3.10+ dataclasses.
