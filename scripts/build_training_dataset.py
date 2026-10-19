"""Builds balanced multi-domain in-domain training, validation, and test datasets.

Covers:
1. Multi-candidate Chatbot / Personality evaluation rationales (e.g. Internal 545293106).
2. Multimodal Image Critique justifications (e.g. GemPix Rep 1, 2, 3).
3. In-the-wild human annotations and authentic LLM rater outputs.
"""
import glob
import os
import random
import re
from typing import List, Dict
import pandas as pd

random.seed(42)

DOWNLOADS_DIR = r"C:\Users\prabd\Downloads"
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(WORKSPACE_DIR, "data", "training")
os.makedirs(OUT_DIR, exist_ok=True)

GEMPIX_RATIONALE_COLS = [
    "overall_assessment_rationale",
    "image_factuality_rationale",
    "design_clarity_quality_rationale",
    "i18n_issues_rationale",
    "prompt_alignment_rationale",
    "usability_rationale"
]

PERSONALITY_COLS = [
    "Response A Specific Comment (MANDATORY)",
    "Response B Specific Comment (MANDATORY)",
    "Response C Specific Comment (MANDATORY)",
    "Winner Justification (MANDATORY)"
]


def extract_gempix_samples() -> List[Dict]:
    """Extracts authentic human justifications from GemPix Rep 1, 2, and 3."""
    samples = []
    seen = set()

    file_patterns = [
        os.path.join(DOWNLOADS_DIR, "*GemPix*Replication 1*.xlsx"),
        os.path.join(DOWNLOADS_DIR, "*GemPix*Replication 2*.xlsx"),
        os.path.join(DOWNLOADS_DIR, "*GemPix*Replication 3*.xlsx"),
    ]

    files = []
    for pat in file_patterns:
        files.extend(glob.glob(pat))

    source_files = [f for f in files if "audited" not in os.path.basename(f)]

    for fpath in source_files:
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_excel(fpath, sheet_name="en_US")
            df = df[df["Task Assignee"].notna() & (df["Task Assignee"].astype(str).str.strip() != "")]

            for _, row in df.iterrows():
                rater = str(row["Task Assignee"]).strip()
                if "pierre.alston" in rater:
                    continue

                for col in GEMPIX_RATIONALE_COLS:
                    if col in row and pd.notna(row[col]):
                        text = str(row[col]).strip()
                        if len(text.split()) >= 4 and text not in seen:
                            seen.add(text)
                            samples.append({
                                "text": text,
                                "label": 0,
                                "source": "human_gempix",
                                "column": col,
                                "rater": rater
                            })
        except Exception as e:
            print(f"Notice reading {os.path.basename(fpath)}: {e}")

    print(f"Extracted {len(samples)} GemPix human justifications.")
    return samples


def extract_personality_samples() -> tuple[List[Dict], List[Dict]]:
    """Extracts authentic human and real AI copypasta from personality comparative evaluations."""
    human_samples = []
    ai_samples = []
    seen = set()

    files = glob.glob(os.path.join(WORKSPACE_DIR, "*Personality*.xlsx"))
    if not files:
        files = glob.glob(os.path.join(DOWNLOADS_DIR, "*Personality*.xlsx"))
    source_files = [f for f in files if "audited" not in os.path.basename(f)]

    if not source_files:
        return human_samples, ai_samples

    formulaic_pattern = re.compile(
        r"^(response [a-c] is (the strongest|overall better|preferred|the best|the most grounded|much better)|"
        r"the response [a-c]? is balanced|[a-c] is the strongest overall because|"
        r"response [a-c] stood better|response [a-c] gives the most grounded feedback)",
        re.I
    )

    for fpath in source_files:
        try:
            df = pd.read_excel(fpath, sheet_name="hi_IN")
            for _, row in df.iterrows():
                rater = str(row.get("Task Assignee", "")).strip()
                if not rater or rater.lower() == "nan":
                    continue

                for col in PERSONALITY_COLS:
                    if col in row and pd.notna(row[col]):
                        text = str(row[col]).strip()
                        if len(text.split()) < 4 or text in seen:
                            continue
                        seen.add(text)

                        is_known_ai_rater = any(k in rater for k in ["shalini.pandey", "asim.khan"])
                        is_formulaic = bool(formulaic_pattern.search(text))

                        if is_known_ai_rater and is_formulaic:
                            ai_samples.append({
                                "text": text,
                                "label": 1,
                                "source": "real_ai_copypasta",
                                "column": col,
                                "rater": rater
                            })
                        elif not is_formulaic and not is_known_ai_rater:
                            human_samples.append({
                                "text": text,
                                "label": 0,
                                "source": "human_personality",
                                "column": col,
                                "rater": rater
                            })
        except Exception as e:
            print(f"Notice reading personality file {os.path.basename(fpath)}: {e}")

    print(f"Extracted {len(human_samples)} personality human justifications and {len(ai_samples)} real AI copypastas.")
    return human_samples, ai_samples


COMPARATIVE_WINNER_TEMPLATES = [
    "Response {winner} is the strongest overall because it directly addresses the user's argument, {praise_clause} while {balance_clause}.",
    "Response {winner} is the strongest because it gives clear, constructive pushback and {praise_clause} while maintaining a natural, friendly tone.",
    "Response {winner} is preferred as it takes a balanced approach, acknowledging {aspect} without {defect_clause}.",
    "Response {winner} is appropriately candid and offers practical alternatives, whereas Response {loser} {loser_defect}.",
    "Response {winner} provides the most grounded feedback, balancing {trait1} with {trait2} and avoiding {defect_clause}.",
    "In evaluating the candidate responses, Response {winner} emerges as the superior option because it {praise_clause} and provides actionable recommendations.",
    "While Response {loser} is well-meaning, Response {winner} strikes the ideal balance between empathy and candor.",
    "Response {winner} is the most balanced response. It validates the user's feelings, gives practical advice, and avoids {defect_clause}.",
    "Response {winner} stood better overall because it logically evaluated the scenario from both angles and offers realistic guidance.",
    "Response {winner} is the best response because it stays focused on the prompt, balances encouragement with practical considerations, and avoids {loser_defect}."
]

COMPARATIVE_COMMENT_TEMPLATES = [
    "The response is balanced and practical, addressing {aspect} without being overly harsh or judgmental.",
    "The response acknowledges the user's premise with clarity, then offers constructive feedback on {aspect}.",
    "The response takes an empathetic and measured approach, validating the user's concern while maintaining objective boundaries.",
    "The response is helpful and structured, though it leans slightly on unsupported generalizations regarding {aspect}.",
    "The response is appropriately candid, pointing out potential drawbacks while offering feasible and safer alternatives."
]

PRAISE_CLAUSES = [
    "offers nuanced and practical counterpoints",
    "balances encouragement with pragmatic reality",
    "delivers clear and actionable guidance",
    "maintains a respectful yet honest tone",
    "provides specific, grounded examples",
    "addresses the core constraints seamlessly",
    "demonstrates high contextual awareness"
]

BALANCE_CLAUSES = [
    "maintaining a balanced and constructive tone throughout",
    "avoiding overly prescriptive or alarmist conclusions",
    "ensuring the feedback remains realistic and feasible",
    "preserving the user's agency while delivering necessary truth",
    "keeping the guidance clear, concise, and pedagogical"
]

DEFECT_CLAUSES = [
    "becoming overly confident or making unsubstantiated claims",
    "relying on excessive praise that ignores the underlying challenges",
    "introducing unnecessary location-specific or subjective assumptions",
    "adopting an overly harsh or dismissive demeanor",
    "overcomplicating the advice with tangential details"
]

LOSER_DEFECTS = [
    "makes several specific research claims without proper grounding",
    "leans heavily on generic praise without actionable suggestions",
    "comes across as unnecessarily alarmist and judgmental",
    "fails to provide concrete steps or practical solutions",
    "stretches the original premise into an unrealistic scenario"
]

ASPECTS = [
    "the core dilemma",
    "the practical trade-offs",
    "the user's emotional state",
    "the technical limitations",
    "the underlying risk factors",
    "the financial and operational considerations"
]

TRAITS_1 = ["empathy", "directness", "candor", "validation", "practicality"]
TRAITS_2 = ["constructive pushback", "clear boundaries", "objective reality", "pragmatic solutions", "tactful honesty"]

IMAGE_CRITIQUE_INTRO_TEMPLATES = [
    "Upon close inspection, the generated visual effectively encapsulates the user's intent, though {defect_clause}.",
    "Furthermore, it is crucial to note that the image meticulously adheres to prompt specifications, albeit {defect_clause}.",
    "The graphic provides a comprehensive and visually coherent representation of the prompt, yet {defect_clause}.",
    "Overall, the generated asset demonstrates commendable alignment with instructions, although {defect_clause}.",
    "In evaluating this output, it serves as a testament to strong conceptual rendering; however, {defect_clause}.",
    "The infographic exhibits robust visual hierarchy and clear aesthetic execution, notwithstanding {defect_clause}."
]

IMAGE_CRITIQUE_PERFECT_TEMPLATES = [
    "The generated image adheres flawlessly to all specified constraints. Typography, structural balance, and color harmony are executed with precision.",
    "Upon comprehensive review, the output seamlessly fulfills the user prompt with exceptional fidelity and zero factual inaccuracies.",
    "This infographic is robust, well-structured, and ready for immediate deployment without requiring any visual or editorial revisions.",
    "In conclusion, the rendering exhibits exemplary design clarity and accurate thematic grounding, fully satisfying all rubric dimensions."
]

IMAGE_DEFECT_CLAUSES = [
    "minor discrepancies in typography and label alignment diminish overall readability",
    "certain factual nuances within the secondary diagrams require editorial correction",
    "the font scaling across subheadings lacks structural consistency",
    "textual clarity in the lower caption area is slightly compromised by compression artifacts",
    "the color contrast between the background gradient and foreground text could be improved for optimal accessibility"
]


def generate_synthetic_ai_samples(target_count: int) -> List[Dict]:
    """Generates synthetic AI samples spanning chatbot comparisons and multimodal image evaluations."""
    samples = []
    candidates = ["A", "B", "C"]

    for _ in range(target_count):
        choice = random.random()
        winner = random.choice(candidates)
        losers = [c for c in candidates if c != winner]
        loser = random.choice(losers)

        if choice < 0.45:
            tmpl = random.choice(COMPARATIVE_WINNER_TEMPLATES)
            text = tmpl.format(
                winner=winner,
                loser=loser,
                praise_clause=random.choice(PRAISE_CLAUSES),
                balance_clause=random.choice(BALANCE_CLAUSES),
                defect_clause=random.choice(DEFECT_CLAUSES),
                loser_defect=random.choice(LOSER_DEFECTS),
                aspect=random.choice(ASPECTS),
                trait1=random.choice(TRAITS_1),
                trait2=random.choice(TRAITS_2)
            )
        elif choice < 0.65:
            tmpl = random.choice(COMPARATIVE_COMMENT_TEMPLATES)
            text = tmpl.format(aspect=random.choice(ASPECTS))
        elif choice < 0.85:
            tmpl = random.choice(IMAGE_CRITIQUE_INTRO_TEMPLATES)
            defect = random.choice(IMAGE_DEFECT_CLAUSES)
            text = tmpl.format(defect_clause=defect)
        else:
            text = random.choice(IMAGE_CRITIQUE_PERFECT_TEMPLATES)

        samples.append({
            "text": text,
            "label": 1,
            "source": "synthetic_ai",
            "column": "eval_text",
            "rater": "synthetic_llm"
        })

    print(f"Generated {len(samples)} synthetic AI justifications.")
    return samples


def main():
    print("[1/3] Extracting authentic data from replication spreadsheets...")
    gempix_human = extract_gempix_samples()
    pers_human, pers_ai = extract_personality_samples()

    all_human = gempix_human + pers_human
    all_ai = pers_ai

    print(f"Extracted Total Human Samples: {len(all_human)}")
    print(f"Extracted Real In-The-Wild AI Samples: {len(all_ai)}")

    target_ai_count = max(len(all_human) - len(all_ai), 500)
    print(f"[2/3] Generating {target_ai_count} synthetic multi-domain AI samples...")
    synthetic_ai = generate_synthetic_ai_samples(target_ai_count)

    full_ai = all_ai + synthetic_ai
    target_total_each = min(len(all_human), len(full_ai))
    random.shuffle(all_human)
    random.shuffle(full_ai)

    final_human = all_human[:target_total_each]
    final_ai = full_ai[:target_total_each]

    dataset = final_human + final_ai
    random.shuffle(dataset)

    df_full = pd.DataFrame(dataset)
    df_full.to_csv(os.path.join(OUT_DIR, "dataset_full.csv"), index=False)

    n = len(df_full)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    df_train = df_full.iloc[:n_train]
    df_val = df_full.iloc[n_train:n_train + n_val]
    df_test = df_full.iloc[n_train + n_val:]

    df_train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    df_test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    print("\n[SUCCESS] Multi-Domain Training Dataset Built Successfully!")
    print(f"  Total Samples : {n} (50% Human, 50% AI)")
    print(f"  Train Set     : {len(df_train)} rows -> data/training/train.csv")
    print(f"  Val Set       : {len(df_val)} rows -> data/training/val.csv")
    print(f"  Test Set      : {len(df_test)} rows -> data/training/test.csv")


if __name__ == "__main__":
    main()
