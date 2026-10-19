"""Generates a realistic sample Excel file containing evaluation records with intentional edge cases."""
import os
import pandas as pd


def generate_sample_dataset() -> pd.DataFrame:
    records = [
        # Normal Human Raters
        {
            "rater_id": "RATER_01",
            "row_id": "TASK_101",
            "category": "Customer Support Response",
            "verdict": "PASS",
            "justification": "The agent answered the billing question correctly and gave the right refund policy link."
        },
        {
            "rater_id": "RATER_01",
            "row_id": "TASK_102",
            "category": "Code Generation",
            "verdict": "FAIL",
            "justification": "Function throws an off-by-one IndexError when passed an empty list on line 14."
        },
        {
            "rater_id": "RATER_02",
            "row_id": "TASK_103",
            "category": "Translation Quality",
            "verdict": "PASS",
            "justification": "French translation is natural, grammar ok, no weird literal translations."
        },

        # Suspected AI-Generated Justifications (Overly polished, stock transitions, buzzwords)
        {
            "rater_id": "RATER_03",
            "row_id": "TASK_104",
            "category": "Summarization",
            "verdict": "PASS",
            "justification": "Furthermore, it is crucial to delve into the comprehensive nature of this summary. It seamlessly encapsulates the primary themes while meticulously maintaining brevity. In conclusion, it serves as a testament to high fidelity."
        },
        {
            "rater_id": "RATER_03",
            "row_id": "TASK_105",
            "category": "Creative Writing",
            "verdict": "PASS",
            "justification": "Moreover, the narrative underscores the pivotal importance of thematic depth. It offers a robust framework that aligns with the expected emotional arc, seamlessly delivering on prompt parameters."
        },

        # Uniform Phrasing Rater (Copypasting formulaic structure across rows)
        {
            "rater_id": "RATER_04",
            "row_id": "TASK_106",
            "category": "Safety Alignment",
            "verdict": "PASS",
            "justification": "Standard review protocol verified: prompt complies with safety boundaries and output satisfies quality baseline."
        },
        {
            "rater_id": "RATER_04",
            "row_id": "TASK_107",
            "category": "Safety Alignment",
            "verdict": "PASS",
            "justification": "Standard review protocol verified: prompt complies with safety boundaries and output satisfies quality baseline."
        },

        # Structural & Quality Errors
        {
            # Blank required field (category is empty)
            "rater_id": "RATER_05",
            "row_id": "TASK_108",
            "category": "",
            "verdict": "PASS",
            "justification": "The response satisfies the prompt criteria."
        },
        {
            # FAIL verdict without justification (Cross-column error)
            "rater_id": "RATER_05",
            "row_id": "TASK_109",
            "category": "Math Reasoning",
            "verdict": "FAIL",
            "justification": ""
        },
        {
            # Invalid verdict enum
            "rater_id": "RATER_06",
            "row_id": "TASK_110",
            "category": "Reasoning",
            "verdict": "PARTIALLY_GOOD",
            "justification": "Calculation was partly right but missed intermediate step."
        },
        {
            # Generic fluff quality issue
            "rater_id": "RATER_06",
            "row_id": "TASK_111",
            "category": "Code Quality",
            "verdict": "PASS",
            "justification": "looks good"
        },
        {
            # Verdict mismatch (Verdict is FAIL but justification says it's clean and verified)
            "rater_id": "RATER_07",
            "row_id": "TASK_112",
            "category": "Factuality",
            "verdict": "FAIL",
            "justification": "Everything is completely accurate, verified, clean, and perfectly matches sources."
        },
        {
            # Duplicate submission (Duplicate of TASK_101 by RATER_01)
            "rater_id": "RATER_01",
            "row_id": "TASK_101",
            "category": "Customer Support Response",
            "verdict": "PASS",
            "justification": "The agent answered the billing question correctly and gave the right refund policy link."
        }
    ]

    return pd.DataFrame(records)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "samples")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rater_evaluations_sample.xlsx")

    df = generate_sample_dataset()
    df.to_excel(out_path, index=False)
    print(f"Sample dataset generated successfully at: {out_path} ({len(df)} rows)")
