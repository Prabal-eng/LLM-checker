import sys
import io
import pandas as pd
from src.detectors.trained_detector import TrainedModelDetector

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

file_path = r'C:\Users\prabd\Downloads\Internal [545293106] - I18n Personality RM Guidelines Translocalize Human Eval] – Anonymised_candor - Replication 1.xlsx'
df = pd.read_excel(file_path, sheet_name='hi_IN')
row2 = df.iloc[0]

mandatory_cols = [c for c in df.columns if 'MANDATORY' in str(c).upper()]
detector = TrainedModelDetector()

print("================ MODEL SCORES FOR ROW 2 ================")
for col in mandatory_cols:
    text = str(row2[col])
    res = detector.detect(text)
    print(f"\nColumn   : {col}")
    print(f"AI Score : {res.ai_score * 100:.2f}%")
    print(f"Flagged? : {res.is_flagged} (Threshold = {detector.threshold * 100:.0f}%)")
    print(f"Analysis : {res.analysis}")
