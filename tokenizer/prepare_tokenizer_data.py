import pandas as pd
from pathlib import Path

INPUT_FILE = Path(
    "data/processed/train/train.csv"
)

OUTPUT_FILE = Path(
    "tokenizer/train_text.txt"
)

df = pd.read_csv(INPUT_FILE)

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(str(row["Hindi"]).strip() + "\n")
        f.write(str(row["Mundari"]).strip() + "\n")

print("Tokenizer training corpus created")
print("----------------------------------")
print("Training pairs:", len(df))
print("Training sentences:", len(df) * 2)
print("Output:", OUTPUT_FILE)