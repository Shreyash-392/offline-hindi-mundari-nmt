import pandas as pd
import unicodedata
from pathlib import Path

INPUT_FILE = Path("data/raw/adibhashaa/mundari-train.csv")
OUTPUT_FILE = Path("data/processed/adibhashaa/adibhashaa_hi_mundari_clean.csv")

df = pd.read_csv(INPUT_FILE)

rows_read = len(df)

# Keep only the translation columns
df = df[["Hindi", "Mundari"]].copy()

# Remove leading/trailing whitespace
df["Hindi"] = df["Hindi"].astype(str).str.strip()
df["Mundari"] = df["Mundari"].astype(str).str.strip()

# Remove empty values
before_empty_removal = len(df)
df = df[(df["Hindi"] != "") & (df["Mundari"] != "")]
empty_rows_removed = before_empty_removal - len(df)

# Unicode NFC normalization
df["Hindi"] = df["Hindi"].apply(
    lambda x: unicodedata.normalize("NFC", x)
)

df["Mundari"] = df["Mundari"].apply(
    lambda x: unicodedata.normalize("NFC", x)
)

# Remove exact duplicate Hindi-Mundari pairs
before_duplicate_removal = len(df)
df = df.drop_duplicates(subset=["Hindi", "Mundari"])
duplicate_rows_removed = before_duplicate_removal - len(df)

# Create output directory if needed
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Save cleaned dataset
df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("AdiBhashaa-Mundari cleaning complete")
print("------------------------------------")
print("Rows read:", rows_read)
print("Empty rows removed:", empty_rows_removed)
print("Duplicate rows removed:", duplicate_rows_removed)
print("Clean pairs:", len(df))
print("Output:", OUTPUT_FILE)