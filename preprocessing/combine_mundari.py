import pandas as pd
from pathlib import Path

ADIBHASHA_FILE = Path(
    "data/processed/adibhashaa/adibhashaa_hi_mundari_clean.csv"
)

KARYA_FILE = Path(
    "data/processed/karya/karya_hi_mundari_clean.csv"
)

OUTPUT_FILE = Path(
    "data/processed/combined_hi_mundari.csv"
)

# Load cleaned datasets
adibhashaa = pd.read_csv(ADIBHASHA_FILE)
karya = pd.read_csv(KARYA_FILE)

# Record dataset source
adibhashaa["source"] = "AdiBhashaa"
karya["source"] = "Karya"

# Combine datasets
combined = pd.concat(
    [adibhashaa, karya],
    ignore_index=True
)

before_dedup = len(combined)

# Remove exact duplicate Hindi-Mundari pairs
combined = combined.drop_duplicates(
    subset=["Hindi", "Mundari"],
    keep="first"
)

duplicates_removed = before_dedup - len(combined)

# Create output directory if necessary
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Save combined dataset
combined.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("Combined Hindi-Mundari dataset created")
print("---------------------------------------")
print("AdiBhashaa pairs:", len(adibhashaa))
print("Karya pairs:", len(karya))
print("Before deduplication:", before_dedup)
print("Duplicate pairs removed:", duplicates_removed)
print("Final unique pairs:", len(combined))
print("Output:", OUTPUT_FILE)