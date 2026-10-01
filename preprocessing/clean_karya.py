import csv
import unicodedata
from pathlib import Path
csv.field_size_limit(100000000)

INPUT_FILE = Path("data/raw/karya/translation-hi-unr.tsv")
OUTPUT_FILE = Path("data/processed/karya/karya_hi_mundari_clean.csv")

rows_read = 0
malformed_rows = 0
empty_rows = 0
duplicate_rows = 0

seen = set()
clean_rows = []

with INPUT_FILE.open("r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f, delimiter="\t")

    for row in reader:
        rows_read += 1

        # Keep only Hindi-Mundari pairs with exactly 2 columns
        if len(row) != 2:
            malformed_rows += 1
            continue

        hindi, mundari = row

        # Remove unnecessary whitespace
        hindi = hindi.strip()
        mundari = mundari.strip()

        # Remove empty pairs
        if not hindi or not mundari:
            empty_rows += 1
            continue

        # Normalize Unicode
        hindi = unicodedata.normalize("NFC", hindi)
        mundari = unicodedata.normalize("NFC", mundari)

        pair = (hindi, mundari)

        # Remove exact duplicate pairs
        if pair in seen:
            duplicate_rows += 1
            continue

        seen.add(pair)
        clean_rows.append(pair)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Hindi", "Mundari"])
    writer.writerows(clean_rows)

print("Karya cleaning complete")
print("-----------------------")
print("Rows read:", rows_read)
print("Malformed rows removed:", malformed_rows)
print("Empty rows removed:", empty_rows)
print("Duplicate rows removed:", duplicate_rows)
print("Clean pairs:", len(clean_rows))
print("Output:", OUTPUT_FILE)