import pandas as pd
from pathlib import Path

INPUT_FILE = Path(
    "data/processed/combined_hi_mundari.csv"
)

TRAIN_FILE = Path(
    "data/processed/train/train.csv"
)

DEV_FILE = Path(
    "data/processed/dev/dev.csv"
)

TEST_FILE = Path(
    "data/test/mundari/test.csv"
)

RANDOM_SEED = 42

# Load the combined dataset
df = pd.read_csv(INPUT_FILE)

# Shuffle the dataset deterministically
df = df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

total = len(df)

# Calculate split sizes
train_size = int(total * 0.80)
dev_size = int(total * 0.10)

# Create splits
train = df.iloc[:train_size].copy()
dev = df.iloc[train_size:train_size + dev_size].copy()
test = df.iloc[train_size + dev_size:].copy()

# Create output directories
TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
DEV_FILE.parent.mkdir(parents=True, exist_ok=True)
TEST_FILE.parent.mkdir(parents=True, exist_ok=True)

# Save splits
train.to_csv(TRAIN_FILE, index=False, encoding="utf-8")
dev.to_csv(DEV_FILE, index=False, encoding="utf-8")
test.to_csv(TEST_FILE, index=False, encoding="utf-8")

print("Hindi-Mundari dataset split complete")
print("-------------------------------------")
print("Total pairs:", total)
print("Train pairs:", len(train))
print("Dev pairs:", len(dev))
print("Test pairs:", len(test))
print("Random seed:", RANDOM_SEED)
print("Train output:", TRAIN_FILE)
print("Dev output:", DEV_FILE)
print("Test output:", TEST_FILE)