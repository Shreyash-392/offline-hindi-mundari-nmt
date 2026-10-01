import pandas as pd
import numpy as np
from tokenizer.tokenizer import HindiMundariTokenizer

TRAIN_FILE = "data/processed/train/train.csv"
DEV_FILE = "data/processed/dev/dev.csv"
TEST_FILE = "data/test/mundari/test.csv"

MAX_LENGTH = 80


def analyze_dataset(name, data, tokenizer):
    hindi_lengths = []
    mundari_lengths = []

    for _, row in data.iterrows():
        hindi_ids = tokenizer.sp.encode(str(row["Hindi"]), out_type=int)
        mundari_ids = tokenizer.sp.encode(str(row["Mundari"]), out_type=int)

        # +2 for BOS and EOS
        hindi_lengths.append(len(hindi_ids) + 2)
        mundari_lengths.append(len(mundari_ids) + 2)

    hindi = np.array(hindi_lengths)
    mundari = np.array(mundari_lengths)

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print("Sentences:", len(data))

    print()
    print("HINDI")
    print("Mean:", round(hindi.mean(), 2))
    print("Median:", int(np.median(hindi)))
    print("95th percentile:", int(np.percentile(hindi, 95)))
    print("99th percentile:", int(np.percentile(hindi, 99)))
    print("Maximum:", int(hindi.max()))
    print(
        "Exceeds max_length:",
        int((hindi > MAX_LENGTH).sum()),
        f"({(hindi > MAX_LENGTH).mean() * 100:.2f}%)"
    )

    print()
    print("MUNDARI")
    print("Mean:", round(mundari.mean(), 2))
    print("Median:", int(np.median(mundari)))
    print("95th percentile:", int(np.percentile(mundari, 95)))
    print("99th percentile:", int(np.percentile(mundari, 99)))
    print("Maximum:", int(mundari.max()))
    print(
        "Exceeds max_length:",
        int((mundari > MAX_LENGTH).sum()),
        f"({(mundari > MAX_LENGTH).mean() * 100:.2f}%)"
    )

    # Pair-level length relationship
    difference = mundari - hindi
    ratio = mundari / np.maximum(hindi, 1)

    print()
    print("PAIR LENGTH RELATIONSHIP")
    print("Average Mundari - Hindi length:", round(difference.mean(), 2))
    print("Median Mundari - Hindi length:", int(np.median(difference)))
    print("Average Mundari/Hindi ratio:", round(ratio.mean(), 2))

    print()
    print("Mundari longer than Hindi:", int((difference > 0).sum()))
    print("Hindi longer than Mundari:", int((difference < 0).sum()))
    print("Same length:", int((difference == 0).sum()))


def main():
    tokenizer = HindiMundariTokenizer()

    train = pd.read_csv(TRAIN_FILE)
    dev = pd.read_csv(DEV_FILE)
    test = pd.read_csv(TEST_FILE)

    analyze_dataset("TRAIN", train, tokenizer)
    analyze_dataset("DEV", dev, tokenizer)
    analyze_dataset("TEST", test, tokenizer)


if __name__ == "__main__":
    main()