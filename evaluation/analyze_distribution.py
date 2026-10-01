import pandas as pd
import numpy as np
from collections import Counter
from tokenizer.tokenizer import HindiMundariTokenizer

TRAIN_FILE = "data/processed/train/train.csv"
DEV_FILE = "data/processed/dev/dev.csv"
TEST_FILE = "data/test/mundari/test.csv"


def get_counter(texts, tokenizer):
    counter = Counter()

    for text in texts:
        pieces = tokenizer.sp.encode(str(text), out_type=str)
        counter.update(pieces)

    return counter


def analyze(name, train_counter, split_counter):
    train_total = sum(train_counter.values())
    split_total = sum(split_counter.values())

    # Probability distribution of tokens
    all_tokens = set(train_counter) | set(split_counter)

    train_probs = np.array([
        train_counter[token] / train_total
        for token in all_tokens
    ])

    split_probs = np.array([
        split_counter[token] / split_total
        for token in all_tokens
    ])

    # Mean absolute difference in token probability
    mean_difference = np.mean(
        np.abs(train_probs - split_probs)
    )

    # Vocabulary overlap
    train_vocab = set(train_counter)
    split_vocab = set(split_counter)

    overlap = train_vocab & split_vocab

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print("Train unique subwords:", len(train_vocab))
    print("Split unique subwords:", len(split_vocab))
    print("Shared subwords:", len(overlap))

    print(
        "Split vocabulary covered by train:",
        f"{len(overlap) / len(split_vocab) * 100:.2f}%"
    )

    print(
        "Mean token-frequency difference:",
        f"{mean_difference:.8f}"
    )

    print()
    print("Most frequent split subwords:")

    for token, count in split_counter.most_common(20):
        train_count = train_counter.get(token, 0)

        print(
            f"{token:<20} "
            f"split={count:<6} "
            f"train={train_count}"
        )


def main():

    tokenizer = HindiMundariTokenizer()

    train = pd.read_csv(TRAIN_FILE)
    dev = pd.read_csv(DEV_FILE)
    test = pd.read_csv(TEST_FILE)

    train_hindi = get_counter(
        train["Hindi"],
        tokenizer
    )

    dev_hindi = get_counter(
        dev["Hindi"],
        tokenizer
    )

    test_hindi = get_counter(
        test["Hindi"],
        tokenizer
    )

    train_mundari = get_counter(
        train["Mundari"],
        tokenizer
    )

    dev_mundari = get_counter(
        dev["Mundari"],
        tokenizer
    )

    test_mundari = get_counter(
        test["Mundari"],
        tokenizer
    )

    analyze(
        "DEV HINDI",
        train_hindi,
        dev_hindi
    )

    analyze(
        "TEST HINDI",
        train_hindi,
        test_hindi
    )

    analyze(
        "DEV MUNDARI",
        train_mundari,
        dev_mundari
    )

    analyze(
        "TEST MUNDARI",
        train_mundari,
        test_mundari
    )


if __name__ == "__main__":
    main()