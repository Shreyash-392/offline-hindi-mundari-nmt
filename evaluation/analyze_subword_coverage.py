import pandas as pd
from collections import Counter
from tokenizer.tokenizer import HindiMundariTokenizer

TRAIN_FILE = "data/processed/train/train.csv"
DEV_FILE = "data/processed/dev/dev.csv"
TEST_FILE = "data/test/mundari/test.csv"


def get_subwords(texts, tokenizer):
    counter = Counter()

    for text in texts:
        pieces = tokenizer.sp.encode(str(text), out_type=str)
        counter.update(pieces)

    return counter


def analyze_split(name, train_counter, split_counter):
    total = sum(split_counter.values())

    # Subwords that occurred in training
    seen_tokens = sum(
        count for token, count in split_counter.items()
        if token in train_counter
    )

    unseen_tokens = total - seen_tokens

    unique_tokens = len(split_counter)

    unseen_unique = sum(
        1 for token in split_counter
        if token not in train_counter
    )

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print("Total subword tokens:", total)
    print("Unique subword tokens:", unique_tokens)

    print(
        "Subword tokens seen in training:",
        seen_tokens,
        f"({seen_tokens / total * 100:.2f}%)"
    )

    print(
        "Subword tokens NOT seen in training:",
        unseen_tokens,
        f"({unseen_tokens / total * 100:.2f}%)"
    )

    print(
        "Unique subwords seen in training:",
        unique_tokens - unseen_unique
    )

    print(
        "Unique subwords NOT seen in training:",
        unseen_unique
    )

    print()
    print("Most frequent subwords:")

    for token, count in split_counter.most_common(20):
        print(f"{token:<20} {count}")


def main():

    tokenizer = HindiMundariTokenizer()

    train = pd.read_csv(TRAIN_FILE)
    dev = pd.read_csv(DEV_FILE)
    test = pd.read_csv(TEST_FILE)

    print("TRAIN:", len(train))
    print("DEV:", len(dev))
    print("TEST:", len(test))

    # Hindi
    train_hindi = get_subwords(train["Hindi"], tokenizer)
    dev_hindi = get_subwords(dev["Hindi"], tokenizer)
    test_hindi = get_subwords(test["Hindi"], tokenizer)

    # Mundari
    train_mundari = get_subwords(train["Mundari"], tokenizer)
    dev_mundari = get_subwords(dev["Mundari"], tokenizer)
    test_mundari = get_subwords(test["Mundari"], tokenizer)

    analyze_split(
        "DEV HINDI",
        train_hindi,
        dev_hindi
    )

    analyze_split(
        "TEST HINDI",
        train_hindi,
        test_hindi
    )

    analyze_split(
        "DEV MUNDARI",
        train_mundari,
        dev_mundari
    )

    analyze_split(
        "TEST MUNDARI",
        train_mundari,
        test_mundari
    )


if __name__ == "__main__":
    main()