import pandas as pd
from collections import Counter

TRAIN_FILE = "data/processed/train/train.csv"
DEV_FILE = "data/processed/dev/dev.csv"
TEST_FILE = "data/test/mundari/test.csv"


def analyze_language(data, column_name, name):
    counter = Counter()

    total_tokens = 0
    total_sentences = len(data)

    for text in data[column_name].astype(str):
        tokens = text.split()
        counter.update(tokens)
        total_tokens += len(tokens)

    unique_tokens = len(counter)

    frequencies = list(counter.values())

    once = sum(1 for x in frequencies if x == 1)
    twice = sum(1 for x in frequencies if x == 2)
    low_frequency = sum(1 for x in frequencies if x <= 5)

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print("Sentences:", total_sentences)
    print("Total tokens:", total_tokens)
    print("Unique tokens:", unique_tokens)

    print("Tokens appearing once:", once)
    print("Tokens appearing twice:", twice)
    print("Tokens appearing <=5 times:", low_frequency)

    print()
    print("Top 20 tokens:")

    for token, count in counter.most_common(20):
        print(f"{token:<30} {count}")


def main():

    train = pd.read_csv(TRAIN_FILE)
    dev = pd.read_csv(DEV_FILE)
    test = pd.read_csv(TEST_FILE)

    print("TRAIN:", len(train))
    print("DEV:", len(dev))
    print("TEST:", len(test))

    analyze_language(
        train,
        "Hindi",
        "TRAIN HINDI"
    )

    analyze_language(
        train,
        "Mundari",
        "TRAIN MUNDARI"
    )

    analyze_language(
        dev,
        "Hindi",
        "DEV HINDI"
    )

    analyze_language(
        dev,
        "Mundari",
        "DEV MUNDARI"
    )

    analyze_language(
        test,
        "Hindi",
        "TEST HINDI"
    )

    analyze_language(
        test,
        "Mundari",
        "TEST MUNDARI"
    )


if __name__ == "__main__":
    main()