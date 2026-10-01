import torch
import pandas as pd
from collections import Counter

from model.transformer import TransformerNMT
from tokenizer.tokenizer import HindiMundariTokenizer


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

VOCAB_SIZE = 8000
MAX_LENGTH = 80
PAD_ID = 3

CHECKPOINT = CHECKPOINT = "checkpoints/transformer_best.pt"
TEST_FILE = "data/test/mundari/test.csv"


def translate(model, tokenizer, source_text):

    model.eval()

    src_ids = tokenizer.encode(
        source_text,
        max_length=MAX_LENGTH,
    )

    src = torch.tensor(
        [src_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    src_padding_mask = src == PAD_ID

    generated = [tokenizer.bos_id]

    for _ in range(MAX_LENGTH - 1):

        tgt = torch.tensor(
            [generated],
            dtype=torch.long,
            device=DEVICE,
        )

        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1)
        ).to(DEVICE)

        with torch.no_grad():

            output = model(
                src,
                tgt,
                src_key_padding_mask=src_padding_mask,
                memory_key_padding_mask=src_padding_mask,
                tgt_mask=tgt_mask,
            )

        next_token = (
            output[:, -1, :]
            .argmax(dim=-1)
            .item()
        )

        generated.append(next_token)

        if next_token == tokenizer.eos_id:
            break

    return generated


def has_repetition(token_ids, tokenizer):

    clean_ids = [
        token_id
        for token_id in token_ids
        if token_id not in {
            tokenizer.bos_id,
            tokenizer.eos_id,
            tokenizer.pad_id,
        }
    ]

    if len(clean_ids) < 4:
        return False

    # Check whether the same token appears
    # at least 4 times in the prediction.
    counts = Counter(clean_ids)

    if max(counts.values()) >= 4:
        return True

    # Check consecutive repetition.
    for i in range(len(clean_ids) - 3):

        if (
            clean_ids[i]
            == clean_ids[i + 1]
            == clean_ids[i + 2]
            == clean_ids[i + 3]
        ):
            return True

    return False


def main():

    print("Device:", DEVICE)

    tokenizer = HindiMundariTokenizer()

    model = TransformerNMT(
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Checkpoint loaded:", CHECKPOINT)

    test_data = pd.read_csv(TEST_FILE)

    total = len(test_data)

    empty_predictions = 0
    eos_predictions = 0
    repetitive_predictions = 0

    total_generated_tokens = 0

    repetitive_examples = []

    for index, row in test_data.iterrows():

        hindi = str(row["Hindi"])

        generated_ids = translate(
            model,
            tokenizer,
            hindi,
        )

        clean_ids = [
            token_id
            for token_id in generated_ids
            if token_id not in {
                tokenizer.bos_id,
                tokenizer.eos_id,
                tokenizer.pad_id,
            }
        ]

        total_generated_tokens += len(clean_ids)

        if len(clean_ids) == 0:
            empty_predictions += 1

        if generated_ids[-1] == tokenizer.eos_id:
            eos_predictions += 1

        if has_repetition(
            generated_ids,
            tokenizer,
        ):
            repetitive_predictions += 1

            if len(repetitive_examples) < 10:

                repetitive_examples.append(
                    {
                        "Hindi": hindi,
                        "Mundari": tokenizer.decode(
                            generated_ids
                        ),
                    }
                )

        if (index + 1) % 500 == 0:

            print(
                f"Processed {index + 1}/{total}"
            )

    average_length = (
        total_generated_tokens / total
    )

    repetition_percentage = (
        repetitive_predictions / total
    ) * 100

    eos_percentage = (
        eos_predictions / total
    ) * 100

    empty_percentage = (
        empty_predictions / total
    ) * 100

    print()
    print("=" * 60)
    print("Repetition Diagnostic")
    print("=" * 60)

    print("Total test samples:", total)

    print(
        "Empty predictions:",
        empty_predictions,
        f"({empty_percentage:.2f}%)",
    )

    print(
        "Predictions ending with EOS:",
        eos_predictions,
        f"({eos_percentage:.2f}%)",
    )

    print(
        "Repetitive predictions:",
        repetitive_predictions,
        f"({repetition_percentage:.2f}%)",
    )

    print(
        "Average generated token count:",
        f"{average_length:.2f}",
    )

    print()
    print("=" * 60)
    print("Examples of repetitive predictions")
    print("=" * 60)

    for i, example in enumerate(
        repetitive_examples,
        start=1,
    ):

        print()
        print(f"Example {i}")

        print("Hindi:")
        print(example["Hindi"])

        print("Predicted Mundari:")
        print(example["Mundari"])


if __name__ == "__main__":
    main()