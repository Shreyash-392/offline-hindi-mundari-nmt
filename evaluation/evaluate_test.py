import torch
import pandas as pd
import sacrebleu

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

        tgt_padding_mask = tgt == PAD_ID

        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1)
        ).to(DEVICE)

        with torch.no_grad():

            output = model(
                src,
                tgt,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
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

    return tokenizer.decode(generated)


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

    print("Test samples:", len(test_data))

    predictions = []
    references = []

    for index, row in test_data.iterrows():

        hindi = str(row["Hindi"])
        mundari = str(row["Mundari"])

        prediction = translate(
            model,
            tokenizer,
            hindi,
        )

        predictions.append(prediction)
        references.append(mundari)

        if (index + 1) % 100 == 0:

            print(
                f"Processed {index + 1}/{len(test_data)}"
            )

    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
    )

    chrf = sacrebleu.corpus_chrf(
        predictions,
        [references],
    )

    print()
    print("=" * 60)
    print("Hindi → Mundari Test Results")
    print("=" * 60)

    print(f"BLEU: {bleu.score:.4f}")
    print(f"chrF++: {chrf.score:.4f}")

    print()
    print("Sample predictions:")

    for i in range(min(10, len(predictions))):

        print()
        print("Hindi:")
        print(test_data.iloc[i]["Hindi"])

        print("Reference Mundari:")
        print(test_data.iloc[i]["Mundari"])

        print("Predicted Mundari:")
        print(predictions[i])


if __name__ == "__main__":
    main()