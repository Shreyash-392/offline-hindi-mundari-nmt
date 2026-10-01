import torch
import pandas as pd
import sacrebleu

from model.transformer import TransformerNMT
from tokenizer.tokenizer import HindiMundariTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

VOCAB_SIZE = 8000
MAX_LENGTH = 80
PAD_ID = 3

CHECKPOINT = "checkpoints/transformer_mundari_hindi_best.pt"
TEST_FILE = "data/test/mundari/test.csv"


# ============================================================
# CAUSAL MASK
# ============================================================

def create_tgt_mask(length):
    return torch.triu(
        torch.ones(
            length,
            length,
            device=DEVICE,
            dtype=torch.bool,
        ),
        diagonal=1,
    )


# ============================================================
# GREEDY DECODING
# ============================================================

def translate_greedy(
    model,
    tokenizer,
    source_text,
):
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

        tgt_mask = create_tgt_mask(
            tgt.size(1)
        )

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


# ============================================================
# BEAM SEARCH
# ============================================================

def translate_beam(
    model,
    tokenizer,
    source_text,
    beam_width=3,
):
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

    beams = [
        ([tokenizer.bos_id], 0.0)
    ]

    completed = []

    for _ in range(MAX_LENGTH - 1):

        candidates = []

        for generated, score in beams:

            if generated[-1] == tokenizer.eos_id:
                completed.append(
                    (generated, score)
                )
                continue

            tgt = torch.tensor(
                [generated],
                dtype=torch.long,
                device=DEVICE,
            )

            tgt_padding_mask = tgt == PAD_ID

            tgt_mask = create_tgt_mask(
                tgt.size(1)
            )

            with torch.no_grad():

                output = model(
                    src,
                    tgt,
                    src_key_padding_mask=src_padding_mask,
                    tgt_key_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=src_padding_mask,
                    tgt_mask=tgt_mask,
                )

            log_probs = torch.log_softmax(
                output[:, -1, :],
                dim=-1,
            )

            top_log_probs, top_tokens = torch.topk(
                log_probs,
                beam_width,
                dim=-1,
            )

            for i in range(beam_width):

                token = top_tokens[
                    0, i
                ].item()

                token_score = top_log_probs[
                    0, i
                ].item()

                new_sequence = (
                    generated + [token]
                )

                new_score = (
                    score + token_score
                )

                candidates.append(
                    (
                        new_sequence,
                        new_score,
                    )
                )

        if not candidates:
            break

        candidates.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        beams = candidates[:beam_width]

    completed.extend(beams)

    if not completed:
        return ""

    best_sequence, best_score = max(
        completed,
        key=lambda x: x[1] / len(x[0]),
    )

    return tokenizer.decode(
        best_sequence
    )


# ============================================================
# REPETITION DETECTION
# ============================================================

def is_repetitive(text):

    tokens = text.split()

    if len(tokens) < 4:
        return False

    for i in range(len(tokens) - 3):

        if (
            tokens[i]
            == tokens[i + 1]
            == tokens[i + 2]
            == tokens[i + 3]
        ):
            return True

    for i in range(len(tokens) - 7):

        if (
            tokens[i:i + 4]
            == tokens[i + 4:i + 8]
        ):
            return True

    return False


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(predictions):

    empty = 0
    repetitive = 0
    total_tokens = 0

    for prediction in predictions:

        if not prediction.strip():
            empty += 1

        if is_repetitive(prediction):
            repetitive += 1

        total_tokens += len(
            prediction.split()
        )

    total = len(predictions)

    return {
        "empty": empty,
        "empty_pct": 100 * empty / total,
        "repetitive": repetitive,
        "repetitive_pct": 100 * repetitive / total,
        "avg_tokens": total_tokens / total,
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    tokenizer,
    test_data,
    method,
    beam_width=None,
):

    predictions = []
    references = []

    total = len(test_data)

    for index, row in test_data.iterrows():

        # Source is Mundari, Reference Target is Hindi
        mundari = str(row["Mundari"])
        hindi = str(row["Hindi"])

        if method == "greedy":

            prediction = translate_greedy(
                model,
                tokenizer,
                mundari,
            )

        else:

            prediction = translate_beam(
                model,
                tokenizer,
                mundari,
                beam_width=beam_width,
            )

        predictions.append(prediction)
        references.append(hindi)

        if (index + 1) % 500 == 0:

            print(
                f"{method}: "
                f"{index + 1}/{total}"
            )

    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
    )

    chrf = sacrebleu.corpus_chrf(
        predictions,
        [references],
    )

    statistics = calculate_statistics(
        predictions
    )

    return (
        predictions,
        bleu.score,
        chrf.score,
        statistics,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Device:", DEVICE)
    print("Evaluating Mundari -> Hindi Translation Model")

    tokenizer = HindiMundariTokenizer()

    model = TransformerNMT(
        vocab_size=VOCAB_SIZE,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        max_length=MAX_LENGTH,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Checkpoint loaded:",
        CHECKPOINT,
    )

    test_data = pd.read_csv(
        TEST_FILE
    )

    print(
        "Test samples:",
        len(test_data),
    )

    # GREEDY
    print()
    print("=" * 60)
    print("GREEDY EVALUATION [Mundari -> Hindi]")
    print("=" * 60)

    (
        greedy_predictions,
        greedy_bleu,
        greedy_chrf,
        greedy_stats,
    ) = evaluate(
        model,
        tokenizer,
        test_data,
        "greedy",
    )

    print(f"Greedy BLEU: {greedy_bleu:.4f}")
    print(f"Greedy chrF++: {greedy_chrf:.4f}")
    print(f"Greedy repetition: {greedy_stats['repetitive_pct']:.2f}%")
    print(f"Greedy empty: {greedy_stats['empty_pct']:.2f}%")
    print(f"Greedy average tokens: {greedy_stats['avg_tokens']:.2f}")

    # BEAM = 3
    print()
    print("=" * 60)
    print("BEAM=3 EVALUATION [Mundari -> Hindi]")
    print("=" * 60)

    (
        beam3_predictions,
        beam3_bleu,
        beam3_chrf,
        beam3_stats,
    ) = evaluate(
        model,
        tokenizer,
        test_data,
        "beam",
        beam_width=3,
    )

    print(f"Beam=3 BLEU: {beam3_bleu:.4f}")
    print(f"Beam=3 chrF++: {beam3_chrf:.4f}")
    print(f"Beam=3 repetition: {beam3_stats['repetitive_pct']:.2f}%")
    print(f"Beam=3 empty: {beam3_stats['empty_pct']:.2f}%")
    print(f"Beam=3 average tokens: {beam3_stats['avg_tokens']:.2f}")

    # BEAM = 5
    print()
    print("=" * 60)
    print("BEAM=5 EVALUATION [Mundari -> Hindi]")
    print("=" * 60)

    (
        beam5_predictions,
        beam5_bleu,
        beam5_chrf,
        beam5_stats,
    ) = evaluate(
        model,
        tokenizer,
        test_data,
        "beam",
        beam_width=5,
    )

    print(f"Beam=5 BLEU: {beam5_bleu:.4f}")
    print(f"Beam=5 chrF++: {beam5_chrf:.4f}")
    print(f"Beam=5 repetition: {beam5_stats['repetitive_pct']:.2f}%")
    print(f"Beam=5 empty: {beam5_stats['empty_pct']:.2f}%")
    print(f"Beam=5 average tokens: {beam5_stats['avg_tokens']:.2f}")

    # FINAL COMPARISON
    print()
    print("=" * 75)
    print("FINAL COMPARISON [Mundari -> Hindi]")
    print("=" * 75)
    print(f"{'Metric':<25}{'Greedy':>15}{'Beam=3':>15}{'Beam=5':>15}")
    print("-" * 70)
    print(f"{'BLEU':<25}{greedy_bleu:>15.4f}{beam3_bleu:>15.4f}{beam5_bleu:>15.4f}")
    print(f"{'chrF++':<25}{greedy_chrf:>15.4f}{beam3_chrf:>15.4f}{beam5_chrf:>15.4f}")
    print(f"{'Repetition %':<25}{greedy_stats['repetitive_pct']:>15.2f}{beam3_stats['repetitive_pct']:>15.2f}{beam5_stats['repetitive_pct']:>15.2f}")
    print(f"{'Empty %':<25}{greedy_stats['empty_pct']:>15.2f}{beam3_stats['empty_pct']:>15.2f}{beam5_stats['empty_pct']:>15.2f}")
    print(f"{'Average tokens':<25}{greedy_stats['avg_tokens']:>15.2f}{beam3_stats['avg_tokens']:>15.2f}{beam5_stats['avg_tokens']:>15.2f}")

    # SAMPLES
    print()
    print("Sample comparison [Mundari -> Hindi]:")
    for i in range(min(5, len(test_data))):
        print()
        print("-" * 60)
        print("Mundari (Source):", test_data.iloc[i]["Mundari"])
        print("Reference Hindi:", test_data.iloc[i]["Hindi"])
        print("Greedy:", greedy_predictions[i])
        print("Beam=3:", beam3_predictions[i])
        print("Beam=5:", beam5_predictions[i])


if __name__ == "__main__":
    main()
