import sys
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

CHECKPOINT = "checkpoints/transformer_mundari_hindi_best.pt"
TEST_FILE = "data/test/mundari/test.csv"


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


def translate_greedy(model, tokenizer, source_text):
    model.eval()

    src_ids = tokenizer.encode(source_text, max_length=MAX_LENGTH)
    src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
    src_padding_mask = src == PAD_ID

    generated = [tokenizer.bos_id]

    for _ in range(MAX_LENGTH - 1):
        tgt = torch.tensor([generated], dtype=torch.long, device=DEVICE)
        tgt_padding_mask = tgt == PAD_ID
        tgt_mask = create_tgt_mask(tgt.size(1))

        with torch.no_grad():
            output = model(
                src,
                tgt,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask,
                tgt_mask=tgt_mask,
            )

        next_token = output[:, -1, :].argmax(dim=-1).item()
        generated.append(next_token)

        if next_token == tokenizer.eos_id:
            break

    return tokenizer.decode(generated)


def translate_beam(model, tokenizer, source_text, beam_width=3):
    model.eval()

    src_ids = tokenizer.encode(source_text, max_length=MAX_LENGTH)
    src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
    src_padding_mask = src == PAD_ID

    beams = [([tokenizer.bos_id], 0.0)]
    completed = []

    for _ in range(MAX_LENGTH - 1):
        candidates = []

        for generated, score in beams:
            if generated[-1] == tokenizer.eos_id:
                completed.append((generated, score))
                continue

            tgt = torch.tensor([generated], dtype=torch.long, device=DEVICE)
            tgt_padding_mask = tgt == PAD_ID
            tgt_mask = create_tgt_mask(tgt.size(1))

            with torch.no_grad():
                output = model(
                    src,
                    tgt,
                    src_key_padding_mask=src_padding_mask,
                    tgt_key_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=src_padding_mask,
                    tgt_mask=tgt_mask,
                )

            log_probs = torch.log_softmax(output[:, -1, :], dim=-1)
            top_log_probs, top_tokens = torch.topk(log_probs, beam_width, dim=-1)

            for i in range(beam_width):
                token = top_tokens[0, i].item()
                token_score = top_log_probs[0, i].item()
                candidates.append((generated + [token], score + token_score))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

    completed.extend(beams)

    if not completed:
        return ""

    best_sequence, _ = max(completed, key=lambda x: x[1] / len(x[0]))
    return tokenizer.decode(best_sequence)


def is_repetitive(text):
    tokens = text.split()
    if len(tokens) < 4:
        return False
    for i in range(len(tokens) - 3):
        if tokens[i] == tokens[i + 1] == tokens[i + 2] == tokens[i + 3]:
            return True
    return False


def calculate_statistics(predictions):
    empty = sum(1 for p in predictions if not p.strip())
    repetitive = sum(1 for p in predictions if is_repetitive(p))
    total_tokens = sum(len(p.split()) for p in predictions)
    total = len(predictions)

    return {
        "empty_pct": 100 * empty / total,
        "repetitive_pct": 100 * repetitive / total,
        "avg_tokens": total_tokens / total,
    }


def main():
    print(f"Device: {DEVICE}", flush=True)
    print("Fast Evaluation for Mundari -> Hindi Model", flush=True)

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

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint loaded: {CHECKPOINT} (Epoch {checkpoint.get('epoch')}, Val Loss: {checkpoint.get('validation_loss'):.4f})", flush=True)

    test_data = pd.read_csv(TEST_FILE)
    # Evaluate first 300 test sentences for immediate empirical results
    eval_subset = test_data.head(300)
    print(f"Evaluating subset of {len(eval_subset)} test samples...", flush=True)

    greedy_preds = []
    beam3_preds = []
    refs = []

    for idx, row in eval_subset.iterrows():
        mundari = str(row["Mundari"])
        hindi = str(row["Hindi"])

        g_pred = translate_greedy(model, tokenizer, mundari)
        b3_pred = translate_beam(model, tokenizer, mundari, beam_width=3)

        greedy_preds.append(g_pred)
        beam3_preds.append(b3_pred)
        refs.append(hindi)

        if (idx + 1) % 50 == 0:
            print(f"Evaluated {idx + 1}/{len(eval_subset)} samples...", flush=True)

    g_bleu = sacrebleu.corpus_bleu(greedy_preds, [refs]).score
    g_chrf = sacrebleu.corpus_chrf(greedy_preds, [refs]).score
    g_stats = calculate_statistics(greedy_preds)

    b3_bleu = sacrebleu.corpus_bleu(beam3_preds, [refs]).score
    b3_chrf = sacrebleu.corpus_chrf(beam3_preds, [refs]).score
    b3_stats = calculate_statistics(beam3_preds)

    print("\n" + "=" * 60, flush=True)
    print("MUNDARI -> HINDI EVALUATION RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Metric':<25}{'Greedy':>15}{'Beam=3':>15}", flush=True)
    print("-" * 55, flush=True)
    print(f"{'BLEU':<25}{g_bleu:>15.4f}{b3_bleu:>15.4f}", flush=True)
    print(f"{'chrF++':<25}{g_chrf:>15.4f}{b3_chrf:>15.4f}", flush=True)
    print(f"{'Repetition %':<25}{g_stats['repetitive_pct']:>15.2f}{b3_stats['repetitive_pct']:>15.2f}", flush=True)
    print(f"{'Empty %':<25}{g_stats['empty_pct']:>15.2f}{b3_stats['empty_pct']:>15.2f}", flush=True)
    print(f"{'Average tokens':<25}{g_stats['avg_tokens']:>15.2f}{b3_stats['avg_tokens']:>15.2f}", flush=True)

    print("\nSample Translations:", flush=True)
    for i in range(3):
        print(f"\nMundari (Source): {eval_subset.iloc[i]['Mundari']}", flush=True)
        print(f"Reference Hindi: {eval_subset.iloc[i]['Hindi']}", flush=True)
        print(f"Greedy Output:   {greedy_preds[i]}", flush=True)
        print(f"Beam=3 Output:   {beam3_preds[i]}", flush=True)


if __name__ == "__main__":
    main()
