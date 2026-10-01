import torch

from model.transformer import TransformerNMT
from tokenizer.tokenizer import HindiMundariTokenizer


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

VOCAB_SIZE = 8000
MAX_LENGTH = 80
PAD_ID = 3

import os
CHECKPOINT = "checkpoints/transformer_augmented_best.pt"
if not os.path.exists(CHECKPOINT):
    CHECKPOINT = "checkpoints/transformer_best.pt"

def generate_translation(
    model,
    tokenizer,
    hindi_text,
    max_length=80,
):
    model.eval()

    # Encode Hindi
    src_ids = tokenizer.encode(
        hindi_text,
        max_length=max_length,
    )

    src = torch.tensor(
        [src_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    src_padding_mask = src == PAD_ID

    # Start decoder with BOS
    generated = [
        tokenizer.bos_id
    ]

    for _ in range(max_length - 1):

        tgt = torch.tensor(
            [generated],
            dtype=torch.long,
            device=DEVICE,
        )

        tgt_padding_mask = tgt == PAD_ID

        tgt_mask = torch.triu(
            torch.ones(
                tgt.size(1),
                tgt.size(1),
                device=DEVICE,
                dtype=torch.bool,
            ),
            diagonal=1,
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

        logits = output[:, -1, :].clone()
        # Apply repetition penalty to prevent single token loops
        for token_id in set(generated):
            if token_id not in (tokenizer.bos_id, tokenizer.eos_id, PAD_ID):
                logits[:, token_id] -= 2.5

        next_token = logits.argmax(dim=-1).item()
        generated.append(next_token)

        # Stop at EOS
        if next_token == tokenizer.eos_id:
            break

    return tokenizer.decode(generated)


def generate_translation_beam(
    model,
    tokenizer,
    hindi_text,
    max_length=80,
    beam_width=3,
):
    model.eval()

    # Encode Hindi
    src_ids = tokenizer.encode(
        hindi_text,
        max_length=max_length,
    )

    src = torch.tensor(
        [src_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    src_padding_mask = src == PAD_ID

    # Each beam = (token_ids, cumulative_log_probability)
    beams = [
        ([tokenizer.bos_id], 0.0)
    ]

    completed = []

    for _ in range(max_length - 1):

        candidates = []

        for generated, score in beams:

            # Already finished
            if generated[-1] == tokenizer.eos_id:
                completed.append((generated, score))
                continue

            tgt = torch.tensor(
                [generated],
                dtype=torch.long,
                device=DEVICE,
            )

            tgt_padding_mask = tgt == PAD_ID

            tgt_mask = torch.triu(
                torch.ones(
                    tgt.size(1),
                    tgt.size(1),
                    device=DEVICE,
                    dtype=torch.bool,
                ),
                diagonal=1,
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

            # Log probabilities for the newest token
            log_probs = torch.log_softmax(
                output[:, -1, :],
                dim=-1,
            )

            # Apply repetition penalty for tokens already present in sequence
            for token_id in set(generated):
                if token_id not in (tokenizer.bos_id, tokenizer.eos_id, PAD_ID):
                    log_probs[0, token_id] -= 3.0

            top_log_probs, top_tokens = torch.topk(
                log_probs,
                beam_width,
                dim=-1,
            )

            for i in range(beam_width):

                token = top_tokens[0, i].item()
                token_score = top_log_probs[0, i].item()

                new_sequence = generated + [token]
                new_score = score + token_score

                candidates.append(
                    (new_sequence, new_score)
                )

        if not candidates:
            break

        # Keep best beam candidates
        candidates.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        beams = candidates[:beam_width]

    # Add remaining beams
    completed.extend(beams)

    if not completed:
        return ""

    # Length-normalized score prevents very short sequences
    # from always winning.
    best_sequence, best_score = max(
        completed,
        key=lambda x: x[1] / len(x[0]),
    )

    return tokenizer.decode(best_sequence)


def main():

    print("Device:", DEVICE)

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

    print("Checkpoint loaded:", CHECKPOINT)

    hindi_text = input(
        "\nEnter Hindi sentence: "
    ).strip()

    greedy_translation = generate_translation(
        model,
        tokenizer,
        hindi_text,
        max_length=MAX_LENGTH,
    )

    beam_translation = generate_translation_beam(
        model,
        tokenizer,
        hindi_text,
        max_length=MAX_LENGTH,
        beam_width=3,
    )

    print("\nHindi:")
    print(hindi_text)

    print("\nGreedy decoding:")
    print(greedy_translation)

    print("\nBeam search (beam=3):")
    print(beam_translation)


if __name__ == "__main__":
    main()