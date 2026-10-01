import torch

from model.transformer import TransformerNMT
from tokenizer.tokenizer import HindiMundariTokenizer


MODEL_PATH = "checkpoints/transformer_epoch_5.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = HindiMundariTokenizer()

model = TransformerNMT(
    vocab_size=8000,
    max_length=80,
).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

text = "मैं बाजार जा रहा हूँ।"

src = torch.tensor(
    [tokenizer.encode(text)],
    dtype=torch.long,
    device=device,
)

generated = [tokenizer.bos_id]

print("Hindi:", text)
print()
print("Generated tokens:")

with torch.no_grad():

    for step in range(10):

        tgt = torch.tensor(
            [generated],
            dtype=torch.long,
            device=device,
        )

        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1)
        ).to(device)

        output = model(
            src,
            tgt,
            tgt_mask=tgt_mask,
        )

        next_token = output[
            0,
            -1
        ].argmax().item()

        generated.append(next_token)

        print(
            f"Step {step + 1}: "
            f"ID={next_token}, "
            f"Piece={repr(tokenizer.sp.id_to_piece(next_token))}"
        )

print()
print("Complete IDs:")
print(generated)