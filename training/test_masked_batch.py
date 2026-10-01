import torch
import torch.nn as nn

from model.transformer import TransformerNMT
from training.train_loader import create_train_loader


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PAD_ID = 3

# Load one real batch
loader = create_train_loader(batch_size=16)
batch = next(iter(loader))

src = batch["src"].to(device)
tgt = batch["tgt"].to(device)

# Prepare decoder inputs
tgt_input = tgt[:, :-1]
tgt_expected = tgt[:, 1:]

# Create padding masks
src_padding_mask = src == PAD_ID
tgt_padding_mask = tgt_input == PAD_ID

# Create causal mask
tgt_mask = nn.Transformer.generate_square_subsequent_mask(
    tgt_input.size(1)
).to(device)

# Initialize model
model = TransformerNMT().to(device)
model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0003
)

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_ID
)

# Forward pass with all masks
output = model(
    src,
    tgt_input,
    src_key_padding_mask=src_padding_mask,
    tgt_key_padding_mask=tgt_padding_mask,
    memory_key_padding_mask=src_padding_mask,
    tgt_mask=tgt_mask,
)

# Calculate loss
loss = criterion(
    output.reshape(-1, output.size(-1)),
    tgt_expected.reshape(-1)
)

# Update model weights
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Device:", device)
print("Source shape:", src.shape)
print("Target input shape:", tgt_input.shape)
print("Output shape:", output.shape)
print("Source PAD positions:", src_padding_mask.sum().item())
print("Target PAD positions:", tgt_padding_mask.sum().item())
print("Loss:", loss.item())
print("Loss is finite:", torch.isfinite(loss).item())
print("Masked training step successful.")