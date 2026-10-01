import torch
import torch.nn as nn

from model.transformer import TransformerNMT
from training.train_loader import create_train_loader, create_dev_loader
from training.validation import validate_model


BATCH_SIZE = 16
MAX_LENGTH = 80
EPOCHS = 10
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.01
PAD_ID = 3

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Training Direction: Mundari -> Hindi")

# --------------------------------
# Training Data (Mundari -> Hindi)
# --------------------------------

train_loader = create_train_loader(
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    src_lang="Mundari",
    tgt_lang="Hindi",
)

print("Training samples:", len(train_loader.dataset))
print("Training batches per epoch:", len(train_loader))


# --------------------------------
# Development Data (Mundari -> Hindi)
# --------------------------------

dev_loader = create_dev_loader(
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    src_lang="Mundari",
    tgt_lang="Hindi",
)

print("Development samples:", len(dev_loader.dataset))
print("Development batches:", len(dev_loader))


# --------------------------------
# Model (Official Baseline Architecture)
# --------------------------------

model = TransformerNMT(
    vocab_size=8000,
    d_model=256,
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=1024,
    max_length=MAX_LENGTH,
    dropout=0.1,
).to(device)


# --------------------------------
# Optimizer & Loss Criterion
# --------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_ID
)


# --------------------------------
# Best-model tracking
# --------------------------------

best_validation_loss = float("inf")


# --------------------------------
# Training Loop
# --------------------------------

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0

    print()
    print("=" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS} [Mundari -> Hindi]")
    print("=" * 60)

    # --------------------------------
    # Training Loop
    # --------------------------------

    for batch_index, batch in enumerate(train_loader):

        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        src_padding_mask = src == PAD_ID
        tgt_padding_mask = tgt_input == PAD_ID

        tgt_mask = torch.triu(
            torch.ones(
                tgt_input.size(1),
                tgt_input.size(1),
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        output = model(
            src,
            tgt_input,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_mask=tgt_mask,
        )

        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt_expected.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        if (batch_index + 1) % 100 == 0:

            current_loss = (
                total_train_loss /
                (batch_index + 1)
            )

            print(
                f"Batch {batch_index + 1}/{len(train_loader)} "
                f"| Train Loss: {current_loss:.4f}"
            )

    # --------------------------------
    # Epoch Training Loss
    # --------------------------------

    train_loss = (
        total_train_loss /
        len(train_loader)
    )

    print()
    print(f"Training Loss: {train_loss:.4f}")

    # --------------------------------
    # Validation Loss
    # --------------------------------

    validation_loss = validate_model(
        model,
        dev_loader,
        device,
    )

    print(
        f"Validation Loss: {validation_loss:.4f}"
    )

    # --------------------------------
    # Save Epoch Checkpoint
    # --------------------------------

    checkpoint_path = (
        f"checkpoints/"
        f"transformer_mundari_hindi_epoch_{epoch + 1}.pt"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "direction": "mundari_to_hindi",
        },
        checkpoint_path,
    )

    print(
        "Checkpoint saved:",
        checkpoint_path,
    )

    # --------------------------------
    # Save Best Model Checkpoint
    # --------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        best_model_path = (
            "checkpoints/"
            "transformer_mundari_hindi_best.pt"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "direction": "mundari_to_hindi",
            },
            best_model_path,
        )

        print("NEW BEST MODEL (Mundari -> Hindi)!")

        print(
            "Best validation loss:",
            f"{best_validation_loss:.4f}",
        )

        print(
            "Best model saved:",
            best_model_path,
        )


print()
print("=" * 60)
print("Training completed for Mundari -> Hindi.")
print("=" * 60)

print(
    "Best validation loss:",
    f"{best_validation_loss:.4f}",
)

print(
    "Best model:",
    "checkpoints/transformer_mundari_hindi_best.pt",
)
