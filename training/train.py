import torch
import torch.nn as nn

from model.transformer import TransformerNMT
from training.train_loader import create_train_loader, create_dev_loader
from training.validation import validate_model


BATCH_SIZE = 16
MAX_LENGTH = 80
EPOCHS = 10
LEARNING_RATE = 0.0003
PAD_ID = 3

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# --------------------------------
# Training Data
# --------------------------------

train_loader = create_train_loader(
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
)

print("Training samples:", len(train_loader.dataset))
print("Training batches per epoch:", len(train_loader))


# --------------------------------
# Development Data
# --------------------------------

dev_loader = create_dev_loader(
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
)

print("Development samples:", len(dev_loader.dataset))
print("Development batches:", len(dev_loader))


# --------------------------------
# Model
# --------------------------------

model = TransformerNMT(
    vocab_size=8000,
    max_length=MAX_LENGTH,
).to(device)


# --------------------------------
# Optimizer
# --------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_ID
)


# --------------------------------
# Best-model tracking
# --------------------------------

best_validation_loss = float("inf")


# --------------------------------
# Training
# --------------------------------

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0

    print()
    print("=" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 60)

    # --------------------------------
    # Training loop
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
    # Training loss
    # --------------------------------

    train_loss = (
        total_train_loss /
        len(train_loader)
    )

    print()
    print(f"Training Loss: {train_loss:.4f}")

    # --------------------------------
    # Validation
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
    # Save checkpoint
    # --------------------------------

    checkpoint_path = (
        f"checkpoints/"
        f"transformer_augmented_epoch_{epoch + 1}.pt"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        },
        checkpoint_path,
    )

    print(
        "Checkpoint saved:",
        checkpoint_path,
    )

    # --------------------------------
    # Save best model
    # --------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        best_model_path = (
            "checkpoints/"
            "transformer_augmented_best.pt"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            },
            best_model_path,
        )

        print("NEW BEST MODEL!")

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
print("Training completed.")
print("=" * 60)

print(
    "Best validation loss:",
    f"{best_validation_loss:.4f}",
)

print(
    "Best model:",
    "checkpoints/transformer_large_best.pt",
)