import torch
import glob

files = sorted(
    glob.glob(
        "checkpoints/transformer_masked_long_epoch_*.pt"
    )
)

print("Epoch | Train Loss | Val Loss")
print("-" * 35)

for file in files:
    checkpoint = torch.load(
        file,
        map_location="cpu",
        weights_only=False
    )

    print(
        f"{checkpoint['epoch']:5} | "
        f"{checkpoint['train_loss']:10.4f} | "
        f"{checkpoint['validation_loss']:8.4f}"
    )