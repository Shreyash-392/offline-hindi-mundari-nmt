import torch

from model.transformer import TransformerNMT
from training.train_loader import create_dev_loader
from training.validation import validate_model


MAX_LENGTH = 80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

dev_loader = create_dev_loader(
    batch_size=16,
    max_length=MAX_LENGTH,
)

model = TransformerNMT(
    vocab_size=8000,
    max_length=MAX_LENGTH,
).to(device)

dev_loss = validate_model(
    model,
    dev_loader,
    device,
)

print("Development samples:", len(dev_loader.dataset))
print("Development batches:", len(dev_loader))
print("Validation loss:", dev_loss)
print("Validation completed successfully.")