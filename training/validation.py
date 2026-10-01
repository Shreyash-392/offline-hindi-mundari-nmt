import torch
import torch.nn as nn


PAD_ID = 3


def validate_model(model, dev_loader, device):
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():

        for batch in dev_loader:

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

            total_loss += loss.item()
            total_batches += 1

    average_loss = total_loss / total_batches

    return average_loss