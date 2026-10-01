import os
import torch
import torch.nn as nn

from model.transformer import TransformerNMT


class ONNXTransformerWrapper(nn.Module):
    """
    Wrapper for TransformerNMT to define standard positional inputs
    for ONNX export.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, tgt_mask):
        output = self.model(
            src=src,
            tgt=tgt,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_mask=tgt_mask,
        )
        return output


def export_model_to_onnx(checkpoint_path, output_onnx_path, model_name):
    print(f"\n--- Exporting {model_name} to ONNX ---")
    device = torch.device("cpu")

    model = TransformerNMT(
        vocab_size=8000,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        max_length=80,
        dropout=0.1,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wrapper = ONNXTransformerWrapper(model)

    # Use max length 80 for tracing so attention reshape nodes match max length
    batch_size = 1
    src_len = 80
    tgt_len = 80

    dummy_src = torch.randint(0, 8000, (batch_size, src_len), dtype=torch.long)
    dummy_tgt = torch.randint(0, 8000, (batch_size, tgt_len), dtype=torch.long)
    dummy_src_padding_mask = (dummy_src == 3)
    dummy_tgt_padding_mask = (dummy_tgt == 3)
    dummy_tgt_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool),
        diagonal=1,
    )

    dynamic_axes = {
        "src": {0: "batch_size", 1: "src_seq_len"},
        "tgt": {0: "batch_size", 1: "tgt_seq_len"},
        "src_padding_mask": {0: "batch_size", 1: "src_seq_len"},
        "tgt_padding_mask": {0: "batch_size", 1: "tgt_seq_len"},
        "tgt_mask": {0: "tgt_seq_len", 1: "tgt_seq_len"},
        "output": {0: "batch_size", 1: "tgt_seq_len"},
    }

    torch.onnx.export(
        wrapper,
        (dummy_src, dummy_tgt, dummy_src_padding_mask, dummy_tgt_padding_mask, dummy_tgt_mask),
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["src", "tgt", "src_padding_mask", "tgt_padding_mask", "tgt_mask"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"Exported successfully to: {output_onnx_path}")
    print(f"ONNX Model File Size: {size_mb:.2f} MB")


def main():
    os.makedirs("checkpoints", exist_ok=True)

    # 1. Hindi -> Mundari Model
    export_model_to_onnx(
        checkpoint_path="checkpoints/transformer_augmented_best.pt",
        output_onnx_path="checkpoints/transformer_hindi_mundari.onnx",
        model_name="Hindi -> Mundari",
    )

    # 2. Mundari -> Hindi Model
    export_model_to_onnx(
        checkpoint_path="checkpoints/transformer_mundari_hindi_best.pt",
        output_onnx_path="checkpoints/transformer_mundari_hindi.onnx",
        model_name="Mundari -> Hindi",
    )


if __name__ == "__main__":
    main()
