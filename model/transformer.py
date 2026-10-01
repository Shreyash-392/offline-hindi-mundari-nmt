import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_length=80):
        super().__init__()

        position = torch.arange(max_length).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_length, d_model)

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


class TransformerNMT(nn.Module):

    def __init__(
        self,
        vocab_size=8000,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        max_length=80,
        dropout=0.1,
    ):

        super().__init__()

        self.d_model = d_model

        self.src_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.tgt_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.src_positional = PositionalEncoding(
            d_model,
            max_length
        )

        self.tgt_positional = PositionalEncoding(
            d_model,
            max_length
        )

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.output_layer = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(
        self,
        src,
        tgt,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_mask=None,
    ):

        src = self.src_embedding(src) * math.sqrt(
            self.d_model
        )

        tgt = self.tgt_embedding(tgt) * math.sqrt(
            self.d_model
        )

        src = self.src_positional(src)

        tgt = self.tgt_positional(tgt)

        output = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return self.output_layer(output)