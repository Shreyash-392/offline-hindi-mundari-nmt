import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer.tokenizer import HindiMundariTokenizer


class HindiMundariDataset(Dataset):
    def __init__(
        self,
        csv_path,
        tokenizer,
        max_length=80,
        src_lang="Hindi",
        tgt_lang="Mundari",
    ):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        src_text = str(row[self.src_lang])
        tgt_text = str(row[self.tgt_lang])

        src = self.tokenizer.encode(
            src_text,
            max_length=self.max_length,
        )

        tgt = self.tokenizer.encode(
            tgt_text,
            max_length=self.max_length,
        )

        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt": torch.tensor(tgt, dtype=torch.long),
        }