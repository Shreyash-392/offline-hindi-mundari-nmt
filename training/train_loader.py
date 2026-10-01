import torch
from torch.utils.data import DataLoader

from tokenizer.tokenizer import HindiMundariTokenizer
from training.dataset import HindiMundariDataset


def create_train_loader(
    batch_size=16,
    max_length=80,
    src_lang="Hindi",
    tgt_lang="Mundari",
):
    tokenizer = HindiMundariTokenizer()

    dataset = HindiMundariDataset(
        "data/processed/train/train.csv",
        tokenizer,
        max_length=max_length,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    return loader


def create_dev_loader(
    batch_size=16,
    max_length=80,
    src_lang="Hindi",
    tgt_lang="Mundari",
):
    tokenizer = HindiMundariTokenizer()

    dataset = HindiMundariDataset(
        "data/processed/dev/dev.csv",
        tokenizer,
        max_length=max_length,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return loader