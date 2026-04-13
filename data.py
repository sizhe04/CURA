from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class TextBinaryExample:
    text: str
    label: int


class ClinicalNotesDataset(Dataset):
    """Turn (clean_text, label) pairs into tokenized tensors for the Trainer."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        text_col: str = "clean_text",
        label_col: str = "label",
        max_length: int = 512,
        use_longformer_global_attention: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length
        self.use_longformer_global_attention = use_longformer_global_attention

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        label = int(row[self.label_col])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        if self.use_longformer_global_attention and "attention_mask" in item:
            global_attention_mask = torch.zeros_like(item["attention_mask"])
            if global_attention_mask.numel() > 0:
                global_attention_mask[0] = 1
            item["global_attention_mask"] = global_attention_mask
        return item

