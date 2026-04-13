from typing import Dict, Iterable, Optional, Tuple, Sequence

import os
import torch
from torch.utils.data import DataLoader


def build_plain_collator():
    """
    Return a simple collate function for dict of tensors produced by tokenizers
    datasets (input_ids/attention_mask/labels) without additional padding logic.
    We rely on upstream tokenizer padding; in Stage-1 we usually pass an HF
    DataCollatorWithPadding instead. This fallback is kept for completeness.
    """
    def _collate(batch):
        keys = batch[0].keys()
        out = {}
        for k in keys:
            if torch.is_tensor(batch[0][k]):
                out[k] = torch.stack([b[k] for b in batch], dim=0)
            else:
                out[k] = [b[k] for b in batch]
        return out
    return _collate


@torch.no_grad()
def extract_embeddings_loop(
    model,
    dataset,
    data_collator,
    batch_size: int,
) -> torch.Tensor:
    """
    Iterate dataset (no shuffle) and return pooled features on CPU with order preserved.
    The model must implement .encode(input_ids, attention_mask) -> (B, H).
    """
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator, drop_last=False)
    model_was_training = model.training
    model.eval()
    feats = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None and torch.is_tensor(attention_mask):
            attention_mask = attention_mask.to(device)
        global_attention_mask = batch.get("global_attention_mask", None)
        if global_attention_mask is not None and torch.is_tensor(global_attention_mask):
            global_attention_mask = global_attention_mask.to(device)
        f = model.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        feats.append(f.detach().cpu())
    if model_was_training:
        model.train()
    return torch.cat(feats, dim=0)


def save_embeddings_pt(
    path: str,
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    row_idx: torch.Tensor,
    split: str,
    pooling: str,
    model_name: str,
    hidden_size: int,
    label_name: str,
    fold_id: int,
    text: Optional[Sequence[str]] = None,
    note_id: Optional[Iterable] = None,
    subject_id: Optional[Iterable] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, object] = {
        "features": features.cpu().contiguous(),
        "labels": labels.cpu().long().contiguous(),
        "row_idx": row_idx.cpu().long().contiguous(),
        "meta": {
            "split": split,
            "pooling": pooling,
            "model_name": model_name,
            "hidden_size": int(hidden_size),
            "label_name": label_name,
            "fold_id": int(fold_id),
        },
    }
    if text is not None:
        # Persist clean_text explicitly for alignment with raw text in Stage-2 or analysis
        payload["text"] = list(map(str, text))
    if note_id is not None:
        payload["note_id"] = list(note_id)
    if subject_id is not None:
        payload["subject_id"] = list(subject_id)
    torch.save(payload, path)

def load_embeddings_pt(path: str) -> Dict[str, object]:
    """
    Load embeddings saved by save_embeddings_pt.
    Always map to CPU for wider compatibility; caller can .to(device) later.
    """
    obj = torch.load(path, map_location="cpu")
    return obj

