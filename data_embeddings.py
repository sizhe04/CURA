from typing import Dict, Optional

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    Dataset over precomputed embeddings.
    Expects a dict with keys:
      - features: FloatTensor [N, H]
      - labels: LongTensor [N]
    Optional:
      - row_idx, note_id, subject_id, meta, etc. (kept for downstream usage)
    """

    def __init__(self, payload: Dict[str, object]) -> None:
        super().__init__()
        feats = payload["features"]
        labels = payload["labels"]
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats)
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels)
        self.features = feats.float().contiguous()
        self.labels = labels.long().contiguous()
        # Optional neighbor-aware fields (all shape [N], float)
        self.neighbor_q: Optional[torch.Tensor] = None
        self.neighbor_hhat: Optional[torch.Tensor] = None
        self.neighbor_w: Optional[torch.Tensor] = None
        self.neighbor_t: Optional[torch.Tensor] = None
        n = int(self.features.shape[0])
        for key in ("neighbor_q", "neighbor_hhat", "neighbor_w", "neighbor_t"):
            if key not in payload:
                continue
            v = payload[key]
            if not torch.is_tensor(v):
                v = torch.tensor(v)
            v = v.float().contiguous().view(-1)
            if int(v.shape[0]) != n:
                raise ValueError(f"[EmbeddingDataset] Field '{key}' length mismatch: got {int(v.shape[0])}, expected {n}")
            setattr(self, key, v)
        # keep optional fields
        self.payload = payload

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {
            "features": self.features[idx],
            "labels": self.labels[idx],
        }
        # Only include neighbor fields if they exist in the payload
        if self.neighbor_q is not None:
            out["neighbor_q"] = self.neighbor_q[idx]
        if self.neighbor_hhat is not None:
            out["neighbor_hhat"] = self.neighbor_hhat[idx]
        if self.neighbor_w is not None:
            out["neighbor_w"] = self.neighbor_w[idx]
        if self.neighbor_t is not None:
            out["neighbor_t"] = self.neighbor_t[idx]
        return out


