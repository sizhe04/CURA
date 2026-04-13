"""
Utility functions for representation-space neighborhood uncertainty (U_repr).

Computes label-entropy and label-margin on precomputed embeddings using kNN.
Both raw scores and normalized scores are returned:
 - entropy_norm = entropy / log(2)  (range [0,1] for binary labels)
 - margin_norm  = margin * 2        (range [0,1] for binary labels)
"""
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch


@dataclass
class ReprUncertConfig:
    k_neighbors: int = 20
    distance_metric: str = "cosine"  # "cosine" | "euclidean"
    search_split: str = "train+val"  # "train" | "train+val"
    score_type: str = "entropy"  # recorded only; we always compute both
    normalize: bool = True
    version: str = "repr_uncert_v1"


def _build_knn(features: np.ndarray, config: ReprUncertConfig) -> NearestNeighbors:
    metric = config.distance_metric
    if metric not in {"cosine", "euclidean"}:
        raise ValueError(f"Unsupported distance_metric: {metric}")
    k = min(max(1, int(config.k_neighbors)), max(1, features.shape[0]))
    nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto")
    nn.fit(features)
    return nn


def _compute_probs(labels: np.ndarray, nn_indices: np.ndarray) -> np.ndarray:
    # labels: (N,), nn_indices: (N, k)
    neigh_labels = labels[nn_indices]  # (N, k)
    # binary labels expected: 0/1
    count_pos = neigh_labels.sum(axis=1)
    k = neigh_labels.shape[1]
    probs_pos = count_pos / float(k)
    probs = np.stack([1.0 - probs_pos, probs_pos], axis=1)  # (N, 2)
    return probs


def compute_neighbor_stats(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    search_feats: np.ndarray,
    search_labels: np.ndarray,
    config: ReprUncertConfig,
    *,
    chunk_size: int = 0,
    progress: Optional[Callable[[int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-sample neighborhood statistics q(x) and H(q(x)) on precomputed embeddings.

    Returns three 1-D arrays of shape (N_train,):
      - q:      empirical positive rate in the kNN neighborhood of each train sample
      - H_raw:  binary entropy H(q) = -[q log q + (1-q) log(1-q)]
      - H_hat:  normalized entropy H(q)/log 2 in [0, 1]

    Notes
    -----
    - This is a lightweight wrapper around `_build_knn` and `_compute_probs`,
      designed so that Stage-2 can reuse exactly the same kNN protocol as U_repr
      without affecting existing Stage-1 behavior.
    - `train_labels` and `search_labels` are only used for sanity checks here;
      the neighbor composition is fully determined by `search_labels`.
    """
    if train_feats.shape[0] != train_labels.shape[0]:
        raise ValueError("train_feats and train_labels must align on first dimension.")
    if search_feats.shape[0] != search_labels.shape[0]:
        raise ValueError("search_feats and search_labels must align on first dimension.")

    nn = _build_knn(search_feats, config)

    N = int(train_feats.shape[0])
    eps = 1e-12

    if chunk_size is None:
        chunk_size = 0
    chunk_size = int(chunk_size)

    q = np.empty((N,), dtype=np.float64)
    H_raw = np.empty((N,), dtype=np.float64)

    if chunk_size <= 0 or chunk_size >= N:
        # single-shot computation (equivalent to original implementation style)
        nn_indices = nn.kneighbors(train_feats, return_distance=False)  # (N, k)
        probs = _compute_probs(search_labels, nn_indices)  # (N, 2)
        q[:] = probs[:, 1]
        H_raw[:] = -(q * np.log(q + eps) + (1.0 - q) * np.log(1.0 - q + eps))
        if progress is not None:
            progress(N)
    else:
        done = 0
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            nn_indices = nn.kneighbors(train_feats[start:end], return_distance=False)  # (B, k)
            probs = _compute_probs(search_labels, nn_indices)  # (B, 2)
            q_batch = probs[:, 1]
            H_batch = -(q_batch * np.log(q_batch + eps) + (1.0 - q_batch) * np.log(1.0 - q_batch + eps))
            q[start:end] = q_batch
            H_raw[start:end] = H_batch
            done = end
            if progress is not None:
                progress(done)

    H_hat = H_raw / np.log(2.0)
    return q, H_raw, H_hat


def compute_repr_scores(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    search_feats: np.ndarray,
    search_labels: np.ndarray,
    config: ReprUncertConfig,
    *,
    chunk_size: int = 0,
    progress: Optional[Callable[[int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (entropy_raw, entropy_norm, margin_raw, margin_norm), each shape (N,).
    When ``chunk_size`` > 0, process in batches and call ``progress`` with the cumulative
    sample count after each batch for progress logging. Results match one-shot processing
    exactly; only logging/visibility differs.
    """
    if train_feats.shape[0] != train_labels.shape[0]:
        raise ValueError("train_feats and train_labels must align on first dimension.")
    if search_feats.shape[0] != search_labels.shape[0]:
        raise ValueError("search_feats and search_labels must align on first dimension.")

    nn = _build_knn(search_feats, config)
    k = nn.n_neighbors

    N = int(train_feats.shape[0])
    eps = 1e-12

    if chunk_size is None:
        chunk_size = 0
    chunk_size = int(chunk_size)

    if chunk_size <= 0 or chunk_size >= N:
        # One-shot processing (equivalent to the original implementation)
        nn_indices = nn.kneighbors(train_feats, return_distance=False)  # (N, k)
        probs = _compute_probs(search_labels, nn_indices)  # (N, 2)
        entropy_raw = -(probs * np.log(probs + eps)).sum(axis=1)
        margin_raw = 1.0 - probs.max(axis=1)
        if progress is not None:
            progress(N)
    else:
        # Batch processing only affects printing and memory use; numerical results unchanged
        entropy_raw = np.empty((N,), dtype=np.float64)
        margin_raw = np.empty((N,), dtype=np.float64)
        done = 0
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            nn_indices = nn.kneighbors(train_feats[start:end], return_distance=False)  # (B, k)
            probs = _compute_probs(search_labels, nn_indices)  # (B, 2)
            e_raw = -(probs * np.log(probs + eps)).sum(axis=1)
            m_raw = 1.0 - probs.max(axis=1)
            entropy_raw[start:end] = e_raw
            margin_raw[start:end] = m_raw
            done = end
            if progress is not None:
                progress(done)

    # normalization by theoretical ranges
    entropy_norm = entropy_raw / np.log(2.0)
    margin_norm = margin_raw * 2.0
    if not config.normalize:
        entropy_norm = entropy_raw.copy()
        margin_norm = margin_raw.copy()

    return entropy_raw, entropy_norm, margin_raw, margin_norm


def attach_repr_to_payload(
    train_payload: Dict[str, object],
    search_payloads: Iterable[Dict[str, object]],
    config: ReprUncertConfig,
) -> Dict[str, object]:
    """
    Mutates and returns train_payload with four new tensors:
      - u_repr_entropy
      - u_repr_entropy_norm
      - u_repr_margin
      - u_repr_margin_norm
    Adds meta['repr_uncert_config'] for reproducibility.
    """
    # gather features/labels
    def _to_np(x):
        if torch.is_tensor(x):
            return x.cpu().numpy()
        return np.asarray(x)

    train_feats = _to_np(train_payload["features"])
    train_labels = _to_np(train_payload["labels"]).astype(np.int64)
    n_train = int(train_feats.shape[0])

    search_feats_list = []
    search_labels_list = []
    for p in search_payloads:
        search_feats_list.append(_to_np(p["features"]))
        search_labels_list.append(_to_np(p["labels"]).astype(np.int64))
    search_feats = np.concatenate(search_feats_list, axis=0)
    search_labels = np.concatenate(search_labels_list, axis=0)

    # Progress logging: print "processed/total" after each batch.
    # Does not change computation; only adds logging inside the batch loop.
    processed = 0

    def _progress(n_done: int) -> None:
        nonlocal processed
        n_done = int(n_done)
        if n_done <= processed:
            return
        processed = n_done
        print(f"[U_repr] train progress: {processed}/{n_train}", flush=True)

    # Default: at most 1000 samples per batch to avoid overly frequent prints
    chunk_size = max(1, min(1000, n_train))

    # 1) Original U_repr (entropy / margin) computation
    ent_raw, ent_norm, mar_raw, mar_norm = compute_repr_scores(
        train_feats=train_feats,
        train_labels=train_labels,
        search_feats=search_feats,
        search_labels=search_labels,
        config=config,
        chunk_size=chunk_size,
        progress=_progress,
    )

    # 2) Extra: neighborhood q(x) and H_hat(q) for Stage-2 neighborhood-aware CE reuse.
    #    Reuses the same kNN configuration and search set as U_repr; computed once offline.
    try:
        q_arr, _, hhat_arr = compute_neighbor_stats(
            train_feats=train_feats,
            train_labels=train_labels,
            search_feats=search_feats,
            search_labels=search_labels,
            config=config,
            chunk_size=chunk_size,
            progress=None,  # Avoid duplicate progress logs; U_repr progress is enough
        )
    except Exception as e:  # Conservative: on failure, skip writing q/H_hat only
        print(f"[U_repr] Warning: failed to compute neighbor q/H_hat: {e}", flush=True)
        q_arr = None
        hhat_arr = None

    train_payload = dict(train_payload)  # shallow copy to avoid side effects
    train_payload["u_repr_entropy"] = torch.tensor(ent_raw, dtype=torch.float)
    train_payload["u_repr_entropy_norm"] = torch.tensor(ent_norm, dtype=torch.float)
    train_payload["u_repr_margin"] = torch.tensor(mar_raw, dtype=torch.float)
    train_payload["u_repr_margin_norm"] = torch.tensor(mar_norm, dtype=torch.float)
    # If q/H_hat succeeded, persist them too; field names stay aligned with Stage-2
    if q_arr is not None and hhat_arr is not None:
        train_payload["neighbor_q"] = torch.tensor(q_arr, dtype=torch.float)
        train_payload["neighbor_hhat"] = torch.tensor(hhat_arr, dtype=torch.float)

    meta = dict(train_payload.get("meta", {}))
    meta["repr_uncert_config"] = asdict(config)
    if q_arr is not None and hhat_arr is not None:
        # Brief flag: this payload also includes neighborhood q/H_hat
        meta.setdefault("neighbor_stats_from_repr", True)
    meta.setdefault("repr_uncert_version", config.version)
    train_payload["meta"] = meta
    return train_payload

