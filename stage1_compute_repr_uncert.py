#!/usr/bin/env python
import argparse
import os
import shutil
from typing import List, Tuple

import torch

from utilities.embeddings_io import load_embeddings_pt
from utilities.repr_uncert import ReprUncertConfig, attach_repr_to_payload

try:
    # tqdm is only used for progress bars; if missing from the environment, fall back to no progress bar
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - only triggered when tqdm is missing
    tqdm = None  # type: ignore


def _list_folds(checkpoint_dir: str) -> List[str]:
    folds = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("fold_") and os.path.isdir(os.path.join(checkpoint_dir, name)):
            folds.append(name)
    return sorted(folds)


def _find_split_file(emb_dir: str, fold_id: int, split: str) -> str:
    """
    Find pt file under embeddings_last with preferred suffix order:
      1) _best_last.pt
      2) _last.pt
      3) no suffix fallback (legacy)
    """
    if not os.path.isdir(emb_dir):
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")
    prefix = f"fold_{fold_id}_{split}_pool-"
    candidates: List[Tuple[int, str]] = []
    for name in os.listdir(emb_dir):
        if not name.startswith(prefix) or not name.endswith(".pt"):
            continue
        if name.endswith("_best_last.pt"):
            prio = 0
        elif name.endswith("_last.pt"):
            prio = 1
        else:
            prio = 2
        candidates.append((prio, name))
    if not candidates:
        raise FileNotFoundError(f"No {split} embedding file found under {emb_dir}")
    best = sorted(candidates, key=lambda x: x[0])[0][1]
    return os.path.join(emb_dir, best)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compute U_repr (entropy/margin) for Stage-1 embeddings_last")
    p.add_argument("--stage1_label_dir", type=str, required=True, help="Path to a single label directory, e.g., .../Stage_1/.../death_in_7")
    p.add_argument("--k_neighbors", type=int, default=20)
    p.add_argument("--distance_metric", type=str, choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--search_split", type=str, choices=["train", "train+val"], default="train+val")
    p.add_argument("--score_type", type=str, choices=["entropy", "margin"], default="entropy", help="Recorded in meta; both scores are always computed")
    p.add_argument("--normalize", action="store_true", help="If set, store normalized scores (entropy/log2, margin*2) as *_norm")
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.set_defaults(normalize=True)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_root = os.path.join(args.stage1_label_dir, "checkpoint")
    folds = _list_folds(ckpt_root)
    if not folds:
        raise FileNotFoundError(f"No fold_* found under {ckpt_root}")

    loop = folds
    if tqdm is not None:
        loop = tqdm(folds, desc="U_repr over folds", unit="fold")  # type: ignore

    for fold_name in loop:
        fold_id = int(fold_name.split("_")[-1])
        fold_dir = os.path.join(ckpt_root, fold_name)
        src_dir = os.path.join(fold_dir, "embeddings_last")
        # Record k_neighbors in the destination directory name to distinguish offline runs for different K
        dst_dir_name = f"embeddings_last_llm_uncertain_K-{int(args.k_neighbors)}"
        dst_dir = os.path.join(fold_dir, dst_dir_name)
        os.makedirs(dst_dir, exist_ok=True)

        print(f"[Fold {fold_id}] Processing train/val/test files under embeddings_last...", flush=True)

        train_path = _find_split_file(src_dir, fold_id, "train")
        val_path = _find_split_file(src_dir, fold_id, "val")
        test_path = _find_split_file(src_dir, fold_id, "test")

        # Load embeddings for all three splits (Stage-1 trained representations only; no further training)
        train_payload = load_embeddings_pt(train_path)
        val_payload = load_embeddings_pt(val_path)
        test_payload = load_embeddings_pt(test_path)

        # kNN search pool: same as original U_repr, still train or train+val only
        search_payloads = [train_payload]
        if args.search_split == "train+val":
            search_payloads.append(val_payload)

        config = ReprUncertConfig(
            k_neighbors=args.k_neighbors,
            distance_metric=args.distance_metric,
            search_split=args.search_split,
            score_type=args.score_type,
            normalize=bool(args.normalize),
        )
        # Attach U_repr and neighborhood q/H_hat to train/val/test for Stage-2 reuse;
        # training still happens only on the train split; this is offline labeling only.
        updated_train = attach_repr_to_payload(train_payload, search_payloads, config)
        updated_val = attach_repr_to_payload(val_payload, search_payloads, config)
        updated_test = attach_repr_to_payload(test_payload, search_payloads, config)

        dst_train = os.path.join(dst_dir, os.path.basename(train_path))
        dst_val = os.path.join(dst_dir, os.path.basename(val_path))
        dst_test = os.path.join(dst_dir, os.path.basename(test_path))
        torch.save(updated_train, dst_train)
        torch.save(updated_val, dst_val)
        torch.save(updated_test, dst_test)

        print(
            f"[Fold {fold_id}] Done. k={config.k_neighbors}, metric={config.distance_metric}, "
            f"search={config.search_split}, normalize={config.normalize}. "
            f"Train samples={len(train_payload['labels'])}.",
            flush=True,
        )


if __name__ == "__main__":
    main()
