#!/usr/bin/env python
"""
On the Stage-2 multi-head model, compute AUROC/AUPRC curves for selective prediction
from coverage/threshold values (fraction of most-certain samples retained).

Conventions:
- Each threshold t in [0, 1] means “retain the t×100% lowest-uncertainty samples; treat the rest as abstentions”;
- e.g. t=0.05 -> answer only the 5% most certain samples; t=0.4 -> answer only the 40% most certain;
- t=0 means abstain on all samples; AUROC/AUPRC are set to 0.
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch

from utilities.metrics import compute_metrics
from utilities.selective_eval import (
    FoldThresholdResult,
    build_model_from_checkpoint,
    compute_threshold_table,
    compute_uncertainty_arrays,
    ensure_evaluate_dir,
    load_fold_dataset,
    parse_thresholds,
    resolve_embedding_source,
    resolve_embeddings_root,
    resolve_pooling,
    run_model_on_dataset,
    summarize_thresholds,
)


def _parse_thresholds(threshold_str: str) -> List[float]:
    if not threshold_str:
        return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    parts = [p.strip() for p in threshold_str.split(",") if p.strip()]
    return parse_thresholds(float(p) for p in parts)


def _available_folds(checkpoint_root: Path) -> List[int]:
    folds = []
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_root}")
    for child in checkpoint_root.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("fold_"):
            try:
                folds.append(int(child.name.split("_")[1]))
            except Exception:
                continue
    if not folds:
        raise ValueError(f"No fold_* subdirectories found under {checkpoint_root}.")
    return sorted(folds)


def _parse_metrics(metrics_str: str) -> List[str]:
    if not metrics_str:
        return ["auroc", "auprc"]
    parts = [m.strip() for m in metrics_str.split(",") if m.strip()]
    return parts or ["auroc", "auprc"]


def _decide_variants(experiment_dir: Path) -> List[str]:
    """
    Decide which uncertainty variants to evaluate for this experiment:
      - Active-none* or name contains "rdm": return ["mi", "var"]
      - Name contains "epis-var": return ["var"]
      - Otherwise default to ["mi"]
    """
    name = experiment_dir.name.lower()
    if "active-none" in name or "rdm" in name:
        return ["mi", "var"]
    if "epis-var" in name:
        return ["var"]
    return ["mi"]


def evaluate_one_fold(
    *,
    fold_id: int,
    experiment_dir: Path,
    checkpoint_root: Path,
    embeddings_root: Path,
    pooling: str,
    embedding_source_override: Optional[str],
    thresholds: List[float],
    device: torch.device,
    batch_size: int,
    metrics: List[str],
) -> FoldThresholdResult:
    checkpoint_dir = checkpoint_root / f"fold_{fold_id}"
    _, test_pt = resolve_embedding_source(
        embeddings_root,
        fold_id,
        "test",
        pooling,
        override=embedding_source_override,
    )
    print("----------------------------------------------------------------------")
    print(f"[fold {fold_id}] Loading checkpoint={checkpoint_dir}")
    print(f"[fold {fold_id}] Test embeddings: {test_pt}")
    dataset = load_fold_dataset(test_pt)
    input_dim = int(dataset.features.shape[1])
    model = build_model_from_checkpoint(checkpoint_dir, input_dim=input_dim, device=device)
    predictions, labels = run_model_on_dataset(model, dataset, batch_size=batch_size, device=device)
    base_metrics = compute_metrics(predictions, labels, f1_find_best=False)
    comps = compute_uncertainty_arrays(predictions)
    variants = _decide_variants(experiment_dir)
    per_variant: Dict[str, List[Dict[str, float]]] = {}
    for v in variants:
        if v not in comps:
            continue
        unc = comps[v]
        per_variant[v] = compute_threshold_table(
            predictions, labels, unc, thresholds, base_metrics, metrics, variant=v
        )
    return FoldThresholdResult(
        fold_id=fold_id,
        base_metrics=base_metrics,
        per_variant=per_variant,
        num_samples=len(labels),
        variants=variants,
    )


def save_fold_result(eval_dir: Path, fold_result: FoldThresholdResult, extra_payload: dict) -> None:
    payload = {
        "fold_id": fold_result.fold_id,
        "num_samples": fold_result.num_samples,
        "base_metrics": fold_result.base_metrics,
        "thresholds": fold_result.per_variant,
        "variants": fold_result.variants,
    }
    payload.update(extra_payload)
    out_path = eval_dir / f"fold_{fold_result.fold_id}_threshold.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser("Stage-2 selective evaluation (coverage/threshold curves)")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Stage-2 experiment directory (Active-... folder)",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="",
        help=(
            "Comma-separated coverage/threshold list; t retains the t×100% lowest-uncertainty samples, "
            "e.g. 0.0,0.05,0.10,...,1.0; t=0 means abstain on all."
        ),
    )
    parser.add_argument(
        "--embeddings_root",
        type=str,
        default=None,
        help="Stage-1 checkpoint/embeddings root (inferred automatically if omitted)",
    )
    parser.add_argument(
        "--embedding_source",
        type=str,
        default=None,
        help="embeddings or embeddings_last; auto-detected by default",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default=None,
        help="Pooling strategy matching Stage-1 (default: read from hparams_stage1.json)",
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Inference batch size")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu; auto-detected if omitted")
    parser.add_argument("--metrics", type=str, default="auroc,auprc", help="Comma-separated metric names, e.g. auroc,auprc")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"experiment_dir does not exist: {experiment_dir}")
    thresholds = _parse_thresholds(args.thresholds)
    embeddings_root = resolve_embeddings_root(experiment_dir, args.embeddings_root)
    pooling = resolve_pooling(experiment_dir, args.pooling)
    embedding_source = args.embedding_source
    checkpoint_root = experiment_dir / "checkpoint"
    folds = _available_folds(checkpoint_root)
    if args.device:
        dev_str = str(args.device).strip()
        if dev_str.isdigit():
            device = torch.device(f"cuda:{dev_str}")
        elif dev_str.lower() == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device(dev_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_fold_results: List[FoldThresholdResult] = []
    eval_dir = ensure_evaluate_dir(experiment_dir)
    metrics = _parse_metrics(args.metrics)

    extra = {
        "experiment_dir": str(experiment_dir),
        "embeddings_root": str(embeddings_root),
        "pooling": pooling,
        "embedding_source": embedding_source,
        "thresholds_requested": thresholds,
        "metrics": metrics,
    }

    for fold_id in folds:
        fold_res = evaluate_one_fold(
            fold_id=fold_id,
            experiment_dir=experiment_dir,
            checkpoint_root=checkpoint_root,
            embeddings_root=embeddings_root,
            pooling=pooling,
            embedding_source_override=embedding_source,
            thresholds=thresholds,
            device=device,
            batch_size=args.batch_size,
            metrics=metrics,
        )
        save_fold_result(eval_dir, fold_res, extra_payload=extra)
        per_fold_results.append(fold_res)
        print(f"[fold {fold_id}] Selective evaluation done; wrote {eval_dir / f'fold_{fold_id}_threshold.json'}")

    summary = summarize_thresholds(per_fold_results, thresholds, metrics)
    summary_payload = {
        **extra,
        **summary,
        "num_folds": len(per_fold_results),
    }
    summary_path = eval_dir / "threshold_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    print(f"[summary] Macro-averaged coverage/threshold results written to {summary_path}")


if __name__ == "__main__":
    main()
