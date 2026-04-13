import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_embeddings import EmbeddingDataset
from modeling_multihead_embedding import MultiHeadBinaryOnEmbedding
from .embeddings_io import load_embeddings_pt
from .metrics import compute_metrics


@dataclass
class FoldThresholdResult:
    fold_id: int
    base_metrics: Dict[str, float]
    per_variant: Dict[str, List[Dict[str, float]]]
    num_samples: int
    variants: List[str]


def parse_thresholds(thresholds: Sequence[float]) -> List[float]:
    uniq = sorted({float(t) for t in thresholds})
    return uniq


def resolve_embeddings_root(experiment_dir: Path, override: Optional[str]) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    # Stage-2/<exp>/ -> Stage-2 parent is ".../<label>/Stage2"
    label_dir = experiment_dir.parent.parent
    candidate = label_dir / "checkpoint"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Cannot infer Stage-1 embeddings root; candidate path does not exist: {candidate}"
        )
    return candidate


def resolve_pooling(experiment_dir: Path, override: Optional[str]) -> str:
    if override:
        return override
    label_dir = experiment_dir.parent.parent
    hparams = label_dir / "hparams_stage1.json"
    if not hparams.exists():
        raise FileNotFoundError(
            f"--pooling not provided and {hparams} is missing; cannot infer Stage-1 pooling strategy."
        )
    with hparams.open("r", encoding="utf-8") as f:
        data = json.load(f)
    pooling = data.get("pooling") or data.get("POOLING")
    if not pooling:
        raise ValueError(f"{hparams} is missing a pooling field; cannot infer.")
    return str(pooling)


def resolve_embedding_source(
    embeddings_root: Path,
    fold_id: int,
    split: str,
    pooling: str,
    override: Optional[str] = None,
) -> Tuple[str, Path]:
    """
    Returns (source_name, file_path).
    """
    sources_to_try: List[str] = []
    if override:
        sources_to_try = [override]
    else:
        sources_to_try = ["embeddings_last", "embeddings"]
    base = f"fold_{fold_id}_{split}_pool-{pooling}"
    for source in sources_to_try:
        fold_dir = embeddings_root / f"fold_{fold_id}" / source
        candidates: List[Path] = []
        if source == "embeddings_last":
            candidates = [
                fold_dir / f"{base}_last.pt",
                fold_dir / f"{base}_best_last.pt",
            ]
        else:
            candidates = [
                fold_dir / f"{base}_best.pt",
                fold_dir / f"{base}_best_last.pt",
            ]
        candidates.append(fold_dir / f"{base}.pt")
        for cand in candidates:
            if cand.exists():
                return source, cand
    tried = ", ".join(sources_to_try)
    raise FileNotFoundError(
        f"No .pt file found under {embeddings_root} for fold_{fold_id} split={split}. "
        f"Tried source={tried}, pooling={pooling}"
    )


def infer_model_shape(state_dict: Dict[str, torch.Tensor], input_dim: int) -> Tuple[int, Optional[int]]:
    head_indices = set()
    for key in state_dict.keys():
        if key.startswith("heads."):
            try:
                head_idx = int(key.split(".")[1])
                head_indices.add(head_idx)
            except Exception:
                continue
    if not head_indices:
        raise ValueError("No heads.* weights found in state_dict; cannot infer multi-head structure.")
    num_heads = max(head_indices) + 1

    hidden_dim: Optional[int] = None
    linear_key = "heads.0.net.weight"
    seq_key = "heads.0.net.0.weight"
    if seq_key in state_dict:
        hidden_dim = int(state_dict[seq_key].shape[0])
    elif linear_key in state_dict:
        hidden_dim = None
    else:
        # Also handle Sequential layers that do not start at index 0
        for key in state_dict.keys():
            if key.startswith("heads.0.net.") and key.endswith(".weight"):
                weight = state_dict[key]
                hidden_dim = int(weight.shape[0])
                break
    return num_heads, hidden_dim


def build_model_from_checkpoint(
    checkpoint_dir: Path,
    input_dim: int,
    device: torch.device,
) -> MultiHeadBinaryOnEmbedding:
    weight_path = checkpoint_dir / "pytorch_model.bin"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing model weights: {weight_path}")
    state_dict = torch.load(weight_path, map_location="cpu")
    num_heads, hidden_dim = infer_model_shape(state_dict, input_dim)
    model = MultiHeadBinaryOnEmbedding(
        input_dim=input_dim,
        num_heads=num_heads,
        head_hidden_dim=hidden_dim,
        dropout=0.0,
    )
    missing = model.load_state_dict(state_dict, strict=False)
    if missing.missing_keys:
        raise RuntimeError(f"Failed to load {checkpoint_dir}; missing_keys={missing.missing_keys}")
    model.to(device)
    model.eval()
    return model


def load_fold_dataset(pt_path: Path) -> EmbeddingDataset:
    payload = load_embeddings_pt(str(pt_path))
    return EmbeddingDataset(payload)


def run_model_on_dataset(
    model: MultiHeadBinaryOnEmbedding,
    dataset: EmbeddingDataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["labels"]
            out = model(features=features)
            logits_list.append(out["logits"].detach().cpu())
            labels_list.append(labels.detach().cpu())
    logits = torch.cat(logits_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    return logits, labels


def compute_uncertainty_arrays(predictions: np.ndarray) -> Dict[str, np.ndarray]:
    """
    During evaluation, compute multi-head probabilities consistently:
      - p_heads: softmax(logits)[..., 1], shape (N, K)
      - p_bar: mean over the head dimension

    Returns two components:
      - "mi": mutual-information-style U_epis_raw_MI = H(p_bar) - mean(H(p_heads)),
        where H is binary entropy with values clipped to [eps, 1 - eps]; not divided by log(2),
        preserving the raw scale
      - "var": variance-style U_epis_raw_Var = var(p_heads, dim=head)

    Note: downstream code picks one or both depending on the experiment for thresholding.
    """
    logits = torch.as_tensor(predictions)          # (N, K, 2)
    probs_heads = torch.softmax(logits, dim=-1)[..., 1]  # (N, K)
    p_bar = probs_heads.mean(dim=1)                # (N,)
    eps = 1e-12
    p_bar_clip = p_bar.clamp(min=eps, max=1.0 - eps)
    p_heads_clip = probs_heads.clamp(min=eps, max=1.0 - eps)

    def _H(q: torch.Tensor) -> torch.Tensor:
        return -(q * torch.log(q) + (1.0 - q) * torch.log(1.0 - q))

    H_total = _H(p_bar_clip)                     # (N,)
    H_heads = _H(p_heads_clip).mean(dim=1)       # (N,)
    U_mi = H_total - H_heads                     # (N,) not divided by log2

    U_var = (probs_heads - p_bar.unsqueeze(-1)) ** 2
    U_var = U_var.mean(dim=1)                    # (N,) ∈ [0, 0.25]

    return {
        "mi": U_mi.detach().cpu().numpy(),
        "var": U_var.detach().cpu().numpy(),
    }


def compute_threshold_table(
    predictions: np.ndarray,
    labels: np.ndarray,
    uncertainties: np.ndarray,
    thresholds: Sequence[float],
    base_metrics: Dict[str, float],
    metrics_to_record: Sequence[str],
    variant: str,
) -> List[Dict[str, float]]:
    """
    Selective evaluation where coverage/threshold equals the fraction of most-certain samples kept.

    Arguments:
    - predictions: multi-head logits, shape (N, K, 2)
    - labels: ground-truth labels, shape (N,)
    - uncertainties: per-sample uncertainty scalar u_i, shape (N,)
    - thresholds: coverage/threshold values t in [0, 1], meaning:
        * t = 0.05 -> answer only the 5% of samples with lowest uncertainty;
        * t = 0.40 -> answer only the 40% with lowest uncertainty (reject the top 60% most uncertain);
        * t = 0    -> reject all samples; no predictions.

      Here the threshold value is the target retained-sample coverage; the name ``threshold``
      is kept for compatibility with existing JSON/CSV fields.

    Implementation:
    - Sort by uncertainty ascending (lower is more certain).
    - For t > 0, take the first round(t * N) most-certain samples (at least 1, at most N) and
      compute AUROC/AUPRC, etc.
    - For t <= 0, treat as full abstention: coverage = 0, set AUROC/AUPRC to 0 without further computation.
    """
    results: List[Dict[str, float]] = []
    total = float(len(labels))
    if total <= 0:
        # Edge case: no samples -> return an empty list
        return results

    # Sort by uncertainty ascending (lower is more certain)
    u = np.asarray(uncertainties, dtype=np.float64)
    order = np.argsort(u)  # shape: (N,)
    preds_sorted = np.asarray(predictions)[order]
    labels_sorted = np.asarray(labels)[order]
    n_total = int(total)

    for thr in thresholds:
        t = float(thr)

        # threshold=0: full abstention; set AUROC/AUPRC to 0
        if t <= 0.0:
            row: Dict[str, float] = {
                "threshold": t,
                "coverage": 0.0,
                "num_samples": 0,
                "variant": variant,
            }
            for metric in metrics_to_record:
                # For coverage=0, set AUC-style metrics to 0
                row[metric] = 0.0
            results.append(row)
            continue

        # t > 0: answer only the top t * 100% most-certain samples
        # Per spec: n_keep = round(t * N_fold), at least 1
        frac = t
        # Allow t slightly outside [0, 1]; effective coverage is still clipped to [0, 1]
        if frac > 1.0:
            frac = 1.0
        n_keep = int(round(frac * n_total))
        if n_keep < 1:
            n_keep = 1
        if n_keep > n_total:
            n_keep = n_total

        keep_idx = slice(0, n_keep)
        preds_sel = preds_sorted[keep_idx]
        labels_sel = labels_sorted[keep_idx]
        metrics = compute_metrics(preds_sel, labels_sel, f1_find_best=False)
        coverage = float(n_keep) / float(n_total)

        row = {
            "threshold": t,  # Here ``threshold`` is the target coverage
            "coverage": coverage,
            "num_samples": n_keep,
            "variant": variant,
        }
        for metric in metrics_to_record:
            row[metric] = metrics.get(metric)
        results.append(row)

    for row in results:
        for metric in metrics_to_record:
            row[f"baseline_{metric}"] = base_metrics.get(metric)
    return results


def summarize_thresholds(
    per_fold: List[FoldThresholdResult],
    thresholds: Sequence[float],
    metrics: Sequence[str],
) -> Dict[str, List[Dict[str, float]]]:
    summaries: Dict[str, List[Dict[str, float]]] = {}
    # Collect all variants seen across folds
    variant_set = set()
    for fold_res in per_fold:
        variant_set.update(fold_res.variants)
    for variant in sorted(variant_set):
        summary: List[Dict[str, float]] = []
        threshold_to_entries: Dict[float, List[Dict[str, float]]] = {thr: [] for thr in thresholds}
        for fold_res in per_fold:
            rows = fold_res.per_variant.get(variant, [])
            for row in rows:
                threshold_to_entries[row["threshold"]].append(row)
        for thr in thresholds:
            rows = threshold_to_entries.get(thr, [])
            def collect(key: str) -> Tuple[float, float, int]:
                vals = [r.get(key) for r in rows if r.get(key) is not None]
                if not vals:
                    return float("nan"), float("nan"), 0
                arr = np.asarray(vals, dtype=np.float64)
                return float(arr.mean()), float(arr.std()), len(vals)
            cov_mean, cov_std, _ = collect("coverage")
            summary_row = {
                "threshold": thr,
                "coverage_mean": cov_mean,
                "coverage_std": cov_std,
                "variant": variant,
            }
            for metric in metrics:
                mean_val, std_val, support = collect(metric)
                summary_row[f"{metric}_mean"] = mean_val
                summary_row[f"{metric}_std"] = std_val
                summary_row[f"{metric}_support"] = support
            summary.append(summary_row)
        summaries[variant] = summary
    return {"thresholds": summaries}


def ensure_evaluate_dir(experiment_dir: Path) -> Path:
    eval_dir = experiment_dir / "evaluate"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir
