from typing import Dict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from torch_uncertainty.metrics import AURC


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def ensure_ndarray(preds) -> np.ndarray:
    if isinstance(preds, (list, tuple)):
        try:
            preds = np.concatenate([np.asarray(p) for p in preds], axis=0)
        except Exception:
            preds = np.asarray(preds[0])
    else:
        preds = np.asarray(preds)
    return preds


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    ece_bins: int = 100,
    pos_ece_bins: int = 20,
    f1_threshold: float = 0.5,
    f1_find_best: bool = False,
    f1_grid_step: float = 0.01,
) -> Dict[str, float]:
    predictions = ensure_ndarray(predictions)
    probs_heads = softmax_np(predictions, axis=-1)[..., 1]
    mean_prob = probs_heads.mean(axis=1)
    var_prob = probs_heads.var(axis=1)
    labels = labels.astype(int)

    eps = 1e-12
    nll = -np.mean(labels * np.log(mean_prob + eps) + (1 - labels) * np.log(1 - mean_prob + eps))
    try:
        auroc = roc_auc_score(labels, mean_prob)
    except ValueError:
        auroc = float("nan")
    try:
        auprc = average_precision_score(labels, mean_prob)
    except ValueError:
        auprc = float("nan")

    best_thr = f1_threshold
    if f1_find_best:
        grid = np.arange(0.0, 1.0 + 1e-8, f1_grid_step)
        f1_vals = []
        for thr in grid:
            f1_vals.append(f1_score(labels, (mean_prob >= thr).astype(int)))
        idx = int(np.argmax(f1_vals))
        best_thr = float(grid[idx])
        f1_best = float(f1_vals[idx])
        preds_bin = (mean_prob >= best_thr).astype(int)
    else:
        preds_bin = (mean_prob >= f1_threshold).astype(int)
    acc = (preds_bin == labels).mean()
    try:
        f1 = f1_score(labels, preds_bin)
    except Exception:
        f1 = float("nan")

    def _ece_bin(probs: np.ndarray, y: np.ndarray, n_bins: int = 30) -> float:
        # Numerical stability: avoid extreme 0/1 causing bin-boundary jitter
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        conf = np.maximum(probs, 1.0 - probs)
        pred = (probs >= 0.5).astype(int)
        correct = (pred == y).astype(float)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
            if mask.any():
                acc_bin = correct[mask].mean()
                conf_bin = conf[mask].mean()
                ece += (mask.mean()) * abs(acc_bin - conf_bin)
        return float(ece)

    def _adaptive_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 30) -> float:
        """
        Adaptive Expected Calibration Error:
        - Sort by confidence, then split into ``n_bins`` equal-frequency bins;
        - Per bin, use |mean(conf) - mean(correctness)|;
        - Weight by the fraction of samples in that bin.
        """
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        conf = np.maximum(probs, 1.0 - probs)
        pred = (probs >= 0.5).astype(int)
        correct = (pred == y).astype(float)
        n = conf.shape[0]
        if n == 0:
            return float("nan")
        bins = max(1, min(int(n_bins), n))
        order = np.argsort(conf)
        conf_sorted = conf[order]
        correct_sorted = correct[order]
        base = n // bins
        rem = n % bins
        start = 0
        aece_val = 0.0
        for i in range(bins):
            size = base + (1 if i < rem else 0)
            end = min(n, start + size)
            if end <= start:
                continue
            conf_bin = conf_sorted[start:end]
            correct_bin = correct_sorted[start:end]
            weight = (end - start) / n
            acc_bin = correct_bin.mean()
            conf_mean = conf_bin.mean()
            aece_val += weight * abs(acc_bin - conf_mean)
            start = end
            if start >= n:
                break
        return float(aece_val)

    ece = _ece_bin(mean_prob, labels, n_bins=ece_bins)
    aece = _adaptive_ece(mean_prob, labels, n_bins=ece_bins)

    def _ece_pos_equal_freq(probs: np.ndarray, y: np.ndarray, n_bins: int = 20, threshold: float = None) -> float:
        """
        Positive ECE (using positive-class probability):
        - If ``threshold`` is provided, evaluate only on the subset predicted positive (probs >= threshold).
        - Equal-frequency bins: sort (subset) samples by probability, split as evenly as possible into ``n_bins`` bins.
        - Per bin, use mean(p) as confidence and mean(y) as accuracy, weighted by sample fraction.
        """
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        # Optional: evaluate only on samples predicted positive
        if threshold is not None:
            mask = (probs >= threshold)
            probs = probs[mask]
            y = y[mask]
        n_sub = len(probs)
        if n_sub == 0:
            print(f"[metrics] ece_pos: no predicted positives at threshold={threshold}")
            return float("nan")
        order = np.argsort(probs)
        probs_sorted = probs[order]
        y_sorted = y[order]
        bins = []
        base = n_sub // n_bins
        rem = n_sub % n_bins
        start = 0
        for i in range(n_bins):
            size = base + (1 if i < rem else 0)
            end = start + size
            if start >= n_sub:
                break
            bins.append((start, min(end, n_sub)))
            start = end
        ece_pf = 0.0
        for s, e in bins:
            p_bin = probs_sorted[s:e]
            y_bin = y_sorted[s:e]
            if len(p_bin) == 0:
                continue
            conf_bin = float(p_bin.mean())
            acc_bin = float(y_bin.mean())
            ece_pf += (len(p_bin) / n_sub) * abs(acc_bin - conf_bin)
        return float(ece_pf)

    # ========== Non-standard ECE variant (commented out) ==========
    # Reason: _ece_p_equal_width is neither standard ECE nor ECE Positive
    # - It uses positive-class probability p as confidence (not max(p, 1 - p))
    # - But evaluates on all samples (not only those predicted positive)
    # - This is a custom variant and does not match standard definitions in the literature
    # Uncomment the block below if you need it
    # def _ece_p_equal_width(probs: np.ndarray, y: np.ndarray, n_bins: int = 30) -> float:
    #     """
    #     Full-dataset ECE based on positive-class probability p (differs from standard ECE conf):
    #     - Equal-width bins;
    #     - Per bin: acc = mean(y), conf = mean(p);
    #     - Weighted by sample fraction.
    #     """
    #     p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    #     bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    #     ece_val = 0.0
    #     for i in range(n_bins):
    #         lo, hi = bin_edges[i], bin_edges[i + 1]
    #         mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
    #         if mask.any():
    #             acc_bin = float(y[mask].mean())
    #             conf_bin = float(p[mask].mean())
    #             ece_val += float(mask.mean()) * abs(acc_bin - conf_bin)
    #     return float(ece_val)
    # ================================================

    # Positive ECE: report two threshold settings to avoid "threshold leakage" ambiguity
    # - ece_pos_best: best F1 threshold found on validation (may leak slightly)
    # - ece_pos_fixed05: fixed 0.5 threshold (no leakage, fairer)
    ece_pos_best = _ece_pos_equal_freq(mean_prob, labels, n_bins=pos_ece_bins, threshold=best_thr)
    ece_pos_fixed05 = _ece_pos_equal_freq(mean_prob, labels, n_bins=pos_ece_bins, threshold=0.5)

    # Non-standard ECE variant commented out
    # ece_p = _ece_p_equal_width(mean_prob, labels, n_bins=ece_bins)

    brier = np.mean((mean_prob - labels) ** 2)
    entropy = -(mean_prob * np.log(mean_prob + eps) + (1 - mean_prob) * np.log(1 - mean_prob + eps))

    probs_2d = np.stack([1.0 - mean_prob, mean_prob], axis=1).astype(np.float32)
    aurc_metric = AURC()
    aurc_metric.update(torch.from_numpy(probs_2d), torch.from_numpy(labels.astype(np.int64)))
    aurc = float(aurc_metric.compute().item())

    out: Dict[str, float] = {
        "accuracy": float(acc),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "brier": float(brier),
        "cls_NLL": float(nll),
        "entropy": float(entropy.mean()),
        # Naming: mean uncertainty (mean of multi-head probability variance)
        "avg_uncertainty": float(var_prob.mean()),
        "AURC": float(aurc),
        "f1": float(f1),
        # Standard ECE (confidence = max(p, 1 - p), all samples, equal-width bins)
        "ece": float(ece),
        "aECE": float(aece),
        # Non-standard variant removed (formerly ece_p)
        # "ece_p": float(ece_p),
        # ECE Positive (confidence = p, only predicted-positive samples, equal-frequency bins)
        # - ece_pos: kept for backward compatibility; same as ece_pos_best
        # - ece_pos_best: filter samples with best F1 threshold (may leak slightly)
        # - ece_pos_fixed_0.5: filter with fixed 0.5 threshold (no leakage; preferred for papers)
        "ece_pos": float(ece_pos_best),
        "ece_pos_best": float(ece_pos_best),
        "ece_pos_fixed_0.5": float(ece_pos_fixed05),
    }
    if f1_find_best:
        out["best_f1"] = float(f1_best)
        out["best_f1_threshold"] = float(best_thr)
    else:
        out["f1_threshold"] = float(best_thr)
    return out


