import json
import os
from typing import Any, Dict, List

import numpy as np

from .metrics import softmax_np, ensure_ndarray


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_val_history_from_logs(log_history: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    try:
        for rec in log_history:
            if any(k.startswith("eval_") for k in rec.keys()):
                vals = {k: float(v) for k, v in rec.items() if k.startswith("eval_") and isinstance(v, (int, float))}
                step = int(rec.get("step", 0))
                epoch = float(rec.get("epoch", 0.0) or 0.0)
                vals.update({"step": step, "epoch": epoch})
                records.append(vals)
    except Exception:
        return []
    return records


def save_test_predictions_json(eval_root: str, fold_id: int, test_out) -> None:
    probs_heads = softmax_np(ensure_ndarray(test_out.predictions), axis=-1)[..., 1]
    mean_prob = probs_heads.mean(axis=1)
    var_prob = probs_heads.var(axis=1)
    pred_records = [
        {"prob_mean": float(p), "prob_var": float(v), "label": int(l)}
        for p, v, l in zip(mean_prob, var_prob, test_out.label_ids.astype(int))
    ]
    out_path = os.path.join(eval_root, f"fold_{fold_id}_test_predictions.json")
    write_json(out_path, pred_records)


def save_test_details_json(eval_root: str, fold_id: int, label_col: str, test_meta_df, test_out) -> None:
    probs_heads = softmax_np(ensure_ndarray(test_out.predictions), axis=-1)[..., 1]
    mean_prob = probs_heads.mean(axis=1)
    var_prob = probs_heads.var(axis=1)
    test_meta_local = test_meta_df.copy()
    test_meta_local["prediction"] = (mean_prob >= 0.5).astype(int)
    test_meta_local["P"] = mean_prob.astype(float)
    test_meta_local["probability"] = mean_prob.astype(float)
    test_meta_local["uncertainty"] = var_prob.astype(float)
    test_meta_local["label_name"] = label_col
    test_meta_local["fold_id"] = fold_id
    details_records = test_meta_local.to_dict(orient="records")
    details_path = os.path.join(eval_root, f"fold_{fold_id}_test_details.json")
    write_json(details_path, details_records)


def combine_fold_summaries(eval_root: str, num_folds: int) -> None:
    try:
        combined: Dict[str, Any] = {}
        val_path = os.path.join(eval_root, "val_summary.json")
        test_path = os.path.join(eval_root, "test_summary.json")
        if os.path.exists(val_path):
            with open(val_path, "r", encoding="utf-8") as f:
                combined["val"] = json.load(f)
        if os.path.exists(test_path):
            with open(test_path, "r", encoding="utf-8") as f:
                combined["test"] = json.load(f)
        combined["num_folds"] = num_folds
        write_json(os.path.join(eval_root, "folds_summary.json"), combined)
    except Exception:
        pass


