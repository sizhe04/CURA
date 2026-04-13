import os
from typing import Any, Dict, Optional

import numpy as np

from .metrics import softmax_np, ensure_ndarray


def init_wandb_run(args: Any, *, base_root: str, sanitized_model: str, label_col: str, fold_id: int):
    if not getattr(args, "use_wandb", False):
        return None
    try:
        import wandb  # type: ignore

        group_name = f"{sanitized_model}-{label_col}"
        base_name = getattr(args, "wandb_run_name", None) or f"{sanitized_model}_{label_col}"
        run_name = f"{base_name}_fold{fold_id}"

        local_wandb_dir = os.path.join(base_root, "wandb")
        os.makedirs(local_wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = local_wandb_dir

        wandb_kwargs: Dict[str, Any] = {
            "name": run_name,
            "group": group_name,
            "dir": local_wandb_dir,
        }
        if args.wandb_project is not None:
            wandb_kwargs["project"] = args.wandb_project
        if args.wandb_entity is not None:
            wandb_kwargs["entity"] = args.wandb_entity

        try:
            wandb_run = wandb.init(
                **wandb_kwargs,
                finish_previous=True,
                return_previous=False,
                resume="never",
            )
        except TypeError:
            wandb_run = wandb.init(**wandb_kwargs)

        # Compatible with Stage-1 and Stage-2: only write fields that exist to avoid AttributeError
        cfg = {
            "model_name": getattr(args, "model_name", None),
            "num_heads": getattr(args, "num_heads", None),
            "head_hidden_dim": getattr(args, "head_hidden_dim", None),
            "lambda_cal": getattr(args, "lambda_cal", None),
            "dropout": getattr(args, "dropout", None),
            "max_length": getattr(args, "max_length", None),
            "learning_rate": getattr(args, "learning_rate", None),
            "weight_decay": getattr(args, "weight_decay", None),
            "warmup_steps": getattr(args, "warmup_steps", None),
            "per_device_train_batch_size": getattr(args, "per_device_train_batch_size", None),
            "per_device_eval_batch_size": getattr(args, "per_device_eval_batch_size", None),
            "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", None),
            "seed": getattr(args, "seed", None),
            "label": label_col,
            "fold_id": fold_id,
        }
        # Drop None values to keep existing plotting and logging behavior unchanged
        cfg = {k: v for k, v in cfg.items() if v is not None}
        wandb.config.update(cfg, allow_val_change=True)
        return wandb_run
    except Exception as e:
        print(f"[wandb] Initialization failed (continuing without wandb): {e}")
        return None


def log_test_curves(wandb_run, *, label_col: str, fold_id: int, test_out) -> None:
    if wandb_run is None:
        return
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve
        import wandb  # type: ignore

        probs_heads = softmax_np(ensure_ndarray(test_out.predictions), axis=-1)[..., 1]
        mean_prob = probs_heads.mean(axis=1)
        labels_np = test_out.label_ids.astype(int)

        fpr, tpr, _ = roc_curve(labels_np, mean_prob)
        precision, recall, _ = precision_recall_curve(labels_np, mean_prob)

        roc_table = wandb.Table(data=[[float(x), float(y)] for x, y in zip(fpr, tpr)], columns=["fpr", "tpr"])
        pr_table = wandb.Table(data=[[float(x), float(y)] for x, y in zip(recall, precision)], columns=["recall", "precision"])

        eps = 1e-12
        entropy = -(mean_prob * np.log(mean_prob + eps) + (1 - mean_prob) * np.log(1 - mean_prob + eps))
        order = np.argsort(entropy)[::-1]
        sorted_labels = labels_np[order]
        coverage_list, risk_list = [], []
        incorrect = 0
        for i, lab in enumerate(sorted_labels, start=1):
            pred_bin = int(mean_prob[order][i - 1] >= 0.5)
            if pred_bin != lab:
                incorrect += 1
            coverage = i / len(sorted_labels)
            risk = incorrect / i
            coverage_list.append(float(coverage))
            risk_list.append(float(risk))
        rc_table = wandb.Table(data=list(zip(coverage_list, risk_list)), columns=["coverage", "risk"])

        wandb.log({
            f"{label_col}/fold_{fold_id}/ROC": wandb.plot.line(roc_table, "fpr", "tpr", title=f"ROC {label_col} fold {fold_id}"),
            f"{label_col}/fold_{fold_id}/PR": wandb.plot.line(pr_table, "recall", "precision", title=f"PR {label_col} fold {fold_id}"),
            f"{label_col}/fold_{fold_id}/Risk-Coverage": wandb.plot.line(rc_table, "coverage", "risk", title=f"Risk-Coverage {label_col} fold {fold_id}"),
        })
    except Exception as e:
        print(f"[wandb] Log visualization failed: {e}")


def finish_wandb(wandb_run) -> None:
    if wandb_run is None:
        return
    try:
        import wandb  # type: ignore
        wandb_run.finish()
    except Exception:
        pass


