from __future__ import annotations

import argparse
import json
import os
import sys
import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback

from data_embeddings import EmbeddingDataset
from modeling_multihead_embedding import MultiHeadBinaryOnEmbedding
from utilities.metrics import compute_metrics
from utilities.training_args import build_training_arguments
from utilities.embeddings_io import load_embeddings_pt
from utilities.repr_uncert import ReprUncertConfig
from utilities.wandb_utils import init_wandb_run, log_test_curves, finish_wandb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Stage-2: Train multi-head classifier on frozen embeddings (CURA)")
    # Input embeddings
    p.add_argument("--embeddings_root", type=str, required=True, help="Path to stage-1 checkpoint directory that contains fold_*/embeddings/")
    p.add_argument(
        "--pooling",
        type=str,
        choices=["cls", "pooler", "mean", "last_hidden_state", "last_token"],
        default="cls",
        help="Must match Stage-1 pooling: cls | pooler | mean | last_hidden_state | last_token",
    )
    p.add_argument(
        "--embedding_source",
        type=str,
        default="embeddings_last",
        help=(
            "Which Stage-1 embeddings folder to use under fold_*/. "
            "Supported: embeddings_last | embeddings | embeddings_last_llm_uncertain_K-<int> "
            "(or any string starting with 'embeddings_last_llm_uncertain')."
        ),
    )
    p.add_argument("--labels", type=str, nargs="+", default=["death_in_30"])
    p.add_argument("--model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT", help="For path naming only")
    # Multi-head params
    p.add_argument("--num_heads", type=int, default=32)
    p.add_argument("--head_hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    # L_ind: individual uncertainty calibration (paper's lambda_ind)
    p.add_argument("--lambda_cal", type=float, default=0.5)
    p.add_argument("--cal_alpha", type=float, default=1.0)
    p.add_argument("--u_min", type=float, default=0.05)
    p.add_argument("--u_max", type=float, default=0.95)
    # L_coh: cohort-aware risk alignment (paper's lambda_coh)
    p.add_argument(
        "--lambda_pop",
        type=float,
        default=0.01,
        help="Cohort-aware weight scale: w(x)=lambda_pop * H_hat(q(x)). Set 0 to disable.",
    )
    p.add_argument("--pop_k_neighbors", type=int, default=200, help="k for neighborhood q(x) in embedding space")
    p.add_argument("--pop_distance_metric", type=str, choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--pop_search_split", type=str, choices=["train", "train+val"], default="train+val")
    # Metrics
    p.add_argument("--ece_bins", type=int, default=100)
    p.add_argument("--f1_find_best", action="store_true")
    p.add_argument("--f1_threshold", type=float, default=0.5)
    p.add_argument("--f1_grid_step", type=float, default=0.01)
    p.add_argument("--pos_ece_bins", type=int, default=20)
    # Training
    p.add_argument("--output_root", type=str, default="./outputs_cv")
    p.add_argument("--num_train_epochs", type=int, default=50)
    p.add_argument("--per_device_train_batch_size", type=int, default=4096)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4096)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--fp16", action="store_true")
    # best model and save policies
    p.add_argument("--best_metric", type=str, default="eval_cls_NLL")
    p.add_argument("--greater_is_better", type=int, default=0)
    p.add_argument("--eval_strategy", type=str, choices=["epoch", "steps"], default="epoch")
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_strategy", type=str, choices=["auto", "epoch", "steps", "no"], default="auto")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=5)
    # early stop
    p.add_argument("--use_early_stopping", action="store_true")
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--early_stopping_threshold", type=float, default=0.001)
    # optional test eval during training
    p.add_argument("--test_eval_strategy", type=str, choices=["end_only", "epoch", "steps", "none"], default="end_only")
    p.add_argument("--test_eval_steps", type=int, default=2000)
    # wandb
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    # seed
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fold_ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional 1-based fold ids to run (e.g. 2 3 4 5). If unset, run all 5 folds.",
    )

    args = p.parse_args()

    src = str(getattr(args, "embedding_source", "embeddings_last"))
    if src in {"embeddings_last", "embeddings"}:
        pass
    elif src.startswith("embeddings_last_llm_uncertain"):
        pass
    else:
        raise ValueError(
            f"--embedding_source='{src}' is not supported. "
            f"Use embeddings_last | embeddings | embeddings_last_llm_uncertain_K-<int> "
            f"(or any prefix 'embeddings_last_llm_uncertain*')."
        )

    return args


def _log_wandb_direct(trainer, payload: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics directly to wandb with monotonic step."""
    if trainer is None or not payload:
        return
    wandb_run = getattr(trainer, "_wandb_run", None)
    if wandb_run is None:
        return
    try:
        current = getattr(wandb_run, "step", None)
    except Exception:
        current = None
    target_step: Optional[int] = step
    if current is not None:
        if target_step is None or target_step <= int(current):
            target_step = int(current) + 1
    try:
        wandb_run.log(payload, step=target_step)
    except Exception:
        pass


def _simple_collate(batch):
    feats = torch.stack([b["features"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    out = {"features": feats, "labels": labels}
    for k in ("neighbor_q", "neighbor_hhat", "neighbor_w", "neighbor_t"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def run_one_label(args: argparse.Namespace, label_col: str) -> None:
    label_dir = os.path.dirname(os.path.abspath(args.embeddings_root))

    early_tag = args.best_metric if args.use_early_stopping else "NA"
    pop_tag = ""
    try:
        lambda_pop = float(getattr(args, "lambda_pop", 0.0))
        if lambda_pop > 0.0:
            pop_tag = (
                f"_POP-{lambda_pop}"
                f"_K{int(getattr(args, 'pop_k_neighbors', 20))}"
                f"_{str(getattr(args, 'pop_search_split', 'train+val'))}"
            )
    except Exception:
        pop_tag = ""

    exp_dir = (
        f"Head-{args.num_heads}{pop_tag}_CAL-{args.lambda_cal}_"
        f"Ear-{early_tag}_Lr-{args.learning_rate}_Epo-{args.num_train_epochs}_Warm-{args.warmup_steps}"
    )
    base_root = os.path.join(label_dir, "Stage2", exp_dir)
    checkpoint_root = os.path.join(base_root, "checkpoint")
    eval_root = os.path.join(base_root, "evaluate")
    os.makedirs(checkpoint_root, exist_ok=True)
    os.makedirs(eval_root, exist_ok=True)
    sanitized_model = args.model_name.replace("/", "-")
    try:
        _orig_proj = os.environ.get("WANDB_PROJECT", None)
        if _orig_proj and (not _orig_proj.endswith(exp_dir)):
            os.environ["WANDB_PROJECT"] = f"{_orig_proj}_{exp_dir}"
    except Exception:
        pass

    def emb_path(fold: int, split: str) -> str:
        fold_dir = os.path.join(args.embeddings_root, f"fold_{fold}")
        src_name = str(args.embedding_source)
        src_dir = os.path.join(fold_dir, src_name)

        if (not os.path.isdir(src_dir)) and src_name.startswith("embeddings_last_llm_uncertain_K-"):
            try:
                k_str = src_name.split("K-")[-1]
                alt_name = f"embeddings_last_llm_uncertain_{k_str}"
                alt_dir = os.path.join(fold_dir, alt_name)
                if os.path.isdir(alt_dir):
                    print(
                        f"[Stage2] embeddings directory '{src_name}' not found for fold {fold}; "
                        f"falling back to '{alt_name}'.",
                        flush=True,
                    )
                    src_dir = alt_dir
            except Exception:
                pass

        base = f"fold_{fold}_{split}_pool-{args.pooling}"
        src = str(args.embedding_source)
        is_last_style = (src == "embeddings_last") or src.startswith("embeddings_last_llm_uncertain")

        if is_last_style:
            cands = [f"{base}_last.pt", f"{base}_best_last.pt"]
        else:
            cands = [f"{base}_best.pt", f"{base}_best_last.pt"]

        for name in cands:
            pth = os.path.join(src_dir, name)
            if os.path.exists(pth):
                return pth
        legacy = os.path.join(src_dir, f"{base}.pt")
        return legacy

    val_metrics_list: List[Dict[str, float]] = []
    test_metrics_list: List[Dict[str, float]] = []
    final_checkpoint_metrics_list: List[Dict[str, float]] = []
    train_samples_list: List[int] = []

    probe = load_embeddings_pt(emb_path(1, "train"))
    input_dim = int(probe["features"].shape[-1])
    try:
        meta = probe.get("meta", {}) if isinstance(probe, dict) else {}
        pooling_in_meta = meta.get("pooling", None)
        if pooling_in_meta is not None and str(pooling_in_meta) != str(args.pooling):
            raise ValueError(
                f"[Stage2] pooling mismatch: Stage-1 embeddings were saved with pooling='{pooling_in_meta}', "
                f"but Stage-2 was launched with --pooling='{args.pooling}'."
            )
    except Exception as e:
        print(f"[Stage2] Warning: failed to verify pooling consistency ({e})")

    selected_folds = None
    if getattr(args, "fold_ids", None):
        try:
            selected_folds = {int(x) for x in args.fold_ids}
        except Exception as e:
            print(f"[Stage2] Warning: invalid --fold_ids value {args.fold_ids}: {e}. Running all folds.", flush=True)
            selected_folds = None

    for fold_id in range(1, 6):
        if selected_folds is not None and fold_id not in selected_folds:
            print(f"[Stage2] Skip fold {fold_id} because it is not in --fold_ids.", flush=True)
            continue
        train_payload = load_embeddings_pt(emb_path(fold_id, "train"))
        val_payload = load_embeddings_pt(emb_path(fold_id, "val"))
        test_payload = load_embeddings_pt(emb_path(fold_id, "test"))

        # Neighbor-aware stats: q(x), H_hat(q), w(x), t(x) for L_coh
        try:
            lambda_pop = float(getattr(args, "lambda_pop", 0.0))
        except Exception:
            lambda_pop = 0.0
        if lambda_pop > 0.0:
            try:
                def _to_np(x):
                    if torch.is_tensor(x):
                        return x.detach().cpu().numpy()
                    return np.asarray(x)

                pop_search_split = str(getattr(args, "pop_search_split", "train+val"))
                if pop_search_split not in {"train", "train+val"}:
                    pop_search_split = "train+val"

                cfg_pop = ReprUncertConfig(
                    k_neighbors=int(getattr(args, "pop_k_neighbors", 20)),
                    distance_metric=str(getattr(args, "pop_distance_metric", "cosine")),
                    search_split=pop_search_split,
                    score_type="entropy",
                    normalize=True,
                )

                def _attach_neighbor_fields(payload: Dict[str, object], split_name: str) -> Dict[str, object]:
                    """Attach neighbor_q / neighbor_hhat / neighbor_w / neighbor_t from pre-computed embeddings."""
                    payload = dict(payload)
                    labels_np = _to_np(payload["labels"]).astype(np.int64)

                    has_offline = ("neighbor_q" in payload) and ("neighbor_hhat" in payload)
                    if not has_offline:
                        msg = (
                            f"[Stage2] ERROR: lambda_pop={lambda_pop} > 0, but fields 'neighbor_q' / 'neighbor_hhat' "
                            f"not found in split='{split_name}' embeddings.\n"
                            f"Please run run_stage1_repr_uncert.sh first, then set "
                            f"EMBEDDING_SOURCE='embeddings_last_llm_uncertain_K-<K>'."
                        )
                        print(msg, flush=True)
                        raise ValueError(msg)

                    q = _to_np(payload["neighbor_q"]).astype(np.float64)
                    hhat = _to_np(payload["neighbor_hhat"]).astype(np.float64)

                    y = labels_np.astype(np.float64)
                    w = lambda_pop * hhat
                    gamma = w / (1.0 + w)
                    t = (1.0 - gamma) * y + gamma * q

                    payload["neighbor_q"] = torch.tensor(q, dtype=torch.float)
                    payload["neighbor_hhat"] = torch.tensor(hhat, dtype=torch.float)
                    payload["neighbor_w"] = torch.tensor(w, dtype=torch.float)
                    payload["neighbor_t"] = torch.tensor(t, dtype=torch.float)

                    meta = dict(payload.get("meta", {}))
                    meta["neighbor_stats"] = {
                        "pop_k_neighbors": int(cfg_pop.k_neighbors),
                        "pop_distance_metric": str(cfg_pop.distance_metric),
                        "pop_search_split": str(cfg_pop.search_split),
                        "lambda_pop": float(lambda_pop),
                        "split": str(split_name),
                        "offline_qhhat": True,
                    }
                    payload["meta"] = meta
                    return payload

                train_payload = _attach_neighbor_fields(train_payload, "train")
                val_payload = _attach_neighbor_fields(val_payload, "val")
                test_payload = _attach_neighbor_fields(test_payload, "test")
            except Exception as e:
                print(f"[Stage2] Warning: failed to apply neighbor-aware stats (lambda_pop={lambda_pop}): {e}")

        ds_train = EmbeddingDataset(train_payload)
        ds_val = EmbeddingDataset(val_payload)
        ds_test = EmbeddingDataset(test_payload)

        train_num_samples = int(len(ds_train))
        train_samples_list.append(train_num_samples)

        labels_np = ds_train.labels.numpy()
        class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=labels_np)
        class_weights_t = torch.tensor(class_weights, dtype=torch.float)

        model = MultiHeadBinaryOnEmbedding(
            input_dim=input_dim,
            num_heads=args.num_heads,
            head_hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
            lambda_cal=args.lambda_cal,
            lambda_pop=float(getattr(args, "lambda_pop", 0.0)),
            cal_alpha=args.cal_alpha,
            u_min=args.u_min,
            u_max=args.u_max,
        )
        model.set_class_weights(class_weights_t)

        out_dir = os.path.join(checkpoint_root, f"fold_{fold_id}")
        os.makedirs(out_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(out_dir, f"training_log_{timestamp}.txt")
        log_file = open(log_file_path, "w", buffering=1)

        class TeeOutput:
            def __init__(self, *files):
                self.files = files
                self._primary = files[0] if files else None

            def write(self, data):
                for f in self.files:
                    f.write(data)
                    f.flush()

            def flush(self):
                for f in self.files:
                    f.flush()

            def isatty(self):
                if self._primary and hasattr(self._primary, "isatty"):
                    return self._primary.isatty()
                return False

            def fileno(self):
                if self._primary and hasattr(self._primary, "fileno"):
                    return self._primary.fileno()
                raise OSError("fileno not available")

            def __getattr__(self, name):
                if self._primary and hasattr(self._primary, name):
                    return getattr(self._primary, name)
                raise AttributeError(f"'TeeOutput' object has no attribute '{name}'")

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        sys.stdout = TeeOutput(original_stdout, log_file)
        sys.stderr = TeeOutput(original_stderr, log_file)

        print(f"{'='*80}")
        print(f"Training Fold {fold_id}/{5}")
        print(f"Log file: {log_file_path}")
        print(f"Started at: {timestamp}")
        print(f"{'='*80}\n")

        training_args = build_training_arguments(
            args,
            out_dir=out_dir,
            sanitized_model=sanitized_model,
            label_col=label_col,
            fold_id=fold_id,
        )
        wandb_run = init_wandb_run(
            args,
            base_root=base_root,
            sanitized_model=sanitized_model,
            label_col=label_col,
            fold_id=fold_id,
        )

        def _compute_metrics(eval_pred):
            predictions, labels = eval_pred
            return compute_metrics(
                predictions,
                labels,
                ece_bins=args.ece_bins,
                pos_ece_bins=args.pos_ece_bins,
                f1_threshold=args.f1_threshold,
                f1_find_best=args.f1_find_best,
                f1_grid_step=args.f1_grid_step,
            )

        eval_bsz = max(1, int(args.per_device_eval_batch_size))

        val_epoch_history: List[Dict[str, float]] = []
        test_epoch_history: List[Dict[str, float]] = []

        callbacks = []
        if args.use_early_stopping:
            class WarmupAwareEarlyStoppingCallback(EarlyStoppingCallback):
                def __init__(self, *, warmup_steps: int, early_stopping_patience: int, early_stopping_threshold: float):
                    super().__init__(
                        early_stopping_patience=early_stopping_patience,
                        early_stopping_threshold=early_stopping_threshold,
                    )
                    self._warmup_steps = int(max(0, warmup_steps))

                def on_evaluate(self, args_ta, state, control, metrics=None, **kwargs):
                    try:
                        cur_step = int(getattr(state, "global_step", 0))
                    except Exception:
                        cur_step = 0
                    if cur_step < self._warmup_steps:
                        return control
                    return super().on_evaluate(args_ta, state, control, metrics=metrics, **kwargs)

            callbacks.append(WarmupAwareEarlyStoppingCallback(
                warmup_steps=int(getattr(args, "warmup_steps", 0)),
                early_stopping_patience=int(getattr(args, "early_stopping_patience", 10)),
                early_stopping_threshold=float(getattr(args, "early_stopping_threshold", 0.001)),
            ))

        # Log loss components (CE, L_cal) per epoch
        class _LogLossComponentsCallback(TrainerCallback):
            def __init__(self, trainer_inst, train_ds, val_ds, collator, eval_bsz):
                self._trainer = trainer_inst
                self.train_ds = train_ds
                self.val_ds = val_ds
                self.collator = collator
                self.eval_bsz = max(1, int(eval_bsz))

            @torch.no_grad()
            def _avg_components(self, model, dataset, device):
                from torch.utils.data import DataLoader
                loader = DataLoader(dataset, batch_size=self.eval_bsz, shuffle=False, collate_fn=self.collator, drop_last=False)
                was_training = model.training
                model.eval()
                sum_ce = 0.0
                sum_lcal = 0.0
                count = 0
                for batch in loader:
                    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    out = model(**batch)
                    bs = int(batch["labels"].shape[0]) if "labels" in batch else 1
                    ce = float(out.get("loss_ce", torch.tensor(0.0, device=device)).detach().item())
                    lcal = float(out.get("loss_lcal", torch.tensor(0.0, device=device)).detach().item()) if ("loss_lcal" in out) else 0.0
                    sum_ce += ce * bs
                    sum_lcal += lcal * bs
                    count += bs
                if was_training:
                    model.train()
                denom = max(1, count)
                return {
                    "CE": sum_ce / denom,
                    "cal": sum_lcal / denom,
                }

            def on_epoch_end(self, args_ta, state, control, **kwargs):
                trainer_inst = self._trainer
                if trainer_inst is None:
                    return
                try:
                    model_local = trainer_inst.model
                    device = next(model_local.parameters()).device
                    train_means = self._avg_components(model_local, self.train_ds, device)
                    val_means = self._avg_components(model_local, self.val_ds, device)
                    payload = {
                        "loss_CE": float(train_means.get("CE", 0.0)),
                        "loss_cal": float(train_means.get("cal", 0.0)),
                        "eval_loss_CE": float(val_means.get("CE", 0.0)),
                        "eval_loss_cal": float(val_means.get("cal", 0.0)),
                    }
                    try:
                        trainer_inst.log(payload)
                    except Exception:
                        pass
                except Exception:
                    pass

        callbacks.append(_LogLossComponentsCallback(None, ds_train, ds_val, _simple_collate, eval_bsz))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=_simple_collate,
            compute_metrics=_compute_metrics,
            callbacks=callbacks,
        )
        trainer._wandb_run = wandb_run
        for cb in trainer.callback_handler.callbacks:
            if isinstance(cb, _LogLossComponentsCallback):
                cb._trainer = trainer

        # Optional: test-set evaluation during training
        enable_test_epoch = (args.test_eval_strategy == "epoch")
        enable_test_steps = (args.test_eval_strategy == "steps")
        test_steps_history: List[Dict[str, float]] = []

        if enable_test_epoch or enable_test_steps:
            class _LogTestEvalCallback(TrainerCallback):
                def __init__(self, trainer_inst, ds, label, *, do_epoch: bool, do_steps: bool, step_interval: int, history_steps: List[Dict[str, float]]):
                    self._trainer = trainer_inst
                    self._ds = ds
                    self._label = label
                    self._do_epoch = do_epoch
                    self._do_steps = do_steps
                    self._step_interval = max(1, int(step_interval))
                    self._history_steps = history_steps

                def on_step_end(self, args_ta, state, control, **kwargs):
                    if not self._do_steps:
                        return
                    if (state.global_step != 0) and (state.global_step % self._step_interval == 0):
                        self._run_eval(state)

                def on_epoch_end(self, args_ta, state, control, **kwargs):
                    if not self._do_epoch:
                        return
                    self._run_eval(state)

                def _run_eval(self, state):
                    try:
                        out = self._trainer.predict(self._ds)
                    except Exception:
                        return
                    try:
                        is_main = True
                        if hasattr(self._trainer, "is_world_process_zero"):
                            is_main = bool(self._trainer.is_world_process_zero())
                    except Exception:
                        is_main = True
                    if is_main:
                        metrics = compute_metrics(
                            out.predictions, out.label_ids,
                            ece_bins=args.ece_bins, pos_ece_bins=args.pos_ece_bins,
                            f1_threshold=args.f1_threshold, f1_find_best=False, f1_grid_step=args.f1_grid_step,
                        )
                        try:
                            step_val = int(getattr(state, "global_step", 0))
                        except Exception:
                            step_val = 0
                        _log_wandb_direct(self._trainer, {f"test/{k}": float(v) for k, v in metrics.items()}, step=step_val)
                        try:
                            ep = getattr(state, "epoch", None)
                            ep_i = int(round(float(ep))) if ep is not None else (len(test_epoch_history) + 1)
                        except Exception:
                            ep_i = len(test_epoch_history) + 1
                        rec_epoch = {"epoch": ep_i}
                        rec_epoch.update({k: float(v) for k, v in metrics.items()})
                        test_epoch_history.append(rec_epoch)
                        try:
                            step_i = int(getattr(state, "global_step", 0))
                        except Exception:
                            step_i = 0
                        rec_steps = {"step": step_i, "epoch": float(getattr(state, "epoch", 0.0) or 0.0)}
                        rec_steps.update({k: float(v) for k, v in metrics.items()})
                        self._history_steps.append(rec_steps)

            trainer.add_callback(_LogTestEvalCallback(
                trainer, ds_test, label_col,
                do_epoch=enable_test_epoch, do_steps=enable_test_steps,
                step_interval=int(getattr(args, "test_eval_steps", 2000)),
                history_steps=test_steps_history,
            ))

        # Validation metrics per epoch for JSON history
        class _LogValidEvalCallback(TrainerCallback):
            def __init__(self, trainer_inst, ds):
                self._trainer = trainer_inst
                self._ds = ds

            def on_epoch_end(self, args_ta, state, control, **kwargs):
                try:
                    out = self._trainer.predict(self._ds)
                except Exception:
                    return
                try:
                    try:
                        ep = getattr(state, "epoch", None)
                        ep_i = int(round(float(ep))) if ep is not None else (len(val_epoch_history) + 1)
                    except Exception:
                        ep_i = len(val_epoch_history) + 1
                    metrics_val = compute_metrics(
                        out.predictions, out.label_ids,
                        ece_bins=args.ece_bins, pos_ece_bins=args.pos_ece_bins,
                        f1_threshold=args.f1_threshold, f1_find_best=False, f1_grid_step=args.f1_grid_step,
                    )
                    rec_v = {"epoch": ep_i}
                    rec_v.update({k: float(v) for k, v in metrics_val.items()})
                    val_epoch_history.append(rec_v)
                except Exception:
                    pass

        trainer.add_callback(_LogValidEvalCallback(trainer, ds_val))

        final_checkpoint_metrics: Optional[Dict[str, float]] = None
        try:
            trainer.train()
            val_metrics = trainer.evaluate()
            val_metrics_clean = {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))}
            val_metrics_list.append(val_metrics_clean)

            test_f1_threshold = args.f1_threshold
            if args.f1_find_best:
                vkey = "eval_best_f1_threshold"
                if vkey in val_metrics:
                    try:
                        test_f1_threshold = float(val_metrics[vkey])
                        print(f"[Fold {fold_id}] Using best F1 threshold from validation: {test_f1_threshold:.4f}")
                    except (ValueError, TypeError):
                        pass
                else:
                    print(f"[Fold {fold_id}] 'eval_best_f1_threshold' not found, performing grid search on validation set...")
                    val_out = trainer.predict(ds_val)
                    val_preds = val_out.predictions
                    val_labels = val_out.label_ids.astype(int)
                    from utilities.metrics import ensure_ndarray, softmax_np
                    val_preds_np = ensure_ndarray(val_preds)
                    probs_heads = softmax_np(val_preds_np, axis=-1)[..., 1]
                    mean_prob = probs_heads.mean(axis=1)
                    grid = np.arange(0.0, 1.0 + 1e-8, args.f1_grid_step)
                    from sklearn.metrics import f1_score
                    f1_vals = [f1_score(val_labels, (mean_prob >= g).astype(int)) for g in grid]
                    best_idx = int(np.argmax(f1_vals))
                    test_f1_threshold = float(grid[best_idx])
                    print(f"[Fold {fold_id}] Best F1 threshold: {test_f1_threshold:.4f} (F1={f1_vals[best_idx]:.4f})")

            test_out = trainer.predict(ds_test)
            test_metrics = compute_metrics(
                test_out.predictions, test_out.label_ids,
                ece_bins=args.ece_bins, pos_ece_bins=args.pos_ece_bins,
                f1_threshold=test_f1_threshold, f1_find_best=False, f1_grid_step=args.f1_grid_step,
            )
            test_metrics_list.append(test_metrics)
            log_test_curves(wandb_run, label_col=label_col, fold_id=fold_id, test_out=test_out)

            # Save per-sample test predictions
            try:
                from utilities.metrics import ensure_ndarray, softmax_np
                preds_np = ensure_ndarray(test_out.predictions)
                probs_heads = softmax_np(preds_np, axis=-1)[..., 1]
                if probs_heads.ndim == 1:
                    probs_heads_2d = probs_heads[:, None].astype(np.float32)
                    mean_prob = probs_heads.astype(float)
                else:
                    probs_heads_2d = probs_heads.astype(np.float32)
                    mean_prob = probs_heads.mean(axis=1).astype(float)

                labels_np = np.asarray(test_out.label_ids).astype(int)
                thr = float(test_f1_threshold)
                pred_bin = (mean_prob >= thr).astype(int)
                correct = (pred_bin == labels_np).astype(int)

                out_path = os.path.join(eval_root, f"fold_{fold_id}_test_predictions.npz")
                np.savez(out_path, mean_prob=mean_prob, probs_heads=probs_heads_2d,
                         label=labels_np, pred=pred_bin, correct=correct, threshold=thr)
                print(f"[Fold {fold_id}] Saved per-sample test predictions to: {out_path} (N={len(mean_prob)}, threshold={thr:.4f})")
            except Exception as e:
                print(f"[Fold {fold_id}] Warning: failed to save per-sample test predictions: {e}")

            # Evaluate final checkpoint on test set
            try:
                ckpt_names = [d for d in os.listdir(out_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(out_dir, d))]
                last_ckpt_dir = None
                if ckpt_names:
                    def _step_from_name(n: str) -> int:
                        try:
                            return int(n.split("-")[-1])
                        except Exception:
                            return -1
                    ckpt_names_sorted = sorted(ckpt_names, key=_step_from_name)
                    last_ckpt = ckpt_names_sorted[-1]
                    last_ckpt_dir = os.path.join(out_dir, last_ckpt)

                if last_ckpt_dir is not None:
                    device = next(trainer.model.parameters()).device
                    cand_files = ["pytorch_model.bin", "model.safetensors", "adapter_model.safetensors", "adapter_model.bin"]
                    weight_path = None
                    for cf in cand_files:
                        pth = os.path.join(last_ckpt_dir, cf)
                        if os.path.exists(pth):
                            weight_path = pth
                            break
                    if weight_path is not None:
                        state_dict = torch.load(weight_path, map_location=device)
                        model_last = MultiHeadBinaryOnEmbedding(
                            input_dim=input_dim, num_heads=args.num_heads,
                            head_hidden_dim=args.head_hidden_dim, dropout=args.dropout,
                            lambda_cal=args.lambda_cal, cal_alpha=args.cal_alpha,
                            u_min=args.u_min, u_max=args.u_max,
                        )
                        model_last.set_class_weights(class_weights_t)
                        model_last.to(device)
                        try:
                            model_last.load_state_dict(state_dict, strict=True)
                        except Exception:
                            model_last.load_state_dict(state_dict, strict=False)

                        from torch.utils.data import DataLoader
                        loader = DataLoader(ds_test, batch_size=args.per_device_eval_batch_size,
                                            shuffle=False, collate_fn=_simple_collate, drop_last=False)
                        logits_list: List[torch.Tensor] = []
                        labels_list: List[torch.Tensor] = []
                        model_last.eval()
                        with torch.no_grad():
                            for batch in loader:
                                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                                out_last = model_last(**batch)
                                logits_list.append(out_last["logits"].detach().cpu())
                                labels_list.append(batch["labels"].detach().cpu())
                        preds_last = torch.cat(logits_list, dim=0).numpy()
                        labels_last = torch.cat(labels_list, dim=0).numpy()
                        final_checkpoint_metrics = compute_metrics(
                            preds_last, labels_last,
                            ece_bins=args.ece_bins, pos_ece_bins=args.pos_ece_bins,
                            f1_threshold=test_f1_threshold, f1_find_best=False, f1_grid_step=args.f1_grid_step,
                        )
                        print(f"[Fold {fold_id}] Evaluated final checkpoint at {last_ckpt_dir} on test set.")

                        try:
                            from utilities.metrics import ensure_ndarray, softmax_np
                            preds_last_np = ensure_ndarray(preds_last)
                            probs_heads_last = softmax_np(preds_last_np, axis=-1)[..., 1]
                            if probs_heads_last.ndim == 1:
                                probs_heads_last_2d = probs_heads_last[:, None].astype(np.float32)
                                mean_prob_last = probs_heads_last.astype(float)
                            else:
                                probs_heads_last_2d = probs_heads_last.astype(np.float32)
                                mean_prob_last = probs_heads_last.mean(axis=1).astype(float)
                            labels_last_np = np.asarray(labels_last).astype(int)
                            thr_last = float(test_f1_threshold)
                            pred_last_bin = (mean_prob_last >= thr_last).astype(int)
                            correct_last = (pred_last_bin == labels_last_np).astype(int)
                            out_path_final = os.path.join(eval_root, f"fold_{fold_id}_test_predictions_final.npz")
                            np.savez(out_path_final, mean_prob=mean_prob_last, probs_heads=probs_heads_last_2d,
                                     label=labels_last_np, pred=pred_last_bin, correct=correct_last, threshold=thr_last)
                            print(f"[Fold {fold_id}] Saved final-checkpoint predictions to: {out_path_final}")
                        except Exception as e:
                            print(f"[Fold {fold_id}] Warning: failed to save final-checkpoint predictions: {e}")
            except Exception as e:
                print(f"[Fold {fold_id}] Warning: failed to evaluate final checkpoint: {e}")

            try:
                trainer.save_model(out_dir)
            except Exception:
                pass

            val_history_steps: List[Dict[str, float]] = []
            val_history_epoch: List[Dict[str, float]] = []
            try:
                for rec in trainer.state.log_history:
                    if not isinstance(rec, dict):
                        continue
                    if not any(k.startswith("eval_") for k in rec.keys()):
                        continue
                    vals = {k: float(v) for k, v in rec.items() if k.startswith("eval_") and isinstance(v, (int, float))}
                    step_i = int(rec.get("step", 0) or 0)
                    epoch_f = float(rec.get("epoch", 0.0) or 0.0)
                    if step_i > 0:
                        val_history_steps.append({**vals, "step": step_i, "epoch": epoch_f})
                    if epoch_f > 0.0:
                        val_history_epoch.append({**vals, "epoch": epoch_f, "step": step_i})
            except Exception:
                pass

            with open(os.path.join(eval_root, f"fold_{fold_id}_val.json"), "w") as f:
                if str(args.eval_strategy) == "steps":
                    json.dump({"final": val_metrics_clean, "history_steps": val_history_steps}, f, indent=2)
                else:
                    json.dump({"final": val_metrics_clean, "history_epoch": val_history_epoch}, f, indent=2)
            with open(os.path.join(eval_root, f"fold_{fold_id}_test.json"), "w") as f:
                base_payload: Dict[str, object] = {
                    "final_best": test_metrics,
                    "final_checkpoint": final_checkpoint_metrics,
                }
                if str(args.test_eval_strategy) == "steps":
                    base_payload["history_steps"] = test_steps_history
                elif str(args.test_eval_strategy) == "epoch":
                    base_payload["history_epoch"] = test_epoch_history
                base_payload["train_num_samples"] = train_num_samples
                json.dump(base_payload, f, indent=2)

            if final_checkpoint_metrics is not None:
                final_checkpoint_metrics_list.append(final_checkpoint_metrics)

            finish_wandb(wandb_run)
        finally:
            end_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"\n{'='*80}")
            print(f"Fold {fold_id} Training Completed")
            print(f"Ended at: {end_timestamp}")
            print(f"Log saved to: {log_file_path}")
            print(f"{'='*80}")

            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()

            print(f"[Fold {fold_id}] Training completed. Log saved to: {log_file_path}")

    if val_metrics_list:
        vdf = pd.DataFrame(val_metrics_list)
        summary = {"mean": vdf.mean().to_dict(), "std": vdf.std().to_dict()}
        summary = {k: {sk: float(sv) for sk, sv in v.items()} for k, v in summary.items()}
        with open(os.path.join(eval_root, "val_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    if test_metrics_list:
        tdf = pd.DataFrame(test_metrics_list)
        summary_best = {"mean": tdf.mean().to_dict(), "std": tdf.std().to_dict()}
        summary_best = {k: {sk: float(sv) for sk, sv in v.items()} for k, v in summary_best.items()}

        summary: Dict[str, object] = {}
        summary["mean"] = summary_best["mean"]
        summary["std"] = summary_best["std"]

        if final_checkpoint_metrics_list:
            fdf = pd.DataFrame(final_checkpoint_metrics_list)
            summary_final = {"mean": fdf.mean().to_dict(), "std": fdf.std().to_dict()}
            summary_final = {k: {sk: float(sv) for sk, sv in v.items()} for k, v in summary_final.items()}
            summary["final_checkpoint_mean"] = summary_final["mean"]
            summary["final_checkpoint_std"] = summary_final["std"]

        if train_samples_list:
            summary["data_volume"] = {
                "train_num_samples_mean": float(np.mean(train_samples_list)),
                "train_num_samples_total": int(sum(train_samples_list)),
            }
        with open(os.path.join(eval_root, "test_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    for label in args.labels:
        run_one_label(args, label)


if __name__ == "__main__":
    main()
