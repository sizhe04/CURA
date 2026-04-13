import argparse
import datetime
import json
import os
import random
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback

from data import ClinicalNotesDataset
from modeling_singlehead import SingleHeadBinaryClassifier, HFSingleHeadBinaryClassifier
from utilities.metrics import compute_metrics, ensure_ndarray
from utilities.training_args import build_training_arguments
from utilities.paths import get_experiment_dirs
from utilities.io_utils import write_json
from utilities.embeddings_io import extract_embeddings_loop, save_embeddings_pt, load_embeddings_pt
from utilities.wandb_utils import init_wandb_run, finish_wandb
from utilities.repr_uncert import ReprUncertConfig, attach_repr_to_payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Stage-1: 5-fold CV fine-tuning with single head")
    # data/label
    p.add_argument("--data_csv", type=str, required=True)
    p.add_argument("--text_col", type=str, default="clean_text")
    p.add_argument("--labels", type=str, nargs="+", default=["death_in_30"], help="one or more label columns")
    # model
    p.add_argument("--model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--head_hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--pooling",
        type=str,
        choices=["cls", "pooler", "mean", "last_hidden_state", "last_token"],
        default="cls",
        help=(
            "Pooling strategy for sentence representation: "
            "'cls' | 'pooler' | 'mean' | 'last_hidden_state' | 'last_token'. "
            "'last_hidden_state' is equivalent to masked mean pooling over the last-layer hidden states; "
            "'last_token' uses the representation of the last non-padding token in the sequence "
            "(recommended for GPT/BioGPT-style models)."
        ),
    )
    p.add_argument("--max_length", type=int, default=512)
    # loss & head type
    p.add_argument("--loss_type", type=str, choices=["ce", "edl"], default="ce")
    p.add_argument("--edl_kl_weight", type=float, default=0.1)
    p.add_argument("--classifier_arch", type=str, choices=["custom", "hf"], default="custom")
    # metrics
    p.add_argument("--ece_bins", type=int, default=100)
    p.add_argument("--f1_find_best", action="store_true")
    p.add_argument("--f1_threshold", type=float, default=0.5)
    p.add_argument("--f1_grid_step", type=float, default=0.01)
    p.add_argument("--pos_ece_bins", type=int, default=20)
    # Test-set evaluation during training (W&B plot frequency only; does not affect final saves)
    p.add_argument("--test_eval_strategy", type=str, choices=["end_only", "epoch", "steps", "none"], default="end_only")
    p.add_argument("--test_eval_steps", type=int, default=2000)
    # training args
    p.add_argument("--output_root", type=str, default="./outputs_cv")
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=1500)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--fp16", action="store_true")
    # Active Learning placeholder to keep directory naming consistent
    p.add_argument("--active_strategy", type=str, default="none")
    p.add_argument("--active_epoch_top_p", type=float, default=0.0)
    p.add_argument("--active_batch_top_p", type=float, default=0.0)
    p.add_argument("--active_warmup_steps", type=int, default=0)
    p.add_argument("--active_eval_batch_size", type=int, default=None)
    # selection for best model / early stop metric
    p.add_argument("--best_metric", type=str, default="eval_cls_NLL")
    p.add_argument("--greater_is_better", type=int, default=0)
    # eval/save policies
    p.add_argument("--eval_strategy", type=str, choices=["epoch", "steps"], default="epoch")
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_strategy", type=str, choices=["auto", "epoch", "steps", "no"], default="auto")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=5)
    # early stopping toggle
    p.add_argument("--use_early_stopping", action="store_true")
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--early_stopping_threshold", type=float, default=0.001)
    # seed
    p.add_argument("--seed", type=int, default=42)
    # wandb
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--num_heads", type=int, default=1)
    p.add_argument("--lambda_cal", type=float, default=0.0)
    p.add_argument("--cal_alpha", type=float, default=1.0)
    p.add_argument("--u_min", type=float, default=0.05)
    p.add_argument("--u_max", type=float, default=0.95)
    # Single-head L_cal toggle (only effective when loss_type=ce)
    p.add_argument(
        "--use_lcal",
        action="store_true",
        help="Whether to add single-head calibration term L_cal to CE (K=1 specialization).",
    )
    # representation-space uncertainty (U_repr)
    p.add_argument("--repr_k_neighbors", type=int, default=0, help="k for neighborhood uncertainty; <=0 to disable")
    p.add_argument("--repr_distance_metric", type=str, choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--repr_search_split", type=str, choices=["train", "train+val"], default="train+val")
    p.add_argument("--repr_score_type", type=str, choices=["entropy", "margin"], default="entropy")
    p.add_argument("--repr_normalize", type=int, default=1, help="1 to store entropy/log2 and margin*2; 0 to keep raw only")
    # optional: 1-based fold ids to run, e.g. --fold_ids 2 3 4 5; omit to run all 5 folds
    p.add_argument(
        "--fold_ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional 1-based fold ids to run (e.g. 2 3 4 5). If unset, run all 5 folds.",
    )
    return p.parse_args()


def _wrap_single_head_predictions(predictions: np.ndarray) -> np.ndarray:
    """
    Ensure predictions have shape (N, K, 2) for metric functions.
    If input is (N, 2), expand to (N, 1, 2).
    """
    # Compatible across transformers versions: predictions may be list/tuple (collected per batch)
    arr = ensure_ndarray(predictions)
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return arr[:, None, :]
    return arr


def run_one_label(args: argparse.Namespace, df: pd.DataFrame, label_col: str, tokenizer) -> None:
    df = df.copy()
    if df[label_col].dtype != int:
        df[label_col] = df[label_col].astype(int)
    if "death_in_30" in df.columns and df["death_in_30"].dtype != int:
        df["death_in_30"] = df["death_in_30"].astype(int)

    sanitized_model, base_root, checkpoint_root, eval_root = get_experiment_dirs(args, label_col)
    os.makedirs(checkpoint_root, exist_ok=True)
    os.makedirs(eval_root, exist_ok=True)
    # Append exp_dir segment to WANDB_PROJECT for grouping under the project
    # base_root: {OUTPUT_ROOT}/{model}/{exp_dir}/{label}
    try:
        exp_dir = os.path.basename(os.path.dirname(base_root))
        if args.use_wandb:
            _orig_proj = os.environ.get("WANDB_PROJECT", None)
            if _orig_proj:
                if not _orig_proj.endswith(exp_dir):
                    os.environ["WANDB_PROJECT"] = f"{_orig_proj}_{exp_dir}"
    except Exception:
        pass
    tokenizer.save_pretrained(os.path.join(base_root, "tokenizer"))
    with open(os.path.join(base_root, "hparams_stage1.json"), "w", encoding="utf-8") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in vars(args).items()}, f, ensure_ascii=False, indent=2)

    # v1 (death_in_30 / death_in_7, etc.): keep original unified stratification by death_in_30;
    # v2 four tasks use their own label for stratification, aligned with LLM_uncertainty v2 code.
    v2_labels = {
        "after_ICU_more_than_3_days",
        "discharge_less_than_12_hours",
        "hospital_expire_flag",
        "ICU_more_than_1_days",
    }
    if label_col in v2_labels:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X = df[args.text_col]
        stratifier = df[label_col].astype(int)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        X = df[args.text_col]
        # stratifier uses death_in_30 if available to keep consistency across labels
        if label_col == "death_in_30":
            stratifier = df[label_col].astype(int)
        else:
            if "death_in_30" not in df.columns:
                raise ValueError("Column death_in_30 is missing from the data; unified stratification cannot be applied.")
            stratifier = df["death_in_30"].astype(int)

    # If --fold_ids lists 1-based folds to run, train/save only on those folds;
    # otherwise run all 5 folds. This affects only the current run, not the data split definition.
    selected_folds = None
    if getattr(args, "fold_ids", None):
        try:
            selected_folds = {int(x) for x in args.fold_ids}
        except Exception:
            selected_folds = None

    fold_id = 1
    val_metrics_list: List[Dict[str, float]] = []
    test_metrics_list: List[Dict[str, float]] = []

    for train_index, test_index in skf.split(X, stratifier):
        # The 5-fold split is fixed via random_state, so it is stable across runs;
        # if this fold_id is not selected, skip training and disk writes to avoid overwriting prior results.
        if selected_folds is not None and fold_id not in selected_folds:
            print(f"[Stage1] Skip fold {fold_id} because it is not in --fold_ids.", flush=True)
            fold_id += 1
            continue

        train_df_all = df.iloc[train_index].reset_index(drop=True)
        test_df = df.iloc[test_index].reset_index(drop=True)

        y_train_all = stratifier.iloc[train_index].values
        # v2 four tasks: match LLM_uncertainty v2 — stratify on each task's label, random_state=42
        if label_col in v2_labels:
            idx_train, idx_val = train_test_split(
                np.arange(len(train_df_all)),
                test_size=0.1,
                stratify=y_train_all,
                random_state=42,
            )
        else:
            idx_train, idx_val = train_test_split(
                np.arange(len(train_df_all)),
                test_size=0.1,
                stratify=y_train_all,
                random_state=args.seed,
            )
        train_df = train_df_all.iloc[idx_train].reset_index(drop=True)
        val_df = train_df_all.iloc[idx_val].reset_index(drop=True)

        labels_np = train_df[label_col].to_numpy().astype(int)
        class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=labels_np)
        class_weights_t = torch.tensor(class_weights, dtype=torch.float)

        is_longformer = "longformer" in args.model_name.lower()

        ds_train = ClinicalNotesDataset(
            train_df[[args.text_col, label_col]].rename(columns={label_col: "label"}),
            tokenizer,
            args.text_col,
            "label",
            args.max_length,
            use_longformer_global_attention=is_longformer,
        )
        ds_val = ClinicalNotesDataset(
            val_df[[args.text_col, label_col]].rename(columns={label_col: "label"}),
            tokenizer,
            args.text_col,
            "label",
            args.max_length,
            use_longformer_global_attention=is_longformer,
        )
        ds_test = ClinicalNotesDataset(
            test_df[[args.text_col, label_col]].rename(columns={label_col: "label"}),
            tokenizer,
            args.text_col,
            "label",
            args.max_length,
            use_longformer_global_attention=is_longformer,
        )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        if args.classifier_arch == "hf":
            model = HFSingleHeadBinaryClassifier(
                model_name=args.model_name,
                pooling=args.pooling,
                loss_type=args.loss_type,
                edl_kl_weight=args.edl_kl_weight,
                # Single-head L_cal
                lambda_cal=args.lambda_cal,
                cal_alpha=args.cal_alpha,
                u_min=args.u_min,
                u_max=args.u_max,
                use_lcal=args.use_lcal,
            )
        else:
            model = SingleHeadBinaryClassifier(
                model_name=args.model_name,
                head_hidden_dim=args.head_hidden_dim,
                dropout=args.dropout,
                pooling=args.pooling,
                loss_type=args.loss_type,
                edl_kl_weight=args.edl_kl_weight,
                # Single-head L_cal
                lambda_cal=args.lambda_cal,
                cal_alpha=args.cal_alpha,
                u_min=args.u_min,
                u_max=args.u_max,
                use_lcal=args.use_lcal,
            )
        model.set_class_weights(class_weights_t)

        out_dir = os.path.join(checkpoint_root, f"fold_{fold_id}")
        os.makedirs(out_dir, exist_ok=True)
        training_args = build_training_arguments(
            args,
            out_dir=out_dir,
            sanitized_model=sanitized_model,
            label_col=label_col,
            fold_id=fold_id,
        )
        # Initialize a separate W&B run per fold (single process / single GPU)
        wandb_run = init_wandb_run(
            args,
            base_root=base_root,
            sanitized_model=sanitized_model,
            label_col=label_col,
            fold_id=fold_id,
        )

        # metric function wrapper for single-head
        def _compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = _wrap_single_head_predictions(predictions)
            return compute_metrics(
                predictions,
                labels,
                ece_bins=args.ece_bins,
                pos_ece_bins=args.pos_ece_bins,
                f1_threshold=args.f1_threshold,
                f1_find_best=args.f1_find_best,
                f1_grid_step=args.f1_grid_step,
            )

        callbacks = []
        if args.use_early_stopping:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold
            ))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=_compute_metrics,
            callbacks=callbacks,
        )
        try:
            is_main_process = bool(trainer.is_world_process_zero())
        except Exception:
            is_main_process = True

        # ====== Optional: log test-set curves to W&B during training ======
        enable_test_epoch = (args.test_eval_strategy == "epoch")
        enable_test_steps = (args.test_eval_strategy == "steps")
        if enable_test_epoch or enable_test_steps:
            try:
                class _LogTestEvalCallback(TrainerCallback):
                    def __init__(self, trainer_inst, ds, *, do_epoch: bool, do_steps: bool, step_interval: int):
                        self._trainer = trainer_inst
                        self._ds = ds
                        self._do_epoch = do_epoch
                        self._do_steps = do_steps
                        self._step_interval = max(1, int(step_interval))

                    def on_step_end(self, args_ta, state, control, **kwargs):
                        if not self._do_steps:
                            return
                        should_run = (state.global_step != 0) and (state.global_step % self._step_interval == 0)
                        if not should_run:
                            return
                        self._run_eval(state)

                    def on_epoch_end(self, args_ta, state, control, **kwargs):
                        if not self._do_epoch:
                            return
                        self._run_eval(state)

                    def _run_eval(self, state):
                        try:
                            out = self._trainer.predict(self._ds)
                        except Exception as e:
                            print(f"[test-callback] predict failed: {e}")
                            return
                        # Log to W&B only on the main process
                        try:
                            ti = self._trainer
                            is_main = True
                            if hasattr(ti, "is_world_process_zero"):
                                is_main = bool(ti.is_world_process_zero())
                        except Exception:
                            is_main = True
                        if not is_main:
                            return
                        metrics = compute_metrics(
                            _wrap_single_head_predictions(out.predictions),
                            out.label_ids,
                            ece_bins=args.ece_bins,
                            pos_ece_bins=args.pos_ece_bins,
                            f1_threshold=args.f1_threshold,
                            f1_find_best=False,
                            f1_grid_step=args.f1_grid_step,
                        )
                        try:
                            import wandb  # type: ignore
                            wandb.log({f"test/{k}": float(v) for k, v in metrics.items()}, step=int(state.global_step))
                        except Exception as we:
                            print(f"[wandb] failed to log test metrics: {we}")

                trainer.add_callback(
                    _LogTestEvalCallback(
                        trainer,
                        ds_test,
                        do_epoch=enable_test_epoch,
                        do_steps=enable_test_steps,
                        step_interval=int(getattr(args, "test_eval_steps", 2000)),
                    )
                )
            except Exception as _cb_e:
                print(f"[callback] failed to register test-curve callback (ignored): {_cb_e}")

        trainer.train()
        val_metrics = trainer.evaluate()
        val_clean = {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))}
        if is_main_process:
            val_metrics_list.append(val_clean)

        # Choose threshold on validation to avoid test-set threshold search leakage
        thr = args.f1_threshold
        if args.f1_find_best:
            # 1) Prefer reading best_f1_threshold from eval_* metrics
            vkey = "eval_best_f1_threshold"
            if vkey in val_metrics:
                try:
                    thr = float(val_metrics[vkey])
                except Exception:
                    thr = args.f1_threshold
            else:
                # 2) Fallback: grid search on validation predictions
                try:
                    val_out = trainer.predict(ds_val)
                    preds = ensure_ndarray(val_out.predictions)
                    if preds.ndim == 2:
                        probs = torch.softmax(torch.from_numpy(preds), dim=-1)[:, 1].numpy()
                    else:
                        probs = torch.softmax(torch.from_numpy(preds)[:, 0, :], dim=-1)[:, 1].numpy()
                    grid = np.arange(0.0, 1.0 + 1e-8, args.f1_grid_step)
                    f1_vals = [f1_score(val_out.label_ids.astype(int), (probs >= g).astype(int)) for g in grid]
                    thr = float(grid[int(np.argmax(f1_vals))])
                except Exception:
                    thr = args.f1_threshold

        test_out = trainer.predict(ds_test)
        if not is_main_process:
            fold_id += 1
            continue

        metrics_test = compute_metrics(
            _wrap_single_head_predictions(test_out.predictions),
            test_out.label_ids,
            ece_bins=args.ece_bins,
            pos_ece_bins=args.pos_ece_bins,
            f1_threshold=thr,
            f1_find_best=False,
            f1_grid_step=args.f1_grid_step,
        )
        test_metrics_list.append(metrics_test)

        # Save per-fold summaries
        eval_dir = eval_root
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, f"fold_{fold_id}_val.json"), "w") as f:
            json.dump({"final": val_clean}, f, indent=2)
        with open(os.path.join(eval_dir, f"fold_{fold_id}_test.json"), "w") as f:
            json.dump({"final_best": metrics_test}, f, indent=2)

        # Explicitly save the current model (usually best) to out_dir for reuse
        try:
            trainer.save_model(out_dir)
        except Exception:
            pass

        # ============= Embedding extraction and saving =============
        # Generic save helper: suffix controls filenames (e.g. _best / _last / _best_last)
        def _save_all_splits_for_model(model_for_emb, emb_subdir: str, suffix: str) -> None:
            emb_dir = os.path.join(out_dir, emb_subdir)
            os.makedirs(emb_dir, exist_ok=True)

            def _save_split(name: str, split_df: pd.DataFrame, dataset: ClinicalNotesDataset):
                feats = extract_embeddings_loop(model_for_emb, dataset, data_collator, batch_size=args.per_device_eval_batch_size)
                labels_t = torch.tensor(split_df[label_col].to_numpy().astype(int), dtype=torch.long)
                row_idx_t = torch.arange(len(split_df), dtype=torch.long)
                note_id = split_df["note_id"].astype(str).tolist() if "note_id" in split_df.columns else None
                subject_id = split_df["subject_id"].astype(str).tolist() if "subject_id" in split_df.columns else None
                texts = split_df[args.text_col].astype(str).tolist()
                save_path = os.path.join(
                    emb_dir,
                    f"fold_{fold_id}_{name}_pool-{args.pooling}{suffix}.pt"
                )
                hidden_size = int(getattr(model_for_emb, "hidden_size", feats.shape[-1]))
                save_embeddings_pt(
                    save_path,
                    features=feats,
                    labels=labels_t,
                    row_idx=row_idx_t,
                    split=name,
                    pooling=args.pooling,
                    model_name=args.model_name,
                    hidden_size=hidden_size,
                    label_name=label_col,
                    fold_id=fold_id,
                    text=texts,
                    note_id=note_id,
                    subject_id=subject_id,
                )

            _save_split("train", train_df, ds_train)
            _save_split("val", val_df, ds_val)
            _save_split("test", test_df, ds_test)

        # Resolve checkpoint info first to tell whether best and last are the same
        best_ckpt_path = getattr(trainer.state, "best_model_checkpoint", None)
        best_ckpt_name = os.path.basename(best_ckpt_path) if best_ckpt_path else None

        # 2) Last checkpoint (last saved checkpoint-*), for comparing best vs last
        try:
            ckpt_names = [
                d for d in os.listdir(out_dir)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(out_dir, d))
            ]
        except FileNotFoundError:
            ckpt_names = []

        last_ckpt = None
        last_ckpt_dir = None
        if ckpt_names:
            # Treat the checkpoint with the largest step number as the last checkpoint
            def _step_from_name(n: str) -> int:
                try:
                    return int(n.split("-")[-1])
                except Exception:
                    return -1

            ckpt_names_sorted = sorted(ckpt_names, key=_step_from_name)
            last_ckpt = ckpt_names_sorted[-1]
            last_ckpt_dir = os.path.join(out_dir, last_ckpt)

        # Whether best and last refer to the same checkpoint (by directory name)
        best_is_last = best_ckpt_name is not None and last_ckpt is not None and (best_ckpt_name == last_ckpt)

        # Filename suffixes by case:
        # - Usually: best uses _best, last uses _last
        # - If best == last: both use _best_last
        if best_is_last:
            best_suffix = "_best_last"
            last_suffix = "_best_last"
        else:
            best_suffix = "_best"
            last_suffix = "_last"

        # 1) Best checkpoint: trainer.model (already loaded with best weights per best_metric)
        #    Under embeddings/: *_best*.pt, or *_best_last*.pt when best == last
        if best_is_last:
            print(f"[Fold {fold_id}] Extracting embeddings from BEST and LAST checkpoint (same checkpoint: {best_ckpt_name or 'unknown'})...")
        else:
            ckpt_info = best_ckpt_name if best_ckpt_name else "final model"
            print(f"[Fold {fold_id}] Extracting embeddings from BEST checkpoint ({ckpt_info})...")
        _save_all_splits_for_model(trainer.model, "embeddings", best_suffix)

        # 2) Last checkpoint: if checkpoint-* exists, save *_last*.pt under embeddings_last/,
        #    or *_best_last*.pt when best == last
        if last_ckpt_dir is not None:
            device = next(trainer.model.parameters()).device
            # Model weight files in checkpoint: support safetensors/bin layouts
            cand_files = [
                "model.safetensors",
                "pytorch_model.bin",
                "adapter_model.safetensors",
                "adapter_model.bin",
            ]
            weight_path = None
            for cf in cand_files:
                pth = os.path.join(last_ckpt_dir, cf)
                if os.path.exists(pth):
                    weight_path = pth
                    break
            if weight_path is None:
                last_ckpt_dir = None
            else:
                def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
                    if path.endswith(".safetensors"):
                        from safetensors.torch import load_file as safe_load  # type: ignore
                        sd = safe_load(path, device=device)
                    else:
                        print(f"[Fold {fold_id}] Extracting embeddings from LAST checkpoint ({last_ckpt})...")
                        sd = torch.load(path, map_location=device)
                    return sd

                state_dict = _load_state_dict(weight_path)

                if args.classifier_arch == "hf":
                    last_model = HFSingleHeadBinaryClassifier(
                        model_name=args.model_name,
                        pooling=args.pooling,
                        loss_type=args.loss_type,
                        edl_kl_weight=args.edl_kl_weight,
                    )
                    last_model.to(device)
                    print(f"    [DEBUG] state_dict keys sample: {list(state_dict.keys())[:3]}")
                    print(f"    [DEBUG] Expected model keys sample: {list(last_model.state_dict().keys())[:3]}")
                    try:
                        load_result = last_model.load_state_dict(state_dict, strict=True)
                    except Exception:
                        load_result = last_model.load_state_dict(state_dict, strict=False)
                    if getattr(load_result, "missing_keys", None):
                        print(f"    [WARNING] Missing keys when loading checkpoint: {load_result.missing_keys[:5]}")
                    if getattr(load_result, "unexpected_keys", None):
                        print(f"    [WARNING] Unexpected keys in checkpoint: {load_result.unexpected_keys[:5]}")
                    _save_all_splits_for_model(last_model, "embeddings_last", last_suffix)
                    del last_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    last_model = SingleHeadBinaryClassifier(
                        model_name=args.model_name,
                        head_hidden_dim=args.head_hidden_dim,
                        dropout=args.dropout,
                        pooling=args.pooling,
                        loss_type=args.loss_type,
                        edl_kl_weight=args.edl_kl_weight,
                    )
                    last_model.to(device)
                    print(f"    [DEBUG] state_dict keys sample: {list(state_dict.keys())[:3]}")
                    print(f"    [DEBUG] Expected model keys sample: {list(last_model.state_dict().keys())[:3]}")
                    try:
                        load_result = last_model.load_state_dict(state_dict, strict=True)
                    except Exception:
                        load_result = last_model.load_state_dict(state_dict, strict=False)
                    if getattr(load_result, "missing_keys", None):
                        print(f"    [WARNING] Missing keys when loading checkpoint: {load_result.missing_keys[:5]}")
                    if getattr(load_result, "unexpected_keys", None):
                        print(f"    [WARNING] Unexpected keys in checkpoint: {load_result.unexpected_keys[:5]}")
                    _save_all_splits_for_model(last_model, "embeddings_last", last_suffix)
                    del last_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        # ===========================================================

        # ===== Representation-space uncertainty on LAST embeddings (train only) =====
        try:
            if int(getattr(args, "repr_k_neighbors", 0)) > 0 and last_ckpt_dir is not None:
                train_pt_path = os.path.join(out_dir, "embeddings_last", f"fold_{fold_id}_train_pool-{args.pooling}{last_suffix}.pt")
                val_pt_path = os.path.join(out_dir, "embeddings_last", f"fold_{fold_id}_val_pool-{args.pooling}{last_suffix}.pt")
                if not os.path.exists(train_pt_path):
                    print(f"[U_repr] train embeddings not found, skip: {train_pt_path}")
                else:
                    train_payload = load_embeddings_pt(train_pt_path)
                    search_payloads = [train_payload]
                    if str(getattr(args, "repr_search_split", "train+val")) == "train+val" and os.path.exists(val_pt_path):
                        search_payloads.append(load_embeddings_pt(val_pt_path))
                    cfg = ReprUncertConfig(
                        k_neighbors=int(getattr(args, "repr_k_neighbors", 0)),
                        distance_metric=str(getattr(args, "repr_distance_metric", "cosine")),
                        search_split=str(getattr(args, "repr_search_split", "train+val")),
                        score_type=str(getattr(args, "repr_score_type", "entropy")),
                        normalize=bool(int(getattr(args, "repr_normalize", 1))),
                    )
                    updated = attach_repr_to_payload(train_payload, search_payloads, cfg)
                    torch.save(updated, train_pt_path)
                    print(
                        f"[U_repr] fold {fold_id}: added entropy/margin (raw+norm) to {train_pt_path}; "
                        f"k={cfg.k_neighbors}, metric={cfg.distance_metric}, search={cfg.search_split}, normalize={cfg.normalize}"
                    )
            elif int(getattr(args, "repr_k_neighbors", 0)) <= 0:
                print("[U_repr] repr_k_neighbors<=0, skip U_repr computation.")
        except Exception as e:
            print(f"[U_repr] failed to compute U_repr for fold {fold_id}: {e}")

        # Finish W&B logging for this fold
        finish_wandb(wandb_run)

        fold_id += 1

    # Rewrite aggregate summaries only when all 5 folds ran or --fold_ids was not set;
    # when training a subset of folds, skip aggregate writes to avoid overwriting a full-run summary.
    all_folds = {1, 2, 3, 4, 5}
    write_summary = (selected_folds is None) or (set(selected_folds) == all_folds)

    if val_metrics_list and write_summary:
        vdf = pd.DataFrame(val_metrics_list)
        summary = {"mean": vdf.mean().to_dict(), "std": vdf.std().to_dict()}
        summary = {k: {sk: float(sv) for sk, sv in v.items()} for k, v in summary.items()}
        with open(os.path.join(eval_root, "val_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    if test_metrics_list and write_summary:
        tdf = pd.DataFrame(test_metrics_list)
        summary = {"mean": tdf.mean().to_dict(), "std": tdf.std().to_dict()}
        summary = {k: {sk: float(sv) for sk, sv in v.items()} for k, v in summary.items()}
        with open(os.path.join(eval_root, "test_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    df = pd.read_csv(args.data_csv, compression="infer")
    for label in args.labels:
        if label not in df.columns:
            raise ValueError(f"Label column '{label}' not found in data.")
        subset_cols = [args.text_col, label]
        for extra in ("note_id", "subject_id"):
            if extra in df.columns and extra not in subset_cols:
                subset_cols.append(extra)
        if "death_in_30" not in subset_cols and "death_in_30" in df.columns:
            subset_cols.append("death_in_30")
        if "hospital_expire_flag" in df.columns and "hospital_expire_flag" not in subset_cols:
            subset_cols.append("hospital_expire_flag")

        run_df = df[subset_cols].dropna(subset=[args.text_col, label])

        # Extra filtering for v2 special tasks:
        # For after_ICU_more_than_3_days / discharge_less_than_12_hours,
        # define the label only on survivors (hospital_expire_flag==0),
        # to avoid semantic conflict with the hospital_expire_flag task.
        if label in ("after_ICU_more_than_3_days", "discharge_less_than_12_hours") and "hospital_expire_flag" in run_df.columns:
            run_df = run_df[run_df["hospital_expire_flag"] == 0].reset_index(drop=True)

        run_one_label(args, run_df, label, tokenizer)


if __name__ == "__main__":
    main()
