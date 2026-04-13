import inspect
from typing import Any, Dict, Optional

from transformers import TrainingArguments


def build_training_arguments(
    args: Any,
    *,
    out_dir: str,
    sanitized_model: str,
    label_col: str,
    fold_id: int,
) -> TrainingArguments:
    """Build TrainingArguments with compatibility handling and save/eval policy.

    Mirrors the logic in cv_train.py without changing behavior.
    """
    ta_params = inspect.signature(TrainingArguments.__init__).parameters

    # evaluation strategy compatibility (must match save_strategy when load_best_model_at_end=True)
    eval_value = "steps" if getattr(args, "eval_strategy", "epoch") == "steps" else "epoch"
    strategy_kwargs: Dict[str, Any] = {}
    if "evaluation_strategy" in ta_params:
        strategy_kwargs["evaluation_strategy"] = eval_value
    elif "eval_strategy" in ta_params:
        strategy_kwargs["eval_strategy"] = eval_value
    elif "evaluate_during_training" in ta_params:
        # Old API doesn't distinguish steps/epoch; turn it on at least
        strategy_kwargs["evaluate_during_training"] = True

    # report_to
    report_to = None
    if getattr(args, "use_wandb", False) and "report_to" in ta_params:
        report_to = ["wandb"]
    elif "report_to" in ta_params:
        report_to = []

    # run_name
    run_name_kw: Dict[str, Any] = {}
    if "run_name" in ta_params:
        base = getattr(args, "wandb_run_name", None) or f"{sanitized_model}_{label_col}"
        run_name_kw["run_name"] = f"{base}_fold{fold_id}"

    # save strategy mapping
    _save_total_limit: Optional[int] = args.save_total_limit if args.save_total_limit > 0 else None
    _save_strategy = (
        ("steps" if args.eval_strategy == "steps" else "epoch")
        if args.save_strategy == "auto"
        else args.save_strategy
    )
    _save_steps: Optional[int] = args.save_steps if _save_strategy == "steps" else None

    extra_kwargs: Dict[str, Any] = {}
    # Newer transformers supports save_safetensors; we disable it explicitly and use
    # pytorch_model.bin uniformly so weights load cleanly via torch.load for best/last checkpoints.
    if "save_safetensors" in ta_params:
        extra_kwargs["save_safetensors"] = False

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy=_save_strategy,
        save_steps=_save_steps,
        eval_steps=(args.eval_steps if eval_value == "steps" else None),
        save_total_limit=_save_total_limit,
        save_only_model=True,
        logging_strategy="steps",
        logging_steps=50,
        fp16=args.fp16,
        **({"report_to": report_to} if report_to is not None else {}),
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model=args.best_metric,
        greater_is_better=bool(args.greater_is_better),
        **strategy_kwargs,
        **run_name_kw,
        **extra_kwargs,
    )

    # Redundant safeguard (older versions may not honor eval_steps via ctor)
    if eval_value == "steps":
        try:
            setattr(training_args, "evaluation_strategy", "steps")
        except Exception:
            pass
        try:
            training_args.eval_steps = args.eval_steps
        except Exception:
            pass

    return training_args


