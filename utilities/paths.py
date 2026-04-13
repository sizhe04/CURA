import os
from typing import Tuple


def get_experiment_dirs(args, label_col: str) -> Tuple[str, str, str, str]:
    """Return (sanitized_model, base_root, checkpoint_root, eval_root).

    Naming: Loss-{CE|EDL}_Arch-{custom|hf}_Ear-{metric}_Lr-{lr}_Epo-{E}_Warm-{warmup}
    """
    sanitized_model = args.model_name.replace("/", "-")

    loss_tag = str(getattr(args, "loss_type", "ce")).upper()
    arch_tag = str(getattr(args, "classifier_arch", "custom"))
    if bool(getattr(args, "use_early_stopping", False)):
        early_tag = str(getattr(args, "best_metric", "eval_cls_NLL"))
    else:
        early_tag = "NA"
    lr = getattr(args, "learning_rate", None)
    epochs = getattr(args, "num_train_epochs", None)
    warmup = getattr(args, "warmup_steps", None)
    use_lcal = bool(getattr(args, "use_lcal", False))
    lambda_cal = getattr(args, "lambda_cal", 0.0)
    lcal_tag = f"_Lcal-{lambda_cal}" if (use_lcal and float(lambda_cal) > 0.0) else ""
    exp_dir = (
        f"Loss-{loss_tag}_Arch-{arch_tag}{lcal_tag}_Ear-{early_tag}_"
        f"Lr-{lr}_Epo-{epochs}_Warm-{warmup}"
    )

    base_root = os.path.join(args.output_root, sanitized_model, exp_dir, label_col)
    checkpoint_root = os.path.join(base_root, "checkpoint")
    eval_root = os.path.join(base_root, "evaluate")
    return sanitized_model, base_root, checkpoint_root, eval_root
