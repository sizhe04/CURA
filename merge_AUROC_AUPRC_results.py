#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Read evaluate/threshold_summary.json under each Stage-2 experiment directory
and write two CSV files:
    1) *_threshold_AUROC.csv
    2) *_threshold_AUPRC.csv

Rows: Method (one row per experiment)
Columns: coverage/threshold values (e.g. 0.00, 0.05, ... , 1.00 = fraction of most-certain samples retained)
Cells: mean (std), three decimals, one space between value and parentheses.
"""

import os
import json
import csv
import math

# ========= Modify these paths to point to your Stage 2 results =========
STAGE2_ROOT = "/path/to/Stage1/.../Stage2"

OUT_AUROC_CSV = "/path/to/Stage1/.../Stage2/threshold_AUROC.csv"

OUT_AUPRC_CSV = "/path/to/Stage1/.../Stage2/threshold_AUPRC.csv"
# ======================================================================


def safe_float(x):
    """Safely format a possibly-NaN number as float or empty string."""
    try:
        if x is None:
            return ""
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return ""
        return float(x)
    except Exception:
        return ""


def fmt_mean_std(mean_val, std_val):
    """
    Format mean / std like '0.706 (0.025)' with three decimal places.
    Return empty string if NaN or missing.
    """
    m = safe_float(mean_val)
    s = safe_float(std_val)
    if m == "" or s == "":
        return ""
    return f"{m:.3f} ({s:.3f})"


def parse_exp_name(exp_name):
    """
    Same parsing rules as in your earlier script:
    - returns method name
    - group for sorting
    - sample_used(%) for sorting
    """
    parts = exp_name.split("_")
    if len(parts) < 5:
        return {
            "method": exp_name,
            "group": 99,
            "sample_used": 100,
        }

    # Whether Active learning
    act_part = parts[0]  # "Active-batch" / "Active-none"
    act_mode = act_part.split("-", 1)[1] if "-" in act_part else act_part
    is_active = act_mode != "none"

    # Selection fraction p
    select_part = parts[1]  # "Select-0.3" / "Select-NA"
    select_val = select_part.split("-", 1)[1] if "-" in select_part else "NA"
    if select_val == "NA":
        p = None
    else:
        try:
            p = float(select_val)
        except ValueError:
            p = None

    # L_div / L_cal coefficients
    div_part = parts[3]   # "DIV-0.0" / "DIV-0.5"
    cal_part = parts[4]   # "CAL-0.0" / "CAL-0.1"
    try:
        div_val = float(div_part.split("-", 1)[1])
    except Exception:
        div_val = 0.0
    try:
        cal_val = float(cal_part.split("-", 1)[1])
    except Exception:
        cal_val = 0.0
    mod_loss = (div_val > 0.0) or (cal_val > 0.0)

    # Uncertainty type: MI / Var / Random
    tail = parts[-1]  # "epis-mi" / "epis-var" / "rdm"
    if tail == "epis-mi":
        score_type = "MI"
    elif tail == "epis-var":
        score_type = "Var"
    elif tail == "rdm":
        score_type = "Random"
    else:
        score_type = "Unknown"

    # Percentage of samples used
    if is_active and p is not None:
        sample_pct = round(p * 100)
    else:
        sample_pct = 100

    # Grouping rules (same as before)
    if (not is_active) and (not mod_loss):
        method = "BaseLine"
        group = 0
    elif (not is_active) and mod_loss:
        method = "Only Modified Loss"
        group = 4
    else:
        if score_type == "MI":
            base = f"Active Learning {sample_pct}% MI"
        elif score_type == "Var":
            base = f"Active Learning {sample_pct}% Var"
        elif score_type == "Random":
            base = f"Random {sample_pct}%"
        else:
            base = exp_name

        if mod_loss:
            base += " + Modified Loss"

        method = base

        if not mod_loss:
            group = {"MI": 1, "Var": 2, "Random": 3}.get(score_type, 99)
        else:
            group = {"MI": 5, "Var": 6, "Random": 7}.get(score_type, 99)

    return {
        "method": method,
        "group": group,
        "sample_used": sample_pct,
    }


def collect_threshold_results(stage2_root, out_auroc_csv, out_auprc_csv):
    rows_auroc = []
    rows_auprc = []

    all_thresholds = None  # threshold list shared across all experiments

    for exp_name in sorted(os.listdir(stage2_root)):
        exp_dir = os.path.join(stage2_root, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        # Skip directories with purely numeric names
        if exp_name.isdigit():
            continue

        thr_path = os.path.join(exp_dir, "evaluate", "threshold_summary.json")
        if not os.path.isfile(thr_path):
            continue

        try:
            with open(thr_path, "r") as f:
                thr_json = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {thr_path}: {e}")
            continue

        thresholds_obj = thr_json.get("thresholds", {})
        if not thresholds_obj:
            continue

        # Backward compatibility: thresholds may be a list (single variant) or dict[variant -> list]
        if isinstance(thresholds_obj, dict):
            variants = sorted(thresholds_obj.keys())
            variant_to_thresholds = thresholds_obj
        else:
            variants = ["default"]
            variant_to_thresholds = {"default": thresholds_obj}

        # Use the first variant's threshold list as the baseline for this experiment
        first_variant = variants[0]
        thresholds = variant_to_thresholds[first_variant]
        if not thresholds:
            continue

        # Build / validate threshold sequence (assumed identical across experiments)
        cur_thresholds = [float(t["threshold"]) for t in thresholds]
        if all_thresholds is None:
            all_thresholds = cur_thresholds
        else:
            if len(all_thresholds) != len(cur_thresholds) or any(
                abs(a - b) > 1e-8 for a, b in zip(all_thresholds, cur_thresholds)
            ):
                print(f"[WARN] thresholds mismatch in {exp_name}, skip.")
                continue

        # Ensure all variants in this experiment use the same thresholds
        for v in variants[1:]:
            t_list = variant_to_thresholds.get(v, [])
            v_thresholds = [float(t["threshold"]) for t in t_list]
            if len(v_thresholds) != len(cur_thresholds) or any(
                abs(a - b) > 1e-8 for a, b in zip(cur_thresholds, v_thresholds)
            ):
                print(f"[WARN] thresholds mismatch between variants in {exp_name}, skip.")
                variants = []
                break
        if not variants:
            continue

        info = parse_exp_name(exp_name)
        base_method = info["method"]
        group = info["group"]
        sample_used = info["sample_used"]

        # One row per variant; if JSON has both MI and Var, emit two rows with suffix on method name
        multi_variant = len(variants) > 1
        for v in variants:
            t_list = variant_to_thresholds.get(v, [])
            if not t_list:
                continue

            if multi_variant:
                # Variant name for labeling, e.g. (MI)/(Var)
                v_label = v.upper()
                method = f"{base_method} ({v_label})"
            else:
                method = base_method

            row_auc = {
                "Method": method,
                "_group": group,
                "_sort_sample": sample_used,
            }
            row_auprc = {
                "Method": method,
                "_group": group,
                "_sort_sample": sample_used,
            }

            # Write one cell per threshold
            for t in t_list:
                thr_val = float(t["threshold"])
                # Keep two decimals in column names so 0.05 is not rounded to 0.1
                col_name = f"{thr_val:.2f}"  # e.g. 0.00, 0.05, ... , 1.00

                row_auc[col_name] = fmt_mean_std(t.get("auroc_mean"), t.get("auroc_std"))
                row_auprc[col_name] = fmt_mean_std(t.get("auprc_mean"), t.get("auprc_std"))

            rows_auroc.append(row_auc)
            rows_auprc.append(row_auprc)

    if not rows_auroc:
        print("[INFO] No threshold_summary.json found.")
        return

    # Sort: group, then sample used, then method name
    rows_auroc.sort(key=lambda r: (r["_group"], r["_sort_sample"], r["Method"]))
    rows_auprc.sort(key=lambda r: (r["_group"], r["_sort_sample"], r["Method"]))

    # Column order matches all_thresholds, two decimal places
    threshold_cols = [f"{t:.2f}" for t in all_thresholds]
    fieldnames = ["Method"] + threshold_cols

    os.makedirs(os.path.dirname(out_auroc_csv), exist_ok=True)

    # Write AUROC CSV
    with open(out_auroc_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_auroc:
            r = dict(r)
            r.pop("_group", None)
            r.pop("_sort_sample", None)
            writer.writerow(r)

    # Write AUPRC CSV
    with open(out_auprc_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_auprc:
            r = dict(r)
            r.pop("_group", None)
            r.pop("_sort_sample", None)
            writer.writerow(r)

    print(f"[INFO] Collected {len(rows_auroc)} experiments into:")
    print(f"       AUROC CSV : {out_auroc_csv}")
    print(f"       AUPRC CSV : {out_auprc_csv}")


if __name__ == "__main__":
    collect_threshold_results(STAGE2_ROOT, OUT_AUROC_CSV, OUT_AUPRC_CSV)
