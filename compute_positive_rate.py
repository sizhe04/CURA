#!/usr/bin/env python
"""
Quick utility to inspect positive rates for one or more tasks (labels),
reusing the same filtering logic as Stage-1 CV training.

Usage example:

  # Any v1/v2 CSV works as long as data_csv and labels are correct
  python compute_positive_rate.py \\
    --data_csv /path/to/data.csv \\
    --text_col clean_text \\
    --labels death_in_30 death_in_7

The script will:
  - Select columns per Stage-1 logic and drop rows with NaN text/label;
  - For after_ICU_more_than_3_days / discharge_less_than_12_hours,
    count only samples with hospital_expire_flag==0 (matches v2 training);
  - For each label, print sample counts, positive/negative counts, and positive rate.
"""

from typing import List

import numpy as np
import pandas as pd


# ====== Set v1 / v2 data paths here ======
# v1/v2 CSVs only need the columns used in the project (e.g. clean_text, task labels)
V1_DATA_CSV: str = "/Users/wangsizhe/Documents/Research/Projects/LLM_ClinicalNotes/project_code/data/clean_data/clean_data_v1.csv.gz"  # e.g. "/path/to/v1_data.csv"
V2_DATA_CSV: str = "/Users/wangsizhe/Documents/Research/Projects/LLM_ClinicalNotes/project_code/data/clean_data/clean_data_v2.csv.gz"  # e.g. "/path/to/v2_data.csv"
TEXT_COL: str = "clean_text"

# Six tasks used consistently across the project
TASK_LABELS: List[str] = [
    "death_in_30",
    "death_in_7",
    "after_ICU_more_than_3_days",
    "discharge_less_than_12_hours",
    "hospital_expire_flag",
    "ICU_more_than_1_days",
]
# ====================================================


def compute_positive_rate_for_label(df: pd.DataFrame, label_col: str, text_col: str) -> None:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in data.")

    # Match column selection / filtering in stage1_cv_train.py main()
    subset_cols: List[str] = [text_col, label_col]
    for extra in ("note_id", "subject_id"):
        if extra in df.columns and extra not in subset_cols:
            subset_cols.append(extra)
    if "death_in_30" not in subset_cols and "death_in_30" in df.columns:
        subset_cols.append("death_in_30")
    if "hospital_expire_flag" in df.columns and "hospital_expire_flag" not in subset_cols:
        subset_cols.append("hospital_expire_flag")

    run_df = df[subset_cols].dropna(subset=[text_col, label_col]).copy()

    # v2 special tasks: restrict to survivors when defining the label
    if label_col in ("after_ICU_more_than_3_days", "discharge_less_than_12_hours") and "hospital_expire_flag" in run_df.columns:
        run_df = run_df[run_df["hospital_expire_flag"] == 0].reset_index(drop=True)

    # Cast labels to int 0/1
    y = run_df[label_col].astype(int).to_numpy()
    n_total = int(y.shape[0])
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pos_rate = float(n_pos) / float(n_total) if n_total > 0 else float("nan")

    print(f"=== Label: {label_col} ===")
    print(f"Total samples (after Stage-1 filters): {n_total}")
    print(f"Positive (1): {n_pos}")
    print(f"Negative (0): {n_neg}")
    print(f"Positive rate: {pos_rate:.6f}" if np.isfinite(pos_rate) else "Positive rate: nan")
    print("")


def main() -> None:
    if not V1_DATA_CSV and not V2_DATA_CSV:
        raise ValueError(
            "Set V1_DATA_CSV and/or V2_DATA_CSV at the top of compute_positive_rate.py, "
            "then run `python compute_positive_rate.py`."
        )

    # V1
    if V1_DATA_CSV:
        print("========== V1 DATA ==========")
        print(f"CSV: {V1_DATA_CSV}\n")
        df_v1 = pd.read_csv(V1_DATA_CSV, compression="infer")
        for label in TASK_LABELS:
            if label in df_v1.columns:
                compute_positive_rate_for_label(df_v1, label, TEXT_COL)
            else:
                print(f"=== Label: {label} ===\nNOT FOUND in V1 CSV, skip.\n")

    # V2
    if V2_DATA_CSV:
        print("========== V2 DATA ==========")
        print(f"CSV: {V2_DATA_CSV}\n")
        df_v2 = pd.read_csv(V2_DATA_CSV, compression="infer")
        for label in TASK_LABELS:
            if label in df_v2.columns:
                compute_positive_rate_for_label(df_v2, label, TEXT_COL)
            else:
                print(f"=== Label: {label} ===\nNOT FOUND in V2 CSV, skip.\n")


if __name__ == "__main__":
    main()


