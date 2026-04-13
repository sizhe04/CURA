# CURA: Clinical Uncertainty & Risk Alignment

Official implementation of **CURA: Clinical Uncertainty & Risk Alignment for Language Model Based Risk Prediction**.

CURA is a two-stage framework that improves the calibration and reliability of clinical language models for risk prediction on electronic health records. It achieves this through a novel uncertainty fine-tuning objective that jointly optimizes individual-level uncertainty calibration and cohort-aware risk alignment.

## Overview

Clinical language models fine-tuned on EHR notes often produce overconfident predictions, especially for rare adverse events. CURA addresses this with a two-stage pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Clinical LM Fine-Tuning                          │
│  ─────────────────────────────────                          │
│  Fine-tune a pre-trained clinical LM (e.g., BioGPT,        │
│  BioClinicalBERT) on clinical notes with standard CE loss.  │
│  Extract frozen patient embeddings from the fine-tuned LM.  │
│  Optionally compute neighborhood statistics (q, H_hat)      │
│  for cohort-aware training.                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │  frozen embeddings
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Uncertainty Fine-Tuning                          │
│  ────────────────────────────────                          │
│  Train a K-head MLP classifier on frozen embeddings with    │
│  a composite loss:                                          │
│                                                             │
│    L_total = L_base + λ_ind · L_ind + λ_coh · L_coh        │
│                                                             │
│  • L_base: class-weighted cross-entropy                     │
│  • L_ind:  individual uncertainty calibration               │
│  • L_coh:  cohort-aware risk alignment via neighbor labels  │
└─────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
.
├── run_stage1.sh                  # Launch Stage 1: LM fine-tuning + embedding extraction
├── run_stage1_repr_uncert.sh      # Compute neighborhood stats (q, H_hat) on embeddings
├── run_stage2.sh                  # Launch Stage 2: uncertainty fine-tuning
├── run_stage2_threshold_eval.sh   # Selective prediction evaluation
│
├── stage1_cv_train.py             # Stage 1 training script (5-fold CV)
├── stage1_compute_repr_uncert.py  # Compute representation-space uncertainty
├── stage2_train_from_embeddings.py  # Stage 2 training script (5-fold CV)
├── stage2_threshold_eval.py       # Selective prediction evaluation script
│
├── modeling_singlehead.py         # Single-head classifier for Stage 1
├── modeling_multihead_embedding.py  # Multi-head classifier for Stage 2
├── data.py                        # Dataset class for Stage 1 (tokenized text)
├── data_embeddings.py             # Dataset class for Stage 2 (frozen embeddings)
│
├── params/                        # Configuration files (fill in values before running)
│   ├── Stage1_Bio-ClinicalBERT_30_params.sh
│   ├── Stage1_Bio-ClinicalBERT_7_params.sh
│   ├── Stage1_ClinicalBERT_30_params.sh
│   ├── Stage1_ClinicalBERT_7_params.sh
│   ├── Stage1_repr_uncert_params.sh
│   ├── Stage2_params.sh
│   └── threshold_eval_params.sh
│
├── utilities/
│   ├── metrics.py                 # AUROC, AUPRC, ECE, Brier, NLL, etc.
│   ├── training_args.py           # HuggingFace TrainingArguments builder
│   ├── embeddings_io.py           # Load/save embedding .pt files
│   ├── repr_uncert.py             # Neighborhood uncertainty (q, H_hat, w)
│   ├── wandb_utils.py             # Weights & Biases integration
│   ├── selective_eval.py          # Selective prediction utilities
│   ├── io_utils.py                # General I/O helpers
│   ├── logging_utils.py           # Logging helpers
│   └── paths.py                   # Path utilities
│
├── compute_positive_rate.py       # Post-processing: compute label positive rates
└── merge_AUROC_AUPRC_results.py   # Post-processing: aggregate results across folds
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- Transformers >= 4.30
- scikit-learn
- pandas
- numpy
- wandb (optional, for experiment tracking)

Install dependencies:

```bash
pip install torch transformers scikit-learn pandas numpy wandb
```

## Data Preparation

This project uses clinical notes from the **MIMIC-IV** database. MIMIC-IV is a freely available, de-identified clinical database maintained by the MIT Laboratory for Computational Physiology.

### Accessing MIMIC-IV

1. Complete the [CITI Program](https://about.citiprogram.org/) "Data or Specimens Only Research" course.
2. Sign the data use agreement on [PhysioNet](https://physionet.org/content/mimiciv/).
3. Download the MIMIC-IV dataset from https://physionet.org/content/mimiciv/.
4. For clinical notes specifically, access the MIMIC-IV-Note module: https://physionet.org/content/mimic-iv-note/.

### Pre-processing

After obtaining access, pre-process the clinical notes into a single CSV file with the following columns:

| Column | Description |
|---|---|
| `clean_text` | Pre-processed clinical note text |
| `death_in_7` | Binary label: 7-day mortality |
| `death_in_30` | Binary label: 30-day mortality |
| `Discharge` | Binary label: early discharge |
| `Hospital` | Binary label: in-hospital mortality |
| `after_ICU_more_than_3_days` | Binary label: ICU stay > 1 day |

Place the resulting CSV at `./data/clean_data/clean_data_v1.csv.gz`.

> **Note:** Data files are not included in this repository due to the data use agreement. You must obtain MIMIC-IV access independently.

## Usage

### Stage 1: Clinical LM Fine-Tuning

Stage 1 fine-tunes a pre-trained clinical language model on clinical notes using standard cross-entropy loss, then extracts frozen patient embeddings.

1. Copy and fill in a Stage 1 config file under `params/` (see existing templates for reference). Key parameters to set:

```bash
DATA_CSV="./data/clean_data/clean_data_v1.csv.gz"
OUTPUT_ROOT="./results/Stage1"
MODEL_NAME="emilyalsentzer/Bio_ClinicalBERT"  # or "microsoft/BioGPT", "medicalai/ClinicalBERT"
TARGET_LABELS="death_in_30"
```

2. Run Stage 1:

```bash
bash run_stage1.sh params/Stage1_Bio-ClinicalBERT_30_params.sh
```

This will:
- Fine-tune the LM with 5-fold cross-validation
- Extract embeddings from each fold's best and last checkpoints
- Optionally compute neighborhood uncertainty statistics (U_repr)

### Stage 1.5: Compute Neighborhood Statistics (Required for L_coh)

If using the cohort-aware loss (L_coh, `LAMBDA_POP > 0`), you must pre-compute neighborhood statistics on the frozen embeddings before running Stage 2:

```bash
bash run_stage1_repr_uncert.sh params/Stage1_repr_uncert_params.sh
```

This computes for each sample:
- **q(x)**: neighborhood positive rate (soft label from k nearest neighbors)
- **H_hat(q)**: neighborhood entropy (measures local class ambiguity)

### Stage 2: Uncertainty Fine-Tuning

Stage 2 trains a multi-head MLP classifier on the frozen embeddings from Stage 1. The training objective combines three loss terms:

- **L_base**: Class-weighted cross-entropy for predictive accuracy
- **L_ind** (controlled by `LAMBDA_CAL`): Aligns the model's internal uncertainty with its actual error likelihood
- **L_coh** (controlled by `LAMBDA_POP`): Regularizes predictions via local consistency with neighborhood event rates

1. Fill in `params/Stage2_params.sh` with your paths and hyperparameters. Key parameters:

```bash
# Point to Stage 1 checkpoint directory
EMBEDDINGS_ROOT="/path/to/Stage1/.../checkpoint"
# If using L_coh, use embeddings with pre-computed neighbor stats
EMBEDDING_SOURCE="embeddings_last_llm_uncertain_K-200"
```

2. Run Stage 2:

```bash
bash run_stage2.sh params/Stage2_params.sh
```

### Selective Prediction Evaluation

To evaluate model performance at various retention rates (selective prediction):

```bash
bash run_stage2_threshold_eval.sh params/threshold_eval_params.sh
```

## Results

Stage 2 results are saved under the Stage 1 label directory:

```
{Stage1_label_dir}/Stage2/{exp_name}/
├── checkpoint/
│   ├── fold_1/          # Model checkpoints and training logs
│   ├── fold_2/
│   └── ...
└── evaluate/
    ├── fold_1_val.json   # Validation metrics per fold
    ├── fold_1_test.json  # Test metrics per fold
    ├── fold_1_test_predictions.npz  # Per-sample predictions
    ├── val_summary.json  # Cross-fold validation summary (mean +/- std)
    └── test_summary.json # Cross-fold test summary (mean +/- std)
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{wang2025cura,
  title={CURA: Clinical Uncertainty \& Risk Alignment for Language Model Based Risk Prediction},
  author={Wang, Sizhe and Huang, Jiaxin},
  year={2025}
}
```

## License

This project is for research purposes only. Please ensure compliance with the MIMIC-IV data use agreement when using this code with clinical data.
