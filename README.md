# CURA: Clinical Uncertainty & Risk Alignment

<p align="center">
  <a href="assets/Pipeline_v4.pdf">
    <img src="assets/Pipeline_v4.png" alt="CURA pipeline" width="980">
  </a>
</p>

<p align="center">
  <a href="assets/Pipeline_v4.pdf">Pipeline Figure (PDF)</a> |
  <a href="https://physionet.org/content/mimiciv/">MIMIC-IV</a> |
  <a href="https://physionet.org/content/mimic-iv-note/">MIMIC-IV-Note</a> |
  <a href="#citation">Citation</a>
</p>

Official implementation of **CURA: Clinical Uncertainty & Risk Alignment for Language Model Based Risk Prediction**.

As illustrated in the pipeline above, CURA is a two-stage framework for reliable note-based clinical risk prediction. In **Stage 1**, a clinical language model is fine-tuned on downstream prediction tasks and used to extract frozen patient embeddings. In **Stage 2**, a multi-head classifier is trained on those embeddings with a joint objective that combines predictive accuracy, individual uncertainty calibration, and cohort-aware risk alignment. The goal is not only strong predictive performance, but also uncertainty estimates that better reflect patient-level error likelihood and local cohort ambiguity.

## Highlights

- Two-stage training pipeline aligned with the paper: clinical LM fine-tuning followed by uncertainty fine-tuning.
- Support for multiple clinical backbones, including `BioGPT`, `BioClinicalBERT`, and `ClinicalBERT`.
- Cohort-aware uncertainty modeling through pre-computed neighborhood statistics on Stage 1 embeddings.
- End-to-end scripts for training, embedding export, uncertainty computation, and selective prediction evaluation.

## Code Structure

- `run_stage1.sh`: launches Stage 1 fine-tuning and embedding extraction.
- `run_stage1_repr_uncert.sh`: computes neighborhood statistics on Stage 1 embeddings for cohort-aware Stage 2 training.
- `run_stage2.sh`: launches Stage 2 multi-head uncertainty fine-tuning on frozen embeddings.
- `run_stage2_threshold_eval.sh`: runs selective prediction / threshold-retention evaluation.
- `stage1_cv_train.py`: main 5-fold cross-validation training script for Stage 1.
- `stage1_compute_repr_uncert.py`: computes `q(x)` and neighborhood uncertainty on stored embeddings.
- `stage2_train_from_embeddings.py`: main 5-fold cross-validation training script for Stage 2.
- `stage2_threshold_eval.py`: computes selective prediction metrics across retention thresholds.
- `modeling_singlehead.py`: Stage 1 classifier heads and loss logic.
- `modeling_multihead_embedding.py`: Stage 2 multi-head classifier on frozen embeddings.
- `data.py` and `data_embeddings.py`: dataset loaders for raw note inputs and embedding inputs.
- `params/`: shell parameter templates for Stage 1, Stage 2, representation uncertainty, and threshold evaluation.
- `utilities/`: metrics, training arguments, embedding I/O, W&B helpers, and selective evaluation utilities.
- `compute_positive_rate.py` and `merge_AUROC_AUPRC_results.py`: post-processing helpers for analysis.

## Environment Setup

Recommended environment:

- Python >= 3.9
- PyTorch >= 2.0
- Transformers >= 4.30
- scikit-learn
- pandas
- numpy
- wandb (optional)

Install dependencies with:

```bash
pip install torch transformers scikit-learn pandas numpy wandb
```

## Data

This project uses clinical notes from **MIMIC-IV** and **MIMIC-IV-Note**. Access must be obtained through PhysioNet.

1. Complete the required CITI training.
2. Sign the PhysioNet data use agreement.
3. Request access to [MIMIC-IV](https://physionet.org/content/mimiciv/).
4. Request access to [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/).
5. Build a preprocessed CSV containing the note text column and task labels used by the paper.

Expected columns include:

- `clean_text`
- `death_in_7`
- `death_in_30`
- `Discharge`
- `Hospital`
- `after_ICU_more_than_3_days`

Data files are **not** distributed in this repository.

## How To Run

All parameter files in `params/` are intentionally shipped with blank values. Fill in the paths and experiment settings before running the scripts.

### 1. Stage 1: Clinical LM Fine-Tuning

Choose and edit one of the Stage 1 parameter files, for example:

- `params/Stage1_Bio-ClinicalBERT_30_params.sh`
- `params/Stage1_Bio-ClinicalBERT_7_params.sh`
- `params/Stage1_ClinicalBERT_30_params.sh`
- `params/Stage1_ClinicalBERT_7_params.sh`

Then launch Stage 1:

```bash
bash run_stage1.sh params/Stage1_Bio-ClinicalBERT_30_params.sh
```

This stage fine-tunes the backbone, saves checkpoints, and exports frozen embeddings for each fold.

### 2. Representation Uncertainty on Stage 1 Embeddings

If you plan to use cohort-aware Stage 2 training, fill in `params/Stage1_repr_uncert_params.sh` and run:

```bash
bash run_stage1_repr_uncert.sh params/Stage1_repr_uncert_params.sh
```

This step computes neighborhood statistics such as:

- `q(x)`: local positive rate around each sample
- `H_hat(q)`: local cohort ambiguity / neighborhood entropy

### 3. Stage 2: Uncertainty Fine-Tuning

Fill in `params/Stage2_params.sh`, then run:

```bash
bash run_stage2.sh params/Stage2_params.sh
```

Stage 2 trains the multi-head classifier on frozen embeddings and applies the uncertainty-aware objective described in the paper.

### 4. Selective Prediction Evaluation

Fill in `params/threshold_eval_params.sh`, then run:

```bash
bash run_stage2_threshold_eval.sh params/threshold_eval_params.sh
```

This evaluates performance under different retained-cohort thresholds for selective prediction analysis.

## Output Layout

Stage 2 outputs are organized under the Stage 1 label directory:

```text
{Stage1_label_dir}/Stage2/{experiment_name}/
├── checkpoint/
│   ├── fold_1/
│   ├── fold_2/
│   └── ...
└── evaluate/
    ├── fold_1_val.json
    ├── fold_1_test.json
    ├── fold_1_test_predictions.npz
    ├── val_summary.json
    └── test_summary.json
```

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{wang2025cura,
  title={CURA: Clinical Uncertainty \& Risk Alignment for Language Model Based Risk Prediction},
  author={Wang, Sizhe and Huang, Jiaxin},
  year={2025}
}
```

## License

This repository is intended for research use. Please ensure that all use of clinical data complies with the MIMIC-IV and MIMIC-IV-Note data use agreements.
