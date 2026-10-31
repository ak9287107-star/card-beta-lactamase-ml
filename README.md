# CARD beta-lactamase machine-learning classification

This repository contains a reproducible Python workflow for exploratory classification of CARD beta-lactamase **drug-class spectrum proxies** using compact physicochemical protein-sequence features.

The project is associated with the study:

> Sequence-derived machine-learning classification of CARD beta-lactamase drug-class spectrum proxies using compact physicochemical features

## What the workflow does

The pipeline:

1. prepares a CARD-derived sequence table;
2. assigns deterministic exploratory spectrum-proxy labels from CARD drug-class relationships;
3. filters and audits ambiguous or incomplete rows;
4. extracts 26 sequence-derived features using Biopython;
5. trains class-balanced random forest, RBF-SVM, and XGBoost models; and
6. evaluates them with family-grouped stratified cross-validation.

The labels are **proxies derived from CARD annotations**. They are not direct experimental hydrolysis measurements, MIC measurements, or clinical phenotype calls.

## Methods represented in this repository

- Python
- Biopython
- pandas and NumPy
- scikit-learn
- XGBoost
- joblib
- matplotlib and seaborn
- Google Colab-compatible command-line scripts
- Family-aware `StratifiedGroupKFold` evaluation

The compact feature representation contains amino-acid composition for the 20 standard residues plus molecular weight, theoretical isoelectric point, aromaticity, instability index, GRAVY hydropathicity, and the S/K/E/D charged-catalytic-residue ratio used in the project.

## Installation

Use Python 3.10 or newer in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The same commands can be run in Google Colab by prefixing them with `!` in a notebook cell.

## Input data

CARD data are not redistributed in this repository. Obtain the appropriate dataset directly from the official CARD source and check its current terms of use. See [`data/README.md`](data/README.md) for the expected input format and provenance requirements.

The preprocessing script expects a CSV containing a protein-sequence column named one of:

```text
sequence, protein_sequence, aa_sequence, seq
```

It can also use optional columns such as `card_short_name`, `aro_name`, `drug_classes`, `amr_gene_family`, or `gene_family` when they are available. The script combines relevant annotation text to create the exploratory spectrum-proxy labels and a grouping column.

## Running the pipeline

From the repository root:

```bash
python src/preprocess_data.py \
  --input data/card_beta_lactamase_sequences.csv \
  --output results/modeling_table.csv \
  --audit results/preprocessing_audit.csv
```

```bash
python src/extract_features.py \
  --input results/modeling_table.csv \
  --output results/feature_matrix.csv \
  --exclusions results/feature_exclusion_log.csv
```

```bash
python src/train_models.py \
  --input results/feature_matrix.csv \
  --output-dir results/models \
  --splits 5 \
  --seed 42
```

```bash
python src/evaluate_models.py \
  --model-dir results/models \
  --output-dir results/evaluation \
  --splits 5
```

The scripts save derived tables, fold assignments, serialized models, metrics, classification reports, confusion matrices, and feature-importance outputs under `results/`.

## Reproducibility notes

- The random seed defaults to 42.
- Group identifiers should represent CARD AMR gene families or another independently justified family/identity grouping.
- Do not replace unknown or ambiguous annotations with guessed labels.
- The proxy-label rule is deliberately auditable and should be reviewed before any new CARD release is analyzed.
- The reported performance should not be interpreted as validated enzyme activity or clinical prediction.

## Citation

If you use this code, cite the associated study and the CARD resource. A machine-readable citation template is provided in [`CITATION.cff`](CITATION.cff).

## License

The original source code in this repository is released under the MIT License. CARD data and derived annotations remain subject to the terms of their original source.
