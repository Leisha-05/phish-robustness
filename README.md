# phish-robustness

How Brittle Are Lexical Phishing-URL Detectors? Robustness and Explanation Stability Under Controlled URL Perturbations.

All perturbations are applied to dataset strings and evaluated offline. No live URLs are resolved.

## Setup
```bash
pip install -r requirements.txt
```
Google Colab: open `notebooks/00_colab_setup.ipynb`.

## Data
Download PhiUSIIL (UCI) and save as `data/raw/phiusiil.csv`.

## Interface contracts
- Split file rows: `url_id, url, label (1=phish, 0=benign), split`
- Feature files: `url_id, label, <feature columns>`
- Models: `models/{name}_seed{n}.joblib` (sklearn Pipelines taking raw feature frames)
- Result CSVs: `model, family, severity, seed, metric, value`

## Run order
`make data features train perturb eval explain`

## Ownership
| Folder / file | Owner |
|---|---|
| data_prep.py, features.py, train.py | Person 1 |
| perturb.py, evaluate_robustness.py | Person 2 |
| explain.py, stats.py, paper/ | Person 3 |
