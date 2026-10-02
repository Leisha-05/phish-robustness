# phish-robustness

**How Brittle Are Lexical Phishing-URL Detectors? Robustness and Explanation Stability Under Controlled URL Perturbations.**

Lexical phishing-URL detectors score near-perfectly on static benchmarks, but those benchmarks do not test a
URL string that has been edited to look more benign. This repository measures how much standard ML detectors
(Logistic Regression, Random Forest, XGBoost, a small MLP) degrade under controlled string-level perturbations, and uses SHAP
to test whether their explanations stay stable when predictions are attacked.

> **Responsible use.** All perturbations are applied to dataset strings and evaluated offline. No URL is ever
> resolved, visited or generated for deployment. The edits are string-level and do not check that a real phishing
> page would still work, so results estimate *model sensitivity*, not real-world attack success.

---

## Everything is reproducible from this repo

No private files are needed. The only external input is the public dataset; everything else
(splits, features, models, results, figures) is rebuilt by the commands below. Seed is `42` everywhere and
dependency versions are pinned.

## 1. Setup

Requires Python 3.11+.

```bash
git clone https://github.com/<your-username>/phish-robustness.git
cd phish-robustness
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\activate
pip install -r requirements.txt
```
Google Colab: open `notebooks/00_colab_setup.ipynb`.

## 2. Get the data

Download the **PhiUSIIL Phishing URL Dataset** from the UCI Machine Learning Repository
(archive.ics.uci.edu, search "PhiUSIIL Phishing URL") and save the CSV as:

```
data/raw/phiusiil.csv
```

The dataset is not redistributed here (`data/raw/` and `data/processed/` are gitignored). Please check the UCI
page for its license and cite it (see Citation). Only the `URL` and `label` columns are used; all HTML-derived
columns are ignored on purpose to avoid near-leakage.

## 3. Run the pipeline

From the repo root, in this order:

| Step | Command | Produces |
|---|---|---|
| 1. Clean + split | `python src/data_prep.py` | `data/processed/splits.parquet`, `results/dataset_stats.csv` |
| 2. Features | `python src/features.py` | `data/processed/features_{train,val,test}.parquet`, `data/processed/tld_map.json` |
| 3. Tune (validation set only) | `python src/tune.py` | `results/best_params.json`, `results/tuning_results_*.csv` |
| 4. Train + baseline | `python src/train.py` | `models/{lr,rf,xgb,mlp}_seed{0,1,2}.joblib`, `models/feature_columns.json`, `results/baseline_metrics.csv`, `figures/roc_clean.png`, `figures/pr_clean.png` |
| 5. Perturb | `python src/perturb.py` | perturbed test sets (in progress) |
| 6. Robustness evaluation | `python src/evaluate_robustness.py` | `results/robustness.csv`, degradation curves, heatmap (in progress) |
| 7. SHAP + stability | `python src/explain.py` | `results/xai_*.csv`, SHAP figures (in progress) |
| 8. Statistics + tables | `python src/stats.py` | final tables (in progress) |

Or run everything with `make all` (`make` is usually not installed on Windows; use the `python` commands there).

Notes:
- `results/best_params.json` is committed, so `python src/train.py` alone (steps 1, 2, 4) rebuilds the exact tuned
  models without re-running the search.
- `python src/train.py --defaults` retrains with default hyperparameters instead.
- `python src/train.py --models mlp` retrains only the MLP and keeps the other models' rows in `baseline_metrics.csv`.
- Steps 1 to 4 take roughly 10 to 15 minutes on a laptop CPU (tuning is about 4 minutes of that).

## 4. Method

**Data.** PhiUSIIL raw URLs. Cleaning: drop missing values, drop URLs longer than 2000 characters, drop exact
duplicate URLs, drop rows whose registered domain cannot be parsed. Labels are remapped to `1 = phishing, 0 = benign`.

**Split.** 70/15/15 train/val/test, **grouped by registered domain**, so no domain appears in more than one split.
The split is saved to disk as `splits.parquet`.

**Features.** 28 lexical features computed from the URL string only by one stateless function,
`extract(url) -> dict` in `src/features.py`: URL/host/path/query lengths, counts of `. - _ @ ? = % / & #` and digits,
host dots/hyphens/digits, subdomain count, path depth, IP-in-host flag, port flag, HTTPS flag, digit and letter
ratios, host character entropy, suspicious-keyword count, and a TLD frequency feature (fitted on train only; unseen
TLDs map to 0). Perturbed URLs are re-featurized with exactly the same code.

**Models.** Logistic Regression, Random Forest, XGBoost and an MLP (two hidden layers of 64 and 32 units, early stopping on an internal 10% hold-out of train, untuned). Each is an sklearn `Pipeline` (StandardScaler + classifier),
trained with seeds 0, 1, 2. RF and XGBoost hyperparameters come from a 15-iteration `RandomizedSearchCV` that fits on
train and scores ROC-AUC on validation only; the test set is never used for selection. The decision threshold is
**fixed at 0.5** everywhere.

**Planned perturbation families** (seeded, deterministic; applied to phishing and benign test URLs):

| Family | Action | Severities |
|---|---|---|
| A. Path padding | append benign-looking path/query tokens | 1 / 3 / 6 tokens |
| B. Subdomain padding | prepend extra subdomain labels | 1 / 2 / 4 labels |
| C. Character edits | insert hyphens, swap digit look-alikes in host/path | 10% / 20% / 40% of eligible characters |
| D. Random control | random same-length noise matched to A/B/C | matched |

**Metrics.** Clean vs perturbed recall, F1, ROC-AUC; **attack success rate (ASR)** = among phishing URLs correctly
detected before perturbation, the fraction classified benign afterwards.

**Explainability.** TreeSHAP for RF/XGBoost (linear explainer for LR) on a fixed 2,000-sample test subset. Stability
of each sample's explanation before vs after perturbation is measured by Spearman correlation and top-5 Jaccard
overlap, compared between flipped and non-flipped predictions (Mann-Whitney test, bootstrap CIs).

**Hypotheses.** H1: tree ensembles beat LR and all models exceed 0.9 AUC. H2: recall degrades monotonically with
severity, and padding/structure edits hurt more than character edits. H3: a few simple features dominate SHAP
importance. H4: explanations are less stable for flipped predictions than for non-flipped ones.

## 5. Results so far

**Clean test set** (36,233 URLs; 15,798 phishing / 20,435 benign), tuned models, seed 0:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9877 | 0.9997 | 0.9721 | 0.9857 | 0.9964 | 0.9969 |
| Random Forest | 0.9910 | 0.9994 | 0.9799 | 0.9896 | 0.9981 | 0.9984 |
| XGBoost | 0.9915 | 0.9998 | 0.9808 | 0.9902 | 0.9987 | 0.9988 |
| MLP | _fill in from `baseline_metrics.csv` after running `--models mlp`_ | | | | | |

Full per-seed numbers are in `results/baseline_metrics.csv`; untuned numbers are in
`results/baseline_metrics_default.csv`. Logistic Regression is deterministic, so its seeds are identical.
Hyperparameter tuning changed test ROC-AUC by only about 0.0005. Baseline accuracy is already near the ceiling,
which is exactly why robustness (not clean accuracy) is the quantity of interest here.

Dataset statistics: `results/dataset_stats.csv`. Robustness and explanation-stability results will be added as steps
5 to 8 are completed.

## 6. Repository layout

```
phish-robustness/
├── README.md
├── requirements.txt          # pinned versions
├── Makefile                  # make data / features / tune / train / perturb / eval / explain
├── configs/config.yaml       # seed, paths, split ratios, threshold, perturbation grid
├── data/
│   ├── raw/                  # (gitignored) phiusiil.csv goes here
│   └── processed/            # (gitignored) splits, features, tld_map.json
├── src/
│   ├── data_prep.py          # cleaning + domain-grouped split
│   ├── features.py           # extract(); frozen feature definition
│   ├── tune.py               # randomized search on validation only
│   ├── train.py              # trains 3 models x 3 seeds, baseline metrics + plots
│   ├── perturb.py            # perturbation families
│   ├── evaluate_robustness.py
│   ├── explain.py            # SHAP + stability
│   └── stats.py
├── models/                   # *.joblib (gitignored, rebuilt by train.py), feature_columns.json
├── results/                  # all CSVs
├── figures/
├── notebooks/                # exploration only
└── paper/
```

## 7. Using the models and extractor in your own code

Contracts used throughout the repo:

- Split file columns: `url_id, url, label, domain, split` (`label` 1 = phishing).
- Feature files: `url_id, label, <28 features>`; the column order is in `models/feature_columns.json`.
- Models: `models/{lr,rf,xgb,mlp}_seed{0,1,2}.joblib`, sklearn Pipelines that take the raw (unscaled) feature frame.
- Result CSVs: `model, family, severity, seed, metric, value` (baseline rows use `family=clean, severity=0`).

```python
# run from the repo root, with src/ on the path (e.g. inside a script in src/)
import json, joblib
from features import featurize, apply_tld_map

cols    = json.load(open("models/feature_columns.json"))
tld_map = json.load(open("data/processed/tld_map.json"))

def to_X(urls):
    return apply_tld_map(featurize(urls), tld_map)[cols]

model = joblib.load("models/xgb_seed0.joblib")
pred  = (model.predict_proba(to_X(["http://example.com/login"]))[:, 1] >= 0.5).astype(int)
```

## 8. Limitations

- Lexical features only; no page content, WHOIS, or network features.
- One dataset (PhiUSIIL); results may not transfer to other URL distributions.
- Perturbations are string-level and do not verify that a perturbed phishing page would still work.
- Near-ceiling clean accuracy means small absolute differences between models are within noise.

## 9. Citation and data attribution

If you use this code, please cite this repository. The data come from:

> Prasad, A., Chandra, S. (2024). *PhiUSIIL: A diverse security profile empowered phishing URL detection framework
> based on similarity index and incremental learning.* Computers & Security.

(Verify the reference and the dataset license on the UCI page before publishing.) Add a `LICENSE` file for the code
(for example MIT) before making the repository public.
