"""tune.py - Step E (Person 1).

Randomized hyperparameter search for Random Forest and XGBoost.

* Fits on TRAIN only and scores on VAL only (PredefinedSplit). Test is never loaded.
* Writes results/best_params.json and results/tuning_results_{rf,xgb}.csv.
* Afterwards run `python src/train.py`: it picks up best_params.json, retrains
  the best configs on seeds 0/1/2, overwrites models/ and baseline_metrics.csv
  and regenerates the ROC/PR plots.

Run from the repo root:  python src/tune.py
Optional:                python src/tune.py --models xgb --n-iter 10
"""
import argparse
import json
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

from train import CFG, ROOT, load_split, make_model

# Lists are sampled uniformly by RandomizedSearchCV. Keys are classifier params;
# they get the "clf__" prefix because every model is a Pipeline.
SPACES = {
    "rf": {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 20, 30, 40],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
    },
    "xgb": {
        "n_estimators": [200, 400, 600],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [1, 5, 10],
    },
}


def _native(o):
    """json-friendly numpy scalars"""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rf,xgb")
    ap.add_argument("--n-iter", type=int, default=15)
    ap.add_argument("--scoring", default="roc_auc",
                    help="sklearn scorer evaluated on VAL (default roc_auc: threshold-free)")
    args = ap.parse_args()
    names = [m.strip() for m in args.models.split(",")]

    seed = CFG["seed"]
    res_dir = ROOT / CFG["paths"]["results_dir"]
    res_dir.mkdir(exist_ok=True)

    # Keep the default-hyperparameter metrics for the paper's "tuning effect" line.
    base, default_copy = res_dir / "baseline_metrics.csv", res_dir / "baseline_metrics_default.csv"
    if base.exists() and not default_copy.exists():
        shutil.copy(base, default_copy)
        print(f"Saved default-hyperparameter metrics to {default_copy.name}")

    # Train and val only. Test is deliberately not loaded in this file.
    X_tr, y_tr = load_split("train")
    X_va, y_va = load_split("val")
    X = pd.concat([X_tr, X_va], ignore_index=True)
    y = np.concatenate([y_tr, y_va])
    # -1 -> always in the training fold; 0 -> the single validation fold
    cv = PredefinedSplit(np.r_[np.full(len(X_tr), -1), np.zeros(len(X_va))])

    best_file = res_dir / "best_params.json"
    best_all = json.loads(best_file.read_text()) if best_file.exists() else {}

    for name in names:
        t0 = time.time()
        # reference point: default hyperparameters, fit on train, scored on val
        ref = make_model(name, seed).fit(X_tr, y_tr)
        ref_auc = roc_auc_score(y_va, ref.predict_proba(X_va)[:, 1])

        search = RandomizedSearchCV(
            make_model(name, seed),
            {f"clf__{k}": v for k, v in SPACES[name].items()},
            n_iter=args.n_iter, scoring=args.scoring, cv=cv,
            refit=False,                  # we retrain over 3 seeds in train.py
            random_state=seed, n_jobs=1,  # the models themselves use all cores
            verbose=2,
        ).fit(X, y)

        cvr = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        cols = ["rank_test_score", "mean_test_score", "mean_fit_time"] + \
               [c for c in cvr.columns if c.startswith("param_")]
        cvr[cols].to_csv(res_dir / f"tuning_results_{name}.csv", index=False)

        best = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
        best_all[name] = best
        best_file.write_text(json.dumps(best_all, indent=2, default=_native))

        print(f"\n[{name}] best val {args.scoring}={search.best_score_:.5f} "
              f"(defaults: val roc_auc={ref_auc:.5f})  [{(time.time() - t0) / 60:.1f} min]")
        print(f"[{name}] best params: {best}\n")

    print(f"Saved {best_file}. Now run:  python src/train.py")


if __name__ == "__main__":
    main()
