"""train.py - Step C/D/E (Person 1).

Trains Logistic Regression, Random Forest and XGBoost (3 seeds each) as sklearn
Pipelines (StandardScaler + model), saves them to models/, writes clean-test
metrics to results/baseline_metrics.csv and ROC/PR plots to figures/.

If results/best_params.json exists (written by src/tune.py) the tuned
hyperparameters are used for RF/XGB; otherwise (or with --defaults) defaults.

Run from the repo root:  python src/train.py
Optional:                python src/train.py --models lr,rf
                         python src/train.py --defaults
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
THRESHOLD = CFG["threshold"]          # 0.5 everywhere; Person 2's ASR depends on it


def make_model(name: str, seed: int, params: dict | None = None) -> Pipeline:
    if name == "lr":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    elif name == "xgb":
        clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", n_jobs=-1, random_state=seed,
                            eval_metric="logloss")
    else:
        raise ValueError(name)
    if params:                      # tuned hyperparameters from tune.py
        clf.set_params(**params)
    # Same pipeline format for every model so teammates can treat them identically.
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def load_split(name: str):
    df = pd.read_parquet(ROOT / CFG["paths"]["features_dir"] / f"features_{name}.parquet")
    X = df.drop(columns=["url_id", "label"])
    return X, df["label"].values


def metrics(y, proba) -> dict:
    pred = (proba >= THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }


def load_best_params() -> dict:
    f = ROOT / CFG["paths"]["results_dir"] / "best_params.json"
    return json.loads(f.read_text()) if f.exists() else {}


def make_plots(test_scores: dict, y_te, fig_dir: Path, tag: str) -> None:
    """ROC and PR curves on the clean test set (seed-0 model of each type)."""
    fig_dir.mkdir(exist_ok=True)
    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(5, 4.5))
        for name, proba in test_scores.items():
            if kind == "roc":
                x, y, _ = roc_curve(y_te, proba)
                lab = f"{name} (AUC={roc_auc_score(y_te, proba):.4f})"
            else:
                y, x, _ = precision_recall_curve(y_te, proba)
                lab = f"{name} (AP={average_precision_score(y_te, proba):.4f})"
            ax.plot(x, y, label=lab)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            ax.set(xlabel="False positive rate", ylabel="True positive rate",
                   title=f"ROC, clean test set ({tag})", xlim=(0, 0.1), ylim=(0.9, 1.0))
        else:
            ax.set(xlabel="Recall", ylabel="Precision",
                   title=f"Precision-recall, clean test set ({tag})",
                   xlim=(0.9, 1.0), ylim=(0.95, 1.0))
        ax.legend(loc="lower right" if kind == "roc" else "lower left", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{kind}_clean.png", dpi=200)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lr,rf,xgb")
    ap.add_argument("--defaults", action="store_true",
                    help="ignore results/best_params.json and use default hyperparameters")
    args = ap.parse_args()
    names = [m.strip() for m in args.models.split(",")]
    best = {} if args.defaults else load_best_params()
    tag = "tuned" if best else "default"
    print(f"Hyperparameters: {tag}" + (f" (tuned for {sorted(best)})" if best else ""))

    X_tr, y_tr = load_split("train")
    X_te, y_te = load_split("test")
    models_dir = ROOT / CFG["paths"]["models_dir"]
    res_dir = ROOT / CFG["paths"]["results_dir"]
    models_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)

    # Column order contract for Person 2 / 3
    (models_dir / "feature_columns.json").write_text(json.dumps(list(X_tr.columns)))

    rows, test_scores = [], {}
    for name in names:
        for seed in CFG["model_seeds"]:
            model = make_model(name, seed, best.get(name)).fit(X_tr, y_tr)
            joblib.dump(model, models_dir / f"{name}_seed{seed}.joblib")
            proba = model.predict_proba(X_te)[:, 1]
            if seed == CFG["model_seeds"][0]:
                test_scores[name] = proba
            m = metrics(y_te, proba)
            for k, v in m.items():
                rows.append({"model": name, "family": "clean", "severity": 0,
                             "seed": seed, "metric": k, "value": v})
            print(f"{name} seed{seed}: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    out = res_dir / "baseline_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    make_plots(test_scores, y_te, ROOT / CFG["paths"]["figures_dir"], tag)
    print(f"\nSaved models to {models_dir}, metrics to {out}, plots to figures/")


if __name__ == "__main__":
    main()
