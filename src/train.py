"""train.py - Step C/D (Person 1).

Trains Logistic Regression, Random Forest and XGBoost (3 seeds each) as sklearn
Pipelines (StandardScaler + model), saves them to models/, and writes
clean-test metrics to results/baseline_metrics.csv.

Run from the repo root:  python src/train.py
Optional:                python src/train.py --models lr,rf
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
THRESHOLD = CFG["threshold"]          # 0.5 everywhere; Person 2's ASR depends on it


def make_model(name: str, seed: int) -> Pipeline:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lr,rf,xgb")
    args = ap.parse_args()
    names = [m.strip() for m in args.models.split(",")]

    X_tr, y_tr = load_split("train")
    X_te, y_te = load_split("test")
    models_dir = ROOT / CFG["paths"]["models_dir"]
    res_dir = ROOT / CFG["paths"]["results_dir"]
    models_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)

    # Column order contract for Person 2 / 3
    (models_dir / "feature_columns.json").write_text(json.dumps(list(X_tr.columns)))

    rows = []
    for name in names:
        for seed in CFG["model_seeds"]:
            model = make_model(name, seed).fit(X_tr, y_tr)
            joblib.dump(model, models_dir / f"{name}_seed{seed}.joblib")
            m = metrics(y_te, model.predict_proba(X_te)[:, 1])
            for k, v in m.items():
                rows.append({"model": name, "family": "clean", "severity": 0,
                             "seed": seed, "metric": k, "value": v})
            print(f"{name} seed{seed}: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    out = res_dir / "baseline_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved models to {models_dir} and metrics to {out}")


if __name__ == "__main__":
    main()
