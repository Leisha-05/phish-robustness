"""Person 2: generate perturbed test sets and evaluate all saved models.

Usage (from repo root):
    python src/evaluate_robustness.py generate    # perturb + re-featurize test set
    python src/evaluate_robustness.py evaluate    # score every model on every set
    python src/evaluate_robustness.py all

READ-ONLY on Person 1's files (splits, features, models). Writes only:
    data/processed/perturbed/{family}_{severity}.parquet
    results/perturbation_stats.csv
    results/robustness.csv                (model,family,severity,seed,metric,value)
    results/robustness_predictions.parquet (per-sample, for Person 3)
"""
import argparse
import re
import sys
from multiprocessing import Pool
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perturb import ALL_FAMILIES, CONTROL_OF, perturb  # noqa: E402

try:
    from features import extract, apply_tld_map  # noqa: E402
except ImportError:  # pragma: no cover
   from src.features import extract, apply_tld_map  # noqa: E402

# ---- PATHS: adjust after `ls data/processed models` if names differ ----------
PROC = ROOT / "data" / "processed"
SPLITS_CANDIDATES = ["splits.parquet", "splits.csv", "data_splits.parquet",
                     "clean.parquet", "clean.csv", "dataset.parquet"]
FEATURES_TEST = PROC / "features_test.parquet"
PERT_DIR = PROC / "perturbed"
MODELS_DIR = ROOT / "models"
RESULTS = ROOT / "results"
ID_COLS = ["url_id", "label"]
TLD_MAP = PROC / "tld_map.json"
# -------------------------------------------------------------------------------


def load_cfg():
    return yaml.safe_load(open(ROOT / "configs" / "config.yaml"))


def find_key(d, key):
    """Recursively find a key (e.g. 'threshold') in nested config."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find_key(v, key)
            if r is not None:
                return r
    return None


SEV_KEYS = {
    "path_padding": "path_padding_tokens",
    "subdomain_padding": "subdomain_labels",
    "char_edit": "char_edit_fraction",
}

def severities(cfg, family):
    base = CONTROL_OF.get(family, family)

    if base == "path_strip":
        return [1, 2, 3]

    return cfg["perturb"][SEV_KEYS[base]]


def sev_tag(s):
    return str(s).replace(".", "p")


def load_test_urls():
    for name in SPLITS_CANDIDATES:
        p = PROC / name
        if p.exists():
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            if {"url_id", "url", "split"} <= set(df.columns):
                return df[df["split"] == "test"][["url_id", "url", "label"]]
    raise FileNotFoundError(
        "Could not find the splits file with url_id,url,label,split. "
        f"Edit SPLITS_CANDIDATES at the top of this script. Looked in {PROC}")


def _featurize(args):
    url, tld_map = args
    raw = pd.DataFrame([extract(url)])
    return apply_tld_map(raw, tld_map).iloc[0].to_dict()


def generate(cfg, workers):
    seed = cfg["seed"]
    test_urls = load_test_urls()
    clean = pd.read_parquet(FEATURES_TEST)
    tld_map = json.loads(TLD_MAP.read_text())
    feat_cols = [c for c in clean.columns if c not in ID_COLS]
    # align URL rows with feature rows by url_id
    test_urls = test_urls.set_index("url_id").loc[clean["url_id"]].reset_index()
    PERT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    # sanity: extract() on clean URLs must reproduce Person 1's features
    chk_raw = pd.DataFrame(
        [extract(u) for u in test_urls["url"].iloc[:200]]
    )

    chk = apply_tld_map(chk_raw, tld_map)
    chk = chk.reindex(columns=feat_cols).astype(float).values
    ref = clean[feat_cols].iloc[:200].astype(float).values
    if not np.allclose(chk, ref, equal_nan=True, atol=1e-6):
        print("WARNING: extract() does not reproduce features_test.parquet on "
              "the first 200 rows. Ask Person 1 before continuing.")

    stats = []
    with Pool(workers) as pool:
        for fam in ALL_FAMILIES:
            for sev in severities(cfg, fam):
                perturbed = [perturb(u, fam, sev, seed) for u in test_urls["url"]]
                changed = float(np.mean([a != b for a, b in
                                         zip(perturbed, test_urls["url"])]))
                rows = pool.map(
                                _featurize,
                                [(u, tld_map) for u in perturbed],
                                chunksize=500,
                            )
                feats = pd.DataFrame(rows).reindex(columns=feat_cols)
                out = pd.concat([test_urls[["url_id", "label"]].reset_index(drop=True),
                                 feats], axis=1)
                out["perturbed_url"] = perturbed
                out.to_parquet(PERT_DIR / f"{fam}_{sev_tag(sev)}.parquet")
                stats.append(dict(family=fam, severity=sev, frac_changed=changed,
                                  mean_len_delta=float(np.mean(
                                      [len(a) - len(b) for a, b in
                                       zip(perturbed, test_urls["url"])]))))
                print(f"generated {fam} {sev}  changed={changed:.2%}")
    pd.DataFrame(stats).to_csv(RESULTS / "perturbation_stats.csv", index=False)


def load_model(path):
    obj = joblib.load(path)
    if isinstance(obj, dict):
        obj = obj.get("model", obj.get("pipeline", obj))
    return obj


def prob_pos(model, X):
    return model.predict_proba(X)[:, 1]


def metrics(y, prob, thr):
    pred = prob >= thr
    neg = (y == 0)
    return dict(
        accuracy=accuracy_score(y, pred),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred),
        f1=f1_score(y, pred),
        roc_auc=roc_auc_score(y, prob),
        fpr=float(((pred) & neg).sum() / max(neg.sum(), 1)),
    )


def evaluate(cfg):
    thr = float(find_key(cfg, "threshold") or 0.5)
    print(f"threshold = {thr}")
    clean = pd.read_parquet(FEATURES_TEST)
    feat_cols = [c for c in clean.columns if c not in ID_COLS]
    y = clean["label"].values
    model_files = sorted(MODELS_DIR.glob("*_seed*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No models found in {MODELS_DIR}")

    pert_sets = []
    for fam in ALL_FAMILIES:
        for sev in severities(cfg, fam):
            df = pd.read_parquet(PERT_DIR / f"{fam}_{sev_tag(sev)}.parquet")
            assert (df["url_id"].values == clean["url_id"].values).all()
            pert_sets.append((fam, sev, df[feat_cols].astype(float)))

    rows, preds = [], []
    first_seed = {}
    for mf in model_files:
        m = re.match(r"(.+)_seed(\d+)\.joblib$", mf.name)
        name, seed = m.group(1), int(m.group(2))
        first_seed.setdefault(name, seed)
        first_seed[name] = min(first_seed[name], seed)
        model = load_model(mf)
        Xc = clean[feat_cols].astype(float)
        p_clean = prob_pos(model, Xc)
        clean_pred = p_clean >= thr
        correct_phish = (y == 1) & clean_pred
        for k, v in metrics(y, p_clean, thr).items():
            rows.append((name, "clean", 0, seed, k, v))
        rows.append((name, "clean", 0, seed, "asr", 0.0))

        for fam, sev, Xp in pert_sets:
            p_pert = prob_pos(model, Xp)
            for k, v in metrics(y, p_pert, thr).items():
                rows.append((name, fam, sev, seed, k, v))
            asr = float((p_pert[correct_phish] < thr).mean()) if correct_phish.any() else np.nan
            rows.append((name, fam, sev, seed, "asr", asr))
            if seed == first_seed[name]:
                preds.append(pd.DataFrame(dict(
                    url_id=clean["url_id"].values, model=name, seed=seed,
                    family=fam, severity=sev, label=y,
                    clean_prob=p_clean, pert_prob=p_pert,
                    clean_pred=clean_pred.astype(int),
                    pert_pred=(p_pert >= thr).astype(int))))
        print(f"evaluated {mf.name}")

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["model", "family", "severity", "seed",
                                "metric", "value"]).to_csv(
        RESULTS / "robustness.csv", index=False)
    pd.concat(preds).to_parquet(RESULTS / "robustness_predictions.parquet")
    print("wrote results/robustness.csv and results/robustness_predictions.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["generate", "evaluate", "all"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = load_cfg()
    if a.step in ("generate", "all"):
        generate(cfg, a.workers)
    if a.step in ("evaluate", "all"):
        evaluate(cfg)