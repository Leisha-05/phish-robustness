import json
import numpy as np, pandas as pd, joblib, shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# ---------- helpers ----------
def get_feats(df):
    try:
        cols = json.load(open("models/feature_columns.json"))
        if isinstance(cols, dict):
            cols = cols.get("features") or cols.get("columns") or list(cols.values())[0]
        return list(cols)
    except Exception:
        return [c for c in df.columns if c not in ("url_id", "label", "perturbed_url")]

def shap_vals(model, X):
    """(n_samples, n_features) SHAP values for the phishing class.
    Handles sklearn Pipelines by explaining only the final estimator."""
    est, Xt = model, X
    if hasattr(model, "steps"):                    # it's a Pipeline
        est = model.steps[-1][1]
        if len(model.steps) > 1:
            Xt = model[:-1].transform(X)           # run earlier steps (scaler etc.)
    sv = shap.TreeExplainer(est).shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv

def top_k(v, k=5):
    return set(np.argsort(-np.abs(v))[:k])

def stability(sv_a, sv_b):
    rows = []
    for a, b in zip(sv_a, sv_b):
        rho = np.nan if (np.std(a) == 0 or np.std(b) == 0) else spearmanr(a, b).correlation
        ta, tb = top_k(a), top_k(b)
        rows.append((rho, len(ta & tb) / len(ta | tb)))
    return rows

# ---------- part 1: global SHAP ----------
def global_shap(n=2000):
    test = pd.read_parquet("data/processed/features_test.parquet")
    feats = get_feats(test)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(test), size=min(n, len(test)), replace=False)
    sub = test.iloc[idx]
    sub[["url_id"]].to_csv("data/processed/shap_subset_ids.csv", index=False)
    rows = []
    for name in ["xgb", "rf"]:
        print("global SHAP:", name)
        model = joblib.load(f"models/{name}_seed0.joblib")
        sv = shap_vals(model, sub[feats])
        for f, v in zip(feats, np.abs(sv).mean(axis=0)):
            rows.append({"model": name, "feature": f, "mean_abs_shap": v})
        shap.summary_plot(sv, sub[feats], plot_type="bar", max_display=10, show=False)
        plt.tight_layout()
        plt.savefig(f"figures/shap_top10_{name}.png", dpi=300)
        plt.close()
    out = pd.DataFrame(rows).sort_values(["model", "mean_abs_shap"], ascending=[True, False])
    out.to_csv("results/xai_global.csv", index=False)
    print(out.groupby("model").head(10).to_string())

# ---------- part 2: stability ----------
def run_stability(model_name, family, sev, kind="evasion", cap=1000, seed=42):
    ft = pd.read_parquet("data/processed/features_test.parquet")
    feats = get_feats(ft)
    ft = ft.set_index("url_id")
    splits = pd.read_parquet("data/processed/splits.parquet")
    test_urls = splits[splits.split == "test"][["url_id", "url"]]
    fname = f"{family}_{int(sev) if float(sev).is_integer() else str(sev).replace('.', 'p')}"
    pert = pd.read_parquet(f"data/processed/perturbed/{fname}.parquet")
    pred = pd.read_parquet("results/robustness_predictions.parquet")
    p = pred[(pred.model == model_name) & (pred.family == family) &
             (np.isclose(pred.severity, float(sev)))]

    ch = pert[["url_id", "perturbed_url"]].merge(test_urls, on="url_id")
    ch["changed"] = ch.perturbed_url != ch.url
    p = p.merge(ch[["url_id", "changed"]], on="url_id")

    if kind == "evasion":
        base = p[(p.label == 1) & (p.clean_pred == 1) & p.changed]
        flipped, kept = base[base.pert_pred == 0], base[base.pert_pred == 1]
    else:
        base = p[(p.label == 0) & (p.clean_pred == 0) & p.changed]
        flipped, kept = base[base.pert_pred == 1], base[base.pert_pred == 0]

    n = min(len(flipped), cap)
    flipped = flipped.sample(n=n, random_state=seed) if n else flipped
    kept = kept.sample(n=min(len(kept), max(n, 1)), random_state=seed)
    print(model_name, family, sev, "| flipped:", len(flipped), "| kept:", len(kept))
    if len(flipped) == 0:
        return None

    ids = pd.concat([flipped.assign(group="flipped"), kept.assign(group="non_flipped")])
    pert_i = pert.set_index("url_id")
    Xc, Xp = ft.loc[ids.url_id, feats], pert_i.loc[ids.url_id, feats]
    model = joblib.load(f"models/{model_name}_seed0.joblib")

    a1 = ((model.predict_proba(Xc)[:, 1] >= 0.5).astype(int) == ids.clean_pred.values).mean()
    a2 = ((model.predict_proba(Xp)[:, 1] >= 0.5).astype(int) == ids.pert_pred.values).mean()
    print("   prediction agreement clean/pert:", round(a1, 4), round(a2, 4), "(should be ~1.0)")

    res = stability(shap_vals(model, Xc), shap_vals(model, Xp))
    out = pd.DataFrame(res, columns=["spearman", "jaccard5"])
    out["url_id"] = ids.url_id.values
    out["group"] = ids.group.values
    out["model"], out["family"], out["severity"], out["kind"] = model_name, family, float(sev), kind
    return out

if __name__ == "__main__":
    import sys
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("global", "all"):
        global_shap()
    if step in ("stability", "all"):
        parts = []
        for m in ["xgb", "rf"]:
            for s in [2, 3]:
                r = run_stability(m, "path_strip", s, "evasion")
                if r is not None:
                    parts.append(r)
        if parts:
            pd.concat(parts).to_csv("results/xai_stability_samples.csv", index=False)
            print("saved results/xai_stability_samples.csv")