import numpy as np, pandas as pd, joblib
from scipy.stats import spearmanr
from explain import get_feats, shap_vals

ft = pd.read_parquet("data/processed/features_test.parquet")
feats = get_feats(ft); ft = ft.set_index("url_id")
sp_ = pd.read_parquet("data/processed/splits.parquet")
urls = sp_[sp_.split == "test"].set_index("url_id").url
pert = pd.read_parquet("data/processed/perturbed/path_strip_3.parquet").set_index("url_id")
st = pd.read_csv("results/xai_stability_samples.csv")
st = st[(st.model == "xgb") & (st.severity == 3)]
st["is_https"] = ft.loc[st.url_id, "is_https"].values
st = st[st.is_https == 1]

# Check A: how many perturbations are only a removed trailing slash?
orig = urls.loc[st.url_id].values
newu = pert.loc[st.url_id, "perturbed_url"].values
st["slash_only"] = [o.rstrip("/") == n for o, n in zip(orig, newu)]
print(st.groupby("group").slash_only.mean())

# Check B: Spearman on ABSOLUTE SHAP values instead of signed
model = joblib.load("models/xgb_seed0.joblib")
ids = st.url_id.values
sc = shap_vals(model, ft.loc[ids, feats]); sp = shap_vals(model, pert.loc[ids, feats])
def rho(a, b):
    a, b = np.abs(a), np.abs(b)
    return np.nan if a.std() == 0 or b.std() == 0 else spearmanr(a, b).correlation
st["spearman_abs"] = [rho(a, b) for a, b in zip(sc, sp)]
print(st.groupby("group")[["spearman", "spearman_abs"]].mean().round(3))