import numpy as np, pandas as pd, joblib
from explain import get_feats, shap_vals

ft = pd.read_parquet("data/processed/features_test.parquet")
feats = get_feats(ft); ft = ft.set_index("url_id")
splits = pd.read_parquet("data/processed/splits.parquet")
urls = splits[splits.split == "test"].set_index("url_id").url
pert = pd.read_parquet("data/processed/perturbed/path_strip_3.parquet").set_index("url_id")
st = pd.read_csv("results/xai_stability_samples.csv")
st = st[(st.model == "xgb") & (st.severity == 3)]
st["is_https"] = ft.loc[st.url_id, "is_https"].values
st = st[st.is_https == 1]

model = joblib.load("models/xgb_seed0.joblib")
ids = st.url_id.values
sc = shap_vals(model, ft.loc[ids, feats])
sp = shap_vals(model, pert.loc[ids, feats])
delta = pd.DataFrame(sp - sc, columns=feats)
delta["group"] = st.group.values

# 1) mean SHAP shift (perturbed - clean) per feature, by group
tab = delta.groupby("group")[feats].mean().T
tab["abs_diff"] = (tab.flipped - tab.non_flipped).abs()
print(tab.sort_values("abs_diff", ascending=False).head(10).round(3))
tab.round(4).to_csv("results/xai_shap_shift_xgb_sev3.csv")

# 2) five flipped examples with their biggest SHAP shifts
fl = st[st.group == "flipped"].sample(5, random_state=0)
for i, uid in enumerate(fl.url_id):
    j = list(ids).index(uid)
    top = np.argsort(-np.abs(sp[j] - sc[j]))[:3]
    print("\nCLEAN:", urls.loc[uid])
    print("PERT :", pert.loc[uid, "perturbed_url"])
    for t in top:
        print(f"   {feats[t]}: {sc[j][t]:+.2f} -> {sp[j][t]:+.2f}")