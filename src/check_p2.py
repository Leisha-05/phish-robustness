import pandas as pd
pred = pd.read_parquet("results/robustness_predictions.parquet")
sp = pd.read_parquet("data/processed/splits.parquet")
test = sp[sp.split == "test"][["url_id", "url"]]
for lvl in [1, 2, 3]:
    pert = pd.read_parquet(f"data/processed/perturbed/path_strip_{lvl}.parquet")[["url_id", "perturbed_url"]].merge(test, on="url_id")
    pert["changed"] = pert.perturbed_url != pert.url
    for m in ["lr", "rf", "xgb", "mlp"]:
        p = pred[(pred.model == m) & (pred.family == "path_strip") & (pred.severity == lvl)].merge(pert[["url_id", "changed"]], on="url_id")
        det = p[(p.label == 1) & (p.clean_pred == 1)]
        ch = det[det.changed]
        print(lvl, m, "changed share", round(len(ch)/len(det), 3), "ASR among changed", round((ch.pert_pred == 0).mean(), 3))
ft = pd.read_parquet("data/processed/features_train.parquet")
print(ft.groupby("label")[["path_len", "query_len"]].mean())
print("benign with no path:", (ft[ft.label == 0].path_len == 0).mean())