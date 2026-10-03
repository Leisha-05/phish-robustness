import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

def boot_ci(x, n=1000, seed=0):
    r = np.random.default_rng(seed)
    m = [r.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])

def holm(pvals):
    p = np.array(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj

df = pd.read_csv("results/xai_stability_samples.csv")
rows = []
for (m, fam, sev, kind), g in df.groupby(["model", "family", "severity", "kind"]):
    for metric in ["spearman", "jaccard5"]:
        f = g[g.group == "flipped"][metric].dropna().values
        k = g[g.group == "non_flipped"][metric].dropna().values
        row = dict(model=m, family=fam, severity=sev, kind=kind, metric=metric,
                   n_flipped=len(f), n_kept=len(k),
                   n_dropped_nan=int((g[metric].isna()).sum()))
        if len(f) >= 20 and len(k) >= 20:
            lo_f, hi_f = boot_ci(f)
            lo_k, hi_k = boot_ci(k)
            row.update(mean_flipped=round(f.mean(), 4), ci_flipped=f"[{lo_f:.3f}, {hi_f:.3f}]",
                       mean_kept=round(k.mean(), 4), ci_kept=f"[{lo_k:.3f}, {hi_k:.3f}]",
                       diff=round(f.mean() - k.mean(), 4),
                       p=mannwhitneyu(f, k, alternative="less").pvalue)
        else:
            row["note"] = "too few flipped samples"
        rows.append(row)

out = pd.DataFrame(rows)
if "p" in out.columns and out["p"].notna().any():
    mask = out["p"].notna()
    out.loc[mask, "p_holm"] = holm(out.loc[mask, "p"])
out.to_csv("results/xai_stats.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(out.to_string())

import pandas as pd 
from scipy.stats import mannwhitneyu
df = pd.read_csv("results/xai_stability_samples.csv")
for (m, s), g in df.groupby(["model", "severity"]):
    f = g[g.group == "flipped"].jaccard5; k = g[g.group == "non_flipped"].jaccard5
    print(m, s, mannwhitneyu(f, k, alternative="two-sided").pvalue)
    
import pandas as pd
df = pd.read_csv("results/xai_stability_samples.csv")
ft = pd.read_parquet("data/processed/features_test.parquet").set_index("url_id")
d = df[(df.model == "xgb") & (df.severity == 3)]
x = ft.loc[d.url_id, ["path_len", "n_slash", "url_len", "is_https", "n_subdomains"]].copy()
x["group"] = d.group.values
print(x.groupby("group").mean())   