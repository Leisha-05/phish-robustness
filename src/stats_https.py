import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

def boot_ci(x, n=1000, seed=0):
    r = np.random.default_rng(seed)
    m = [r.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])

def holm(p):
    p = np.array(p, float); o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

df = pd.read_csv("results/xai_stability_samples.csv")
ft = pd.read_parquet("data/processed/features_test.parquet").set_index("url_id")
df["is_https"] = ft.loc[df.url_id, "is_https"].values
df = df[df.is_https == 1]          # HTTPS-only: removes the confound

rows = []
for (m, s), g in df.groupby(["model", "severity"]):
    for metric in ["spearman", "jaccard5"]:
        f = g[g.group == "flipped"][metric].dropna().values
        k = g[g.group == "non_flipped"][metric].dropna().values
        row = dict(model=m, severity=s, metric=metric, n_flipped=len(f), n_kept=len(k))
        if len(f) >= 20 and len(k) >= 20:
            lf, hf = boot_ci(f); lk, hk = boot_ci(k)
            row.update(mean_flipped=round(f.mean(), 4), ci_flipped=f"[{lf:.3f},{hf:.3f}]",
                       mean_kept=round(k.mean(), 4), ci_kept=f"[{lk:.3f},{hk:.3f}]",
                       diff=round(f.mean() - k.mean(), 4),
                       p_two_sided=mannwhitneyu(f, k, alternative="two-sided").pvalue)
        else:
            row["note"] = "too few samples"
        rows.append(row)

out = pd.DataFrame(rows)
if "p_two_sided" in out.columns:
    mk = out.p_two_sided.notna()
    out.loc[mk, "p_holm"] = holm(out.loc[mk, "p_two_sided"])
out.to_csv("results/xai_stats_https_only.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(out.to_string())