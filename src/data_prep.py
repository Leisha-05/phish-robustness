"""data_prep.py - Step 3 (Person 1).

Reads the raw PhiUSIIL CSV, keeps only URL + label, cleans it, and writes a
domain-grouped 70/15/15 split to data/processed/splits.parquet.

Run from the repo root:  python src/data_prep.py
"""
from pathlib import Path

import pandas as pd
import tldextract
import yaml
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())

# Offline extractor: uses the bundled public-suffix snapshot, no network calls.
_EXT = tldextract.TLDExtract(suffix_list_urls=())


def registered_domain(url: str) -> str:
    e = _EXT(url)
    return (e.top_domain_under_public_suffix
            if hasattr(e, "top_domain_under_public_suffix") else e.registered_domain)


def main() -> None:
    seed = CFG["seed"]
    d = CFG["data"]

    # 1. Load only the two columns we need
    df = pd.read_csv(ROOT / CFG["paths"]["raw_csv"], usecols=[d["url_col"], d["label_col"]])
    df = df.rename(columns={d["url_col"]: "url", d["label_col"]: "label"})
    print(f"Loaded: {df.shape}")
    print("Raw label counts (before any flip):")
    print(df.label.value_counts().to_string())

    # 2. Label direction: PhiUSIIL has 1=legitimate, 0=phishing. We want 1=phish.
    if d["flip_label"]:
        df["label"] = 1 - df["label"]

    # 3. Clean
    n0 = len(df)
    df = df.dropna(subset=["url", "label"])
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df.url.str.len().between(1, d["max_url_len"])]
    df = df.drop_duplicates(subset="url")
    print(f"After dropna / length filter / dedupe: {len(df)} (dropped {n0 - len(df)})")

    # 4. Registered domain (used for the group split)
    df["domain"] = df.url.map(registered_domain)
    df = df[df.domain != ""].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["url_id"] = df.index
    print(f"After dropping unparseable domains: {len(df)}")

    # 5. Group split 70/15/15 by registered domain
    s = CFG["split"]
    gss = GroupShuffleSplit(n_splits=1, test_size=1 - s["train"], random_state=seed)
    tr_idx, rest_idx = next(gss.split(df, groups=df["domain"]))
    rest = df.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=s["test"] / (s["val"] + s["test"]),
                             random_state=seed)
    va_rel, te_rel = next(gss2.split(rest, groups=rest["domain"]))

    df["split"] = "train"
    df.loc[rest.index[va_rel], "split"] = "val"
    df.loc[rest.index[te_rel], "split"] = "test"

    # 6. Checks: no domain leaks across splits
    dom_sets = {k: set(g.domain) for k, g in df.groupby("split")}
    assert not (dom_sets["train"] & dom_sets["test"]), "domain leak train/test"
    assert not (dom_sets["train"] & dom_sets["val"]), "domain leak train/val"
    assert not (dom_sets["val"] & dom_sets["test"]), "domain leak val/test"

    # 7. Save splits + dataset statistics table
    out = ROOT / CFG["paths"]["splits"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["url_id", "url", "label", "domain", "split"]].to_parquet(out, index=False)

    stats = (df.groupby("split")["label"]
               .agg(n="size", n_phish="sum")
               .assign(n_benign=lambda x: x.n - x.n_phish,
                       phish_frac=lambda x: (x.n_phish / x.n).round(3),
                       n_domains=df.groupby("split")["domain"].nunique())
               .reindex(["train", "val", "test"]))
    res = ROOT / CFG["paths"]["results_dir"]
    res.mkdir(parents=True, exist_ok=True)
    stats.to_csv(res / "dataset_stats.csv")
    print("\nSplit statistics:")
    print(stats.to_string())

    # 8. Eyeball check of label direction (1 should be phishing)
    print("\nSample URLs - label 1 (should look like PHISHING):")
    print(df[df.label == 1].url.sample(5, random_state=0).to_string(index=False))
    print("\nSample URLs - label 0 (should look LEGITIMATE):")
    print(df[df.label == 0].url.sample(5, random_state=0).to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
