"""features.py - Step 4 (Person 1). FROZEN after Saturday morning.

extract(url) -> dict of lexical features computed from the raw URL string only.
Person 2 must use this same function (plus apply_tld_map) on perturbed URLs.

Run from the repo root to build the feature files:  python src/features.py
"""
import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import tldextract
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())

_EXT = tldextract.TLDExtract(suffix_list_urls=())   # offline
_IPV4 = re.compile(r"\d{1,3}(\.\d{1,3}){3}")
KEYWORDS = ["login", "secure", "verify", "account", "update", "bank",
            "signin", "confirm", "password", "webscr", "support"]

NUMERIC_FEATURES = [
    "url_len", "host_len", "path_len", "query_len",
    "n_dots", "n_hyphens", "n_digits", "n_at", "n_qmark", "n_eq", "n_pct",
    "n_slash", "n_amp", "n_underscore", "n_hash",
    "host_dots", "host_hyphens", "host_digits",
    "n_subdomains", "path_depth", "has_ip", "has_port", "is_https",
    "digit_ratio", "letter_ratio", "host_entropy", "n_keywords",
]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(v / n * math.log2(v / n) for v in Counter(s).values())


def extract(url: str) -> dict:
    """Lexical features from one URL string. Never raises."""
    url = "" if url is None else str(url)
    try:
        p = urlparse(url if "://" in url else "http://" + url)
        host = (p.hostname or "").lower()
        path, query = p.path or "", p.query or ""
        has_port = int(p.port is not None)
    except Exception:
        host, path, query, has_port = "", "", "", 0

    try:
        e = _EXT(host)
        sub = [s for s in e.subdomain.split(".") if s]
        tld = e.suffix
    except Exception:
        sub, tld = [], ""

    low = url.lower()
    n = max(len(url), 1)
    n_digits = sum(c.isdigit() for c in url)
    return {
        "url_len": len(url), "host_len": len(host),
        "path_len": len(path), "query_len": len(query),
        "n_dots": url.count("."), "n_hyphens": url.count("-"),
        "n_digits": n_digits, "n_at": url.count("@"),
        "n_qmark": url.count("?"), "n_eq": url.count("="),
        "n_pct": url.count("%"), "n_slash": url.count("/"),
        "n_amp": url.count("&"), "n_underscore": url.count("_"),
        "n_hash": url.count("#"),
        "host_dots": host.count("."), "host_hyphens": host.count("-"),
        "host_digits": sum(c.isdigit() for c in host),
        "n_subdomains": len(sub),
        "path_depth": len([s for s in path.split("/") if s]),
        "has_ip": int(bool(_IPV4.fullmatch(host))),
        "has_port": has_port,
        "is_https": int(low.startswith("https")),
        "digit_ratio": n_digits / n,
        "letter_ratio": sum(c.isalpha() for c in url) / n,
        "host_entropy": _entropy(host),
        "n_keywords": sum(k in low for k in KEYWORDS),
        "tld": tld,
    }


def fit_tld_map(tlds: pd.Series) -> dict:
    """TLD -> training frequency. Fit on TRAIN only."""
    return tlds.value_counts(normalize=True).to_dict()


def apply_tld_map(feats: pd.DataFrame, tld_map: dict) -> pd.DataFrame:
    """Replace raw 'tld' string by 'tld_freq' (0.0 for unseen TLDs)."""
    out = feats.copy()
    out["tld_freq"] = out["tld"].map(tld_map).fillna(0.0)
    return out.drop(columns=["tld"])


def featurize(urls) -> pd.DataFrame:
    """Raw feature frame (with 'tld' string) for an iterable of URLs."""
    return pd.DataFrame([extract(u) for u in urls])


def main() -> None:
    splits = pd.read_parquet(ROOT / CFG["paths"]["splits"])
    out_dir = ROOT / CFG["paths"]["features_dir"]

    raw = {}
    for name in ["train", "val", "test"]:
        part = splits[splits.split == name].reset_index(drop=True)
        f = featurize(part.url)
        f.insert(0, "label", part.label.values)
        f.insert(0, "url_id", part.url_id.values)
        raw[name] = f
        print(f"{name}: {f.shape}")

    tld_map = fit_tld_map(raw["train"]["tld"])
    (out_dir / "tld_map.json").write_text(json.dumps(tld_map))

    for name, f in raw.items():
        final = apply_tld_map(f, tld_map)
        assert final.drop(columns=["url_id", "label"]).isna().sum().sum() == 0, "NaNs found"
        final.to_parquet(out_dir / f"features_{name}.parquet", index=False)

    tr = apply_tld_map(raw["train"], tld_map).drop(columns=["url_id"])
    const = [c for c in tr.columns if c != "label" and tr[c].nunique() <= 1]
    print("Constant columns (consider dropping):", const or "none")
    print("\nCorrelation with label (train), top 10 by |r|:")
    print(tr.corr()["label"].drop("label").abs().sort_values(ascending=False)
            .head(10).round(3).to_string())
    print(f"\nSaved features_*.parquet and tld_map.json in {out_dir}")


if __name__ == "__main__":
    main()
