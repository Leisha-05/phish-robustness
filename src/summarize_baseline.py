"""summarize_baseline.py - Step H (Person 1).

Reads results/baseline_metrics.csv (one row per model x seed x metric) and produces:

  results/baseline_summary.csv   model, metric, mean, std, min, max, n_seeds   (long format)
  results/baseline_table.md      mean +/- std table for the paper / README (Result #2)
  results/baseline_table.tex     same table as a LaTeX tabular (best value per column in bold)
  results/h1_check.csv           one row per H1 criterion with the numbers behind it
  (plus a printed verdict)

std is the sample standard deviation (ddof=1) over the training seeds. It captures training randomness only,
NOT test-set sampling error. Logistic Regression is deterministic, so its std is 0 by construction.

H1 (operational definition, fixed before looking at the numbers):
  (a) each tree ensemble (rf, xgb) has higher mean F1 than LR
  (b) each tree ensemble has higher mean ROC-AUC than LR
  (c) every model's ROC-AUC exceeds 0.9 in every seed
H1 is "supported" if (a)-(c) all hold, "partly supported" if (c) holds but (a) or (b) fails for some tree model,
and "rejected" otherwise.

Run from the repo root:  python src/summarize_baseline.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

MODEL_ORDER = ["lr", "rf", "xgb", "mlp"]
MODEL_NAMES = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost", "mlp": "MLP"}
METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
METRIC_NAMES = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}
TREES = ["rf", "xgb"]
AUC_FLOOR = 0.9


def load() -> pd.DataFrame:
    df = pd.read_csv(RES / "baseline_metrics.csv")
    df = df[(df["family"] == "clean") & (df["severity"] == 0)]
    n = df.groupby(["model", "metric"])["seed"].nunique()
    if n.min() != n.max():
        raise SystemExit(f"Unequal number of seeds across model/metric:\n{n}")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "metric"])["value"]
    s = g.agg(mean="mean", std="std", min="min", max="max", n_seeds="count").reset_index()
    s["std"] = s["std"].fillna(0.0)
    s["model"] = pd.Categorical(s["model"], MODEL_ORDER, ordered=True)
    s["metric"] = pd.Categorical(s["metric"], METRIC_ORDER, ordered=True)
    return s.sort_values(["model", "metric"]).reset_index(drop=True)


def fmt(m, sd, digits=4):
    return f"{m:.{digits}f} ± {sd:.{digits}f}"


def write_tables(s: pd.DataFrame):
    models = [m for m in MODEL_ORDER if m in set(s["model"])]
    metrics = [m for m in METRIC_ORDER if m in set(s["metric"])]
    cell = {(r.model, r.metric): r for r in s.itertuples()}
    best = {m: max(models, key=lambda k: cell[(k, m)].mean) for m in metrics}
    n_seeds = int(s["n_seeds"].max())

    # Markdown
    lines = ["| Model | " + " | ".join(METRIC_NAMES[m] for m in metrics) + " |",
             "|---|" + "---|" * len(metrics)]
    for k in models:
        cells = []
        for m in metrics:
            t = fmt(cell[(k, m)].mean, cell[(k, m)].std)
            cells.append(f"**{t}**" if best[m] == k else t)
        lines.append(f"| {MODEL_NAMES[k]} | " + " | ".join(cells) + " |")
    lines += ["", f"Clean test set, mean ± sample std (ddof=1) over {n_seeds} training seeds, threshold 0.5. "
              "Bold = best mean per column. Logistic Regression is deterministic (std = 0)."]
    (RES / "baseline_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # LaTeX
    tex = ["\\begin{tabular}{l" + "c" * len(metrics) + "}", "\\hline",
           "Model & " + " & ".join(METRIC_NAMES[m] for m in metrics) + " \\\\", "\\hline"]
    for k in models:
        cells = []
        for m in metrics:
            t = f"{cell[(k, m)].mean:.4f} $\\pm$ {cell[(k, m)].std:.4f}"
            cells.append(f"\\textbf{{{t}}}" if best[m] == k else t)
        tex.append(f"{MODEL_NAMES[k]} & " + " & ".join(cells) + " \\\\")
    tex += ["\\hline", "\\end{tabular}"]
    (RES / "baseline_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def check_h1(df: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    mean = {(r.model, r.metric): r.mean for r in s.itertuples()}
    rows = []
    ok_ab = True
    for t in TREES:
        if t not in set(s["model"]):
            continue
        for metric in ["f1", "roc_auc"]:
            diff = mean[(t, metric)] - mean[("lr", metric)]
            # stricter view: does the WORST tree seed still beat LR?
            worst = df[(df.model == t) & (df.metric == metric)]["value"].min() - mean[("lr", metric)]
            passed = diff > 0
            ok_ab &= passed
            rows.append({"criterion": f"{t} > lr on {metric}", "value": diff,
                         "worst_seed_margin": worst, "passed": passed})
    # (c) AUC floor, every model, every seed
    auc = df[df.metric == "roc_auc"].groupby("model")["value"].min()
    ok_c = True
    for m in MODEL_ORDER:
        if m in auc.index:
            passed = bool(auc[m] > AUC_FLOOR)
            ok_c &= passed
            rows.append({"criterion": f"{m} min-seed roc_auc > {AUC_FLOOR}", "value": auc[m],
                         "worst_seed_margin": auc[m] - AUC_FLOOR, "passed": passed})
    verdict = "supported" if (ok_ab and ok_c) else ("partly supported" if ok_c else "rejected")
    out = pd.DataFrame(rows)
    out["h1_verdict"] = verdict
    return out


def main():
    df = load()
    s = summarize(df)
    s.to_csv(RES / "baseline_summary.csv", index=False)
    write_tables(s)
    h1 = check_h1(df, s)
    h1.to_csv(RES / "h1_check.csv", index=False)

    print((RES / "baseline_table.md").read_text())
    print(h1.drop(columns="h1_verdict").to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nH1 verdict: {h1['h1_verdict'].iloc[0].upper()}")
    print("Wrote results/baseline_summary.csv, baseline_table.md, baseline_table.tex, h1_check.csv")


if __name__ == "__main__":
    main()
