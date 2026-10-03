"""Person 2: figures + tables from results/robustness.csv (read-only on inputs).

Usage:  python src/make_robustness_figures.py

Outputs
  results/robustness_summary.csv      model,family,severity,metric,mean,std,n_seeds
  results/robustness_table.csv/.tex   compact paper table (max severity per family)
  figures/rob_evasion.(pdf|png)       path_strip: ASR and recall vs level
  figures/rob_false_alarm.(pdf|png)   FPR vs severity, targeted vs control
  figures/rob_heatmap.(pdf|png)       ASR and FPR heatmaps, model x (family,severity)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

MODELS = ["lr", "rf", "xgb", "mlp"]
COLORS = {"lr": "#1f77b4", "rf": "#2ca02c", "xgb": "#d62728", "mlp": "#9467bd"}
NICE = {"lr": "LogReg", "rf": "RandForest", "xgb": "XGBoost", "mlp": "MLP"}
FAMILY_NICE = {"path_strip": "Path strip", "path_padding": "Path padding",
               "subdomain_padding": "Subdomain padding", "char_edit": "Char edit"}
CONTROL = {"path_padding": "control_path", "subdomain_padding": "control_subdomain",
           "char_edit": "control_char"}
FALSE_ALARM_FAMILIES = ["path_padding", "subdomain_padding", "char_edit"]
SHORT = {"path_strip": "strip", "path_padding": "padpath",
         "subdomain_padding": "subdom", "char_edit": "chars"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load_summary():
    r = pd.read_csv(RES / "robustness.csv")
    g = r.groupby(["model", "family", "severity", "metric"])["value"]
    s = g.agg(mean="mean", std="std", n_seeds="count").reset_index()
    s["std"] = s["std"].fillna(0.0)
    s.to_csv(RES / "robustness_summary.csv", index=False)
    return s


def series(s, model, family, metric):
    """Mean/std vs severity for one family, with the clean point at x=0."""
    c = s[(s.model == model) & (s.family == "clean") & (s.metric == metric)]
    f = s[(s.model == model) & (s.family == family) & (s.metric == metric)]
    d = pd.concat([c, f]).sort_values("severity")
    return d["severity"].values, d["mean"].values, d["std"].values


def line(ax, s, model, family, metric, **kw):
    x, m, sd = series(s, model, family, metric)
    ax.plot(x, m, marker="o", color=COLORS[model], **kw)
    ax.fill_between(x, m - sd, m + sd, color=COLORS[model], alpha=0.15, lw=0)


def fig_evasion(s):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric, title in zip(axes, ["asr", "recall"],
                                 ["Attack success rate", "Phishing recall"]):
        for m in MODELS:
            line(ax, s, m, "path_strip", metric, label=NICE[m])
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["clean", "1", "2", "3"])
        ax.set_xlabel("Path-strip level")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ASR")
    axes[1].set_ylabel("Recall")
    axes[0].legend(frameon=False)
    fig.suptitle("Evasion: stripping path/query from phishing URLs", y=1.03)
    save(fig, "rob_evasion")


def fig_false_alarm(s):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, fam in zip(axes, FALSE_ALARM_FAMILIES):
        for m in MODELS:
            line(ax, s, m, fam, "fpr", ls="-")
            line(ax, s, m, CONTROL[fam], "fpr", ls="--", alpha=0.7)
        ax.set_title(FAMILY_NICE[fam])
        ax.set_xlabel("Severity")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("False-positive rate (benign flagged)")
    handles = [plt.Line2D([], [], color=COLORS[m], marker="o", label=NICE[m])
               for m in MODELS]
    handles += [plt.Line2D([], [], color="k", ls="-", label="targeted"),
                plt.Line2D([], [], color="k", ls="--", label="random control")]
    axes[-1].legend(handles=handles, frameon=False, fontsize=8)
    fig.suptitle("False-alarm induction: targeted vs length-matched random control",
                 y=1.03)
    save(fig, "rob_false_alarm")


def fig_heatmap(s):
    fams = ["path_strip"] + FALSE_ALARM_FAMILIES
    cols = []
    for fam in fams:
        for sev in sorted(s[s.family == fam].severity.unique()):
            cols.append((fam, sev))
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.6 * len(cols)), 6))
    for ax, metric, title in zip(axes, ["asr", "fpr"],
                                 ["Attack success rate (phish evades)",
                                  "False-positive rate (benign flagged)"]):
        M = np.full((len(MODELS), len(cols)), np.nan)
        for i, m in enumerate(MODELS):
            for j, (fam, sev) in enumerate(cols):
                v = s[(s.model == m) & (s.family == fam) &
                      (s.severity == sev) & (s.metric == metric)]["mean"]
                if len(v):
                    M[i, j] = v.iloc[0]
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis_r", aspect="auto")
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([NICE[m] for m in MODELS])
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{SHORT[f]} {sv:g}" for f, sv in cols],
                           rotation=60, ha="right", fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=6, color="black" if M[i, j] < 0.5 else "white")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    save(fig, "rob_heatmap")


def fmt(row):
    return f"{row['mean']:.3f} ± {row['std']:.3f}"


def paper_table(s):
    rows = []
    for m in MODELS:
        for fam in ["path_strip"] + FALSE_ALARM_FAMILIES:
            sev = s[s.family == fam].severity.max()
            r = {"model": NICE[m], "family": FAMILY_NICE[fam], "max_severity": sev}
            for metric in ["recall", "asr", "fpr"]:
                v = s[(s.model == m) & (s.family == fam) &
                      (s.severity == sev) & (s.metric == metric)]
                r[metric] = fmt(v.iloc[0]) if len(v) else "n/a"
            c = s[(s.model == m) & (s.family == "clean") & (s.metric == "recall")]
            r["clean_recall"] = fmt(c.iloc[0]) if len(c) else "n/a"
            rows.append(r)
    t = pd.DataFrame(rows)[["model", "family", "max_severity", "clean_recall",
                            "recall", "asr", "fpr"]]
    t.to_csv(RES / "robustness_table.csv", index=False)
    (RES / "robustness_table.tex").write_text(
        t.to_latex(index=False, escape=False,
                   column_format="llrcccc"))
    print("saved robustness_table.csv/.tex")


if __name__ == "__main__":
    summary = load_summary()
    fig_evasion(summary)
    fig_false_alarm(summary)
    fig_heatmap(summary)
    paper_table(summary)