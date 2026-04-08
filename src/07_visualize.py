"""
07_visualize.py
---------------
Generates bar charts and drift plots from results.
Saves figures to results/figures/

Reads from:  results/tables/
Writes to:   results/figures/

Run: python src/07_visualize.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TABLE_DIR  = "results/tables"
FIGURE_DIR = "results/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

LANG_LABELS = {"en": "English", "es": "Spanish", "sw": "Swahili"}
TASK_LABELS = {"truthfulqa": "TruthfulQA\n(Factual QA)", "xcopa": "XCOPA\n(Commonsense)"}
COLORS = {"en": "#4C72B0", "es": "#DD8452", "sw": "#55A868"}


def plot_hr_by_language_task(hr_df):
    """Bar chart: HR per language, grouped by task."""
    hr_df["lang_label"] = hr_df["language"].map(LANG_LABELS)
    hr_df["task_label"] = hr_df["task"].map(TASK_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax, (task_key, task_label) in zip(axes, TASK_LABELS.items()):
        subset = hr_df[hr_df["task"] == task_key].sort_values("language")
        bars = ax.bar(
            subset["lang_label"],
            subset["HR"],
            color=[COLORS[l] for l in subset["language"]],
            edgecolor="white",
            linewidth=0.8,
        )
        # Add value labels on bars
        for bar, val in zip(bars, subset["HR"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )
        ax.set_title(task_label, fontsize=13, fontweight="bold")
        ax.set_ylabel("Hallucination Rate (%)" if ax == axes[0] else "")
        ax.set_ylim(0, max(hr_df["HR"]) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Hallucination Rate by Language and Task", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{FIGURE_DIR}/hr_by_language_task.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


def plot_drift(hr_df):
    """Bar chart: ΔHR per non-English language, grouped by task."""
    en_rates = hr_df[hr_df["language"] == "en"].set_index("task")["HR"]
    non_en = hr_df[hr_df["language"] != "en"].copy()
    non_en["delta_HR"] = non_en.apply(
        lambda r: r["HR"] - en_rates.get(r["task"], 0), axis=1
    )
    non_en["lang_label"] = non_en["language"].map(LANG_LABELS)
    non_en["task_label"] = non_en["task"].map(TASK_LABELS)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(non_en["lang_label"].unique()))
    tasks = list(TASK_LABELS.keys())
    width = 0.35

    for i, task in enumerate(tasks):
        subset = non_en[non_en["task"] == task].sort_values("language")
        positions = [j + i * width for j in range(len(subset))]
        bars = ax.bar(
            positions, subset["delta_HR"],
            width=width,
            label=TASK_LABELS[task],
            alpha=0.85,
        )
        for bar, val in zip(bars, subset["delta_HR"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:+.1f}",
                ha="center", va="bottom", fontsize=9,
            )

    langs = sorted(non_en["language"].unique())
    ax.set_xticks([j + width / 2 for j in range(len(langs))])
    ax.set_xticklabels([LANG_LABELS[l] for l in langs])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("ΔHR vs English (pp)")
    ax.set_title("Cross-Lingual Drift by Task Type", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = f"{FIGURE_DIR}/drift_by_task.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


if __name__ == "__main__":
    hr_df = pd.read_csv(f"{TABLE_DIR}/hr_results.csv")
    plot_hr_by_language_task(hr_df)
    plot_drift(hr_df)
    print("All figures saved to results/figures/")
