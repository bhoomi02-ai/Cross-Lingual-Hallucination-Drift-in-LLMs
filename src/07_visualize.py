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
    """Bar chart: ΔHR for each non-English language–task pair."""
    en_rates = hr_df[hr_df["language"] == "en"].set_index("task")["HR"]
    non_en = hr_df[hr_df["language"] != "en"].copy()
    non_en["delta_HR"] = non_en.apply(
        lambda r: r["HR"] - en_rates.get(r["task"], 0), axis=1
    )
    non_en["bar_label"] = non_en.apply(
        lambda r: f"{LANG_LABELS[r['language']]}\n{TASK_LABELS[r['task']]}", axis=1
    )

    # Each language–task pair gets its own bar
    non_en = non_en.sort_values(["task", "language"])
    bar_colors = [COLORS[lang] for lang in non_en["language"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        range(len(non_en)),
        non_en["delta_HR"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
    )

    for bar, val in zip(bars, non_en["delta_HR"]):
        y_pos = bar.get_height() + (0.5 if val >= 0 else -2.5)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{val:+.1f} pp",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xticks(range(len(non_en)))
    ax.set_xticklabels(non_en["bar_label"], fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("ΔHR vs English (pp)")
    ax.set_title("Cross-Lingual Hallucination Drift by Task Type",
                 fontsize=13, fontweight="bold")
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
