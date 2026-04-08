"""
05_compute_metrics.py
---------------------
Computes HR(l,t), ΔHR(l,t), and Φₗ from judge labels.
Saves results table to results/tables/hr_results.csv

Reads from:  outputs/labels/
Writes to:   results/tables/

Run: python src/05_compute_metrics.py
"""

import json
import os
import glob
import pandas as pd

LABEL_DIR  = "outputs/labels"
TABLE_DIR  = "results/tables"
os.makedirs(TABLE_DIR, exist_ok=True)


# ── Load all label files into one DataFrame ───────────────────────────────────

def load_all_labels():
    rows = []
    for path in glob.glob(f"{LABEL_DIR}/*.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows.extend(data)
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} total labelled examples")
    print(df.groupby(["task", "language"])["label"].value_counts())
    return df


# ── Compute hallucination rate ────────────────────────────────────────────────
# Using simplified version: HR = (# Hallucinated) / (total) * 100
# The full formula needs P/R from human spot-check; use raw rate for now.

def compute_hr(df):
    results = []
    for (task, lang), group in df.groupby(["task", "language"]):
        total = len(group)
        hallucinated = (group["label"] == "Hallucinated").sum()
        hr = (hallucinated / total) * 100 if total > 0 else 0
        avg_tokens = group["token_count"].mean()
        results.append({
            "task": task,
            "language": lang,
            "total": total,
            "hallucinated": hallucinated,
            "HR": round(hr, 2),
            "avg_tokens": round(avg_tokens, 1),
        })
    return pd.DataFrame(results)


# ── Compute ΔHR and Φₗ ───────────────────────────────────────────────────────

def compute_drift(hr_df):
    # Get English baseline per task
    en_rates = hr_df[hr_df["language"] == "en"].set_index("task")["HR"]

    hr_df["delta_HR"] = hr_df.apply(
        lambda row: round(row["HR"] - en_rates.get(row["task"], 0), 2),
        axis=1
    )

    # Compute Φₗ = ΔHR(l, TruthfulQA) - ΔHR(l, XCOPA) per language
    phi_rows = []
    for lang in hr_df["language"].unique():
        if lang == "en":
            continue
        tqa_drift = hr_df[(hr_df["language"] == lang) & (hr_df["task"] == "truthfulqa")]["delta_HR"].values
        xcopa_drift = hr_df[(hr_df["language"] == lang) & (hr_df["task"] == "xcopa")]["delta_HR"].values
        if len(tqa_drift) > 0 and len(xcopa_drift) > 0:
            phi = round(float(tqa_drift[0]) - float(xcopa_drift[0]), 2)
            phi_rows.append({"language": lang, "phi_l": phi})

    phi_df = pd.DataFrame(phi_rows)
    return hr_df, phi_df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_all_labels()
    hr_df = compute_hr(df)
    hr_df, phi_df = compute_drift(hr_df)

    print("\n=== Hallucination Rates ===")
    print(hr_df.to_string(index=False))

    print("\n=== Drift Interaction Scores (Φₗ) ===")
    print(phi_df.to_string(index=False))

    hr_df.to_csv(f"{TABLE_DIR}/hr_results.csv", index=False)
    phi_df.to_csv(f"{TABLE_DIR}/phi_scores.csv", index=False)
    print(f"\nSaved to {TABLE_DIR}/")
