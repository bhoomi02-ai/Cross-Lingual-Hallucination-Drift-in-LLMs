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

    # Compute Φₗ per language where the same language appears in both tasks.
    # With the current dataset (es→TruthfulQA only, sw→XCOPA only) this will
    # be empty, so we also compute a cross-task aggregate Φ:
    #   Φ = ΔHR(es, TruthfulQA) − ΔHR(sw, XCOPA)
    # This captures whether factual drift (TruthfulQA) differs from commonsense
    # drift (XCOPA) when each task is evaluated in its available non-English language.
    phi_rows = []
    for lang in hr_df["language"].unique():
        if lang == "en":
            continue
        tqa_drift = hr_df[(hr_df["language"] == lang) & (hr_df["task"] == "truthfulqa")]["delta_HR"].values
        xcopa_drift = hr_df[(hr_df["language"] == lang) & (hr_df["task"] == "xcopa")]["delta_HR"].values
        if len(tqa_drift) > 0 and len(xcopa_drift) > 0:
            phi = round(float(tqa_drift[0]) - float(xcopa_drift[0]), 2)
            phi_rows.append({"language": lang, "comparison": f"{lang}/TruthfulQA vs {lang}/XCOPA", "phi_l": phi})

    phi_df = pd.DataFrame(phi_rows) if phi_rows else pd.DataFrame(columns=["language", "comparison", "phi_l"])

    # Cross-task aggregate Φ: compare drift across the two non-English languages
    # (one per task), as documented in the README.
    agg_phi_rows = []
    non_en = hr_df[hr_df["language"] != "en"]
    for (tqa_lang, xcopa_lang) in [
        (lang, olang)
        for lang in non_en[non_en["task"] == "truthfulqa"]["language"].unique()
        for olang in non_en[non_en["task"] == "xcopa"]["language"].unique()
        if lang != olang
    ]:
        tqa_drift = hr_df[(hr_df["language"] == tqa_lang) & (hr_df["task"] == "truthfulqa")]["delta_HR"].values
        xcopa_drift = hr_df[(hr_df["language"] == xcopa_lang) & (hr_df["task"] == "xcopa")]["delta_HR"].values
        if len(tqa_drift) > 0 and len(xcopa_drift) > 0:
            phi = round(float(tqa_drift[0]) - float(xcopa_drift[0]), 2)
            agg_phi_rows.append({
                "comparison": f"{tqa_lang}/TruthfulQA vs {xcopa_lang}/XCOPA",
                "delta_HR_tqa": float(tqa_drift[0]),
                "delta_HR_xcopa": float(xcopa_drift[0]),
                "phi_l": phi,
            })

    agg_phi_df = pd.DataFrame(agg_phi_rows)
    return hr_df, phi_df, agg_phi_df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_all_labels()
    hr_df = compute_hr(df)
    hr_df, phi_df, agg_phi_df = compute_drift(hr_df)

    print("\n=== Hallucination Rates ===")
    print(hr_df.to_string(index=False))

    print("\n=== Per-Language Drift Interaction Scores (Φₗ) ===")
    if phi_df.empty:
        print("  (none — no language appears in both tasks)")
    else:
        print(phi_df.to_string(index=False))

    print("\n=== Cross-Task Aggregate Φ (en→L drift: TruthfulQA vs XCOPA) ===")
    print(agg_phi_df.to_string(index=False))

    hr_df.to_csv(f"{TABLE_DIR}/hr_results.csv", index=False)
    phi_df.to_csv(f"{TABLE_DIR}/phi_scores.csv", index=False)
    agg_phi_df.to_csv(f"{TABLE_DIR}/phi_aggregate.csv", index=False)
    print(f"\nSaved to {TABLE_DIR}/")
