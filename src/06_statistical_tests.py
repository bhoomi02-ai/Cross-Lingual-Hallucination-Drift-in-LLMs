"""
06_statistical_tests.py
------------------------
Runs statistical significance tests on hallucination rates.
Tests whether drift differs across languages and task types.

Reads from:  outputs/labels/  and  results/tables/
Writes to:   results/tables/statistical_tests.csv

Run: python src/06_statistical_tests.py
"""

import json
import os
import glob
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency

LABEL_DIR = "outputs/labels"
TABLE_DIR = "results/tables"
os.makedirs(TABLE_DIR, exist_ok=True)


def load_all_labels():
    rows = []
    for path in glob.glob(f"{LABEL_DIR}/*.json"):
        with open(path, encoding="utf-8") as f:
            rows.extend(json.load(f))
    df = pd.DataFrame(rows)
    df["is_hallucinated"] = (df["label"] == "Hallucinated").astype(int)
    return df


def run_tests(df):
    results = []

    tasks = df["task"].unique()
    non_en_langs = [l for l in df["language"].unique() if l != "en"]

    for task in tasks:
        en_group = df[(df["task"] == task) & (df["language"] == "en")]["is_hallucinated"]

        for lang in non_en_langs:
            lang_group = df[(df["task"] == task) & (df["language"] == lang)]["is_hallucinated"]

            if len(lang_group) == 0:
                continue

            # Mann-Whitney U test
            stat, p = mannwhitneyu(en_group, lang_group, alternative="two-sided")

            results.append({
                "task": task,
                "lang_vs_english": lang,
                "en_HR": round(en_group.mean() * 100, 2),
                f"{lang}_HR": round(lang_group.mean() * 100, 2),
                "mann_whitney_stat": round(stat, 2),
                "p_value": round(p, 4),
                "significant": "YES" if p < 0.05 else "NO",
            })
            print(f"  {task} | EN vs {lang}: p={p:.4f} {'✓' if p < 0.05 else '✗'}")

    # Test: does drift differ between tasks? (Φₗ test)
    print("\n=== Task-Dependency Test (Φₗ) ===")
    for lang in non_en_langs:
        tqa = df[(df["task"] == "truthfulqa") & (df["language"] == lang)]["is_hallucinated"]
        xcopa = df[(df["task"] == "xcopa") & (df["language"] == lang)]["is_hallucinated"]
        tqa_en = df[(df["task"] == "truthfulqa") & (df["language"] == "en")]["is_hallucinated"]
        xcopa_en = df[(df["task"] == "xcopa") & (df["language"] == "en")]["is_hallucinated"]

        if len(tqa) > 0 and len(xcopa) > 0:
            delta_tqa = tqa.mean() - tqa_en.mean()
            delta_xcopa = xcopa.mean() - xcopa_en.mean()
            phi = round((delta_tqa - delta_xcopa) * 100, 2)
            print(f"  Φ_{lang} = {phi:.2f} pp  (TruthfulQA drift − XCOPA drift)")

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = load_all_labels()
    print("=== Cross-Language Significance Tests ===")
    results_df = run_tests(df)
    results_df.to_csv(f"{TABLE_DIR}/statistical_tests.csv", index=False)
    print(f"\nSaved to {TABLE_DIR}/statistical_tests.csv")
