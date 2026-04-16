"""
06_statistical_tests.py
------------------------
Runs statistical significance tests on hallucination rates.
Tests whether drift differs across languages and task types.

Tests performed:
  1. Mann-Whitney U: EN vs non-EN hallucination rates per task
  2. Chi-squared: 2×2 contingency (task × hallucination outcome)
     to test whether drift is task-dependent (cross-task aggregate Φ)

Reads from:  outputs/labels/  and  results/tables/
Writes to:   results/tables/statistical_tests.csv

Run: python src/06_statistical_tests.py
"""

import json
import os
import glob
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact

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

    # ── 1. Per-task: EN vs non-EN (Mann-Whitney U) ──────────────────────────
    for task in tasks:
        en_group = df[(df["task"] == task) & (df["language"] == "en")]["is_hallucinated"]

        for lang in non_en_langs:
            lang_group = df[(df["task"] == task) & (df["language"] == lang)]["is_hallucinated"]

            if len(lang_group) == 0:
                continue

            stat, p = mannwhitneyu(en_group, lang_group, alternative="two-sided")

            results.append({
                "test": "Mann-Whitney U",
                "comparison": f"{task}: en vs {lang}",
                "en_HR": round(en_group.mean() * 100, 2),
                "other_HR": round(lang_group.mean() * 100, 2),
                "statistic": round(stat, 2),
                "p_value": round(p, 6),
                "significant": "YES" if p < 0.05 else "NO",
            })
            print(f"  {task} | EN vs {lang}: U={stat:.1f}, p={p:.4f} {'✓' if p < 0.05 else '✗'}")

    # ── 2. Cross-task Φ significance (chi-squared / Fisher's exact) ─────────
    # Since no language appears in both tasks, we test whether the *pattern*
    # of drift differs across tasks using the cross-task aggregate Φ approach
    # from 05_compute_metrics.py.
    #
    # Build a 2×2 contingency table from the non-English cells:
    #   rows = task (truthfulqa, xcopa)
    #   cols = outcome (hallucinated, faithful)
    # This tests: is the hallucination rate independent of task type for
    # non-English responses?

    print("\n=== Task-Dependency Test (Cross-Task Φ) ===")

    non_en = df[df["language"] != "en"]
    if len(non_en["task"].unique()) == 2:
        ct = pd.crosstab(non_en["task"], non_en["is_hallucinated"])
        chi2, p_chi2, dof, expected = chi2_contingency(ct)

        # Also compute drift values for reporting
        hr_by = non_en.groupby("task")["is_hallucinated"].mean() * 100
        en_hr_by = df[df["language"] == "en"].groupby("task")["is_hallucinated"].mean() * 100
        delta_tqa = hr_by.get("truthfulqa", 0) - en_hr_by.get("truthfulqa", 0)
        delta_xcopa = hr_by.get("xcopa", 0) - en_hr_by.get("xcopa", 0)
        phi_agg = round(delta_tqa - delta_xcopa, 2)

        print(f"  ΔHR(es, TruthfulQA) = {delta_tqa:+.2f} pp")
        print(f"  ΔHR(sw, XCOPA)      = {delta_xcopa:+.2f} pp")
        print(f"  Φ (aggregate)       = {phi_agg:+.2f} pp")
        print(f"  χ²={chi2:.2f}, df={dof}, p={p_chi2:.6f} {'✓' if p_chi2 < 0.05 else '✗'}")

        results.append({
            "test": "Chi-squared (task × outcome)",
            "comparison": f"es/TruthfulQA vs sw/XCOPA (Φ={phi_agg:+.2f}pp)",
            "en_HR": "",
            "other_HR": "",
            "statistic": round(chi2, 2),
            "p_value": round(p_chi2, 6),
            "significant": "YES" if p_chi2 < 0.05 else "NO",
        })

        # Fisher's exact as robustness check (better for small samples)
        if ct.shape == (2, 2):
            odds_ratio, p_fisher = fisher_exact(ct)
            print(f"  Fisher's exact: OR={odds_ratio:.2f}, p={p_fisher:.6f} {'✓' if p_fisher < 0.05 else '✗'}")
            results.append({
                "test": "Fisher exact (robustness)",
                "comparison": f"es/TruthfulQA vs sw/XCOPA",
                "en_HR": "",
                "other_HR": "",
                "statistic": round(odds_ratio, 2),
                "p_value": round(p_fisher, 6),
                "significant": "YES" if p_fisher < 0.05 else "NO",
            })

    # ── 3. Per-language Φₗ (only if a language has both tasks) ──────────────
    print("\n=== Per-Language Φₗ ===")
    found_any = False
    for lang in non_en_langs:
        tqa = df[(df["task"] == "truthfulqa") & (df["language"] == lang)]["is_hallucinated"]
        xcopa = df[(df["task"] == "xcopa") & (df["language"] == lang)]["is_hallucinated"]
        tqa_en = df[(df["task"] == "truthfulqa") & (df["language"] == "en")]["is_hallucinated"]
        xcopa_en = df[(df["task"] == "xcopa") & (df["language"] == "en")]["is_hallucinated"]

        if len(tqa) > 0 and len(xcopa) > 0:
            delta_tqa = tqa.mean() - tqa_en.mean()
            delta_xcopa = xcopa.mean() - xcopa_en.mean()
            phi = round((delta_tqa - delta_xcopa) * 100, 2)
            print(f"  Φ_{lang} = {phi:.2f} pp")
            found_any = True

    if not found_any:
        print("  (none — no language appears in both tasks; see cross-task Φ above)")

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = load_all_labels()
    print("=== Cross-Language Significance Tests ===")
    results_df = run_tests(df)
    results_df.to_csv(f"{TABLE_DIR}/statistical_tests.csv", index=False)
    print(f"\nSaved to {TABLE_DIR}/statistical_tests.csv")
