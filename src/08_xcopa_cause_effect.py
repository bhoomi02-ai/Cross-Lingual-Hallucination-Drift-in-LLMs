"""
08_xcopa_cause_effect.py
------------------------
XCOPA cause-vs-effect hallucination breakdown.

Splits XCOPA labels by question_type (cause | effect) per language and reports
hallucination rate for each cell. Tests whether causal direction affects the
failure mode (grader request, post-midway).

Reads from:  outputs/labels/xcopa_*_labels.json  (must include question_type)
Writes to:   results/tables/xcopa_cause_effect.csv

Run: python src/08_xcopa_cause_effect.py
"""

import json
import os
import glob
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

LABEL_DIR = "outputs/labels"
TABLE_DIR = "results/tables"
os.makedirs(TABLE_DIR, exist_ok=True)


def load_xcopa_labels():
    rows = []
    for path in sorted(glob.glob(f"{LABEL_DIR}/xcopa_*_labels.json")):
        with open(path, encoding="utf-8") as f:
            rows.extend(json.load(f))
    df = pd.DataFrame(rows)
    if "question_type" not in df.columns:
        raise SystemExit(
            "Label files missing `question_type`. Re-run src/04_run_judge.py "
            "after the patch so XCOPA labels carry cause/effect metadata."
        )
    df["is_hallucinated"] = (df["label"] == "Hallucinated").astype(int)
    return df


def breakdown(df):
    rows = []
    for (lang, qtype), group in df.groupby(["language", "question_type"]):
        n = len(group)
        h = int(group["is_hallucinated"].sum())
        rows.append({
            "language": lang,
            "question_type": qtype,
            "total": n,
            "hallucinated": h,
            "HR": round(100 * h / n, 2) if n else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["language", "question_type"])


def per_lang_cause_vs_effect_test(df):
    """Within each non-English language, test cause vs effect HR (chi² + Fisher)."""
    rows = []
    for lang in sorted(df["language"].unique()):
        if lang == "en":
            continue
        sub = df[df["language"] == lang]
        if sub["question_type"].nunique() < 2:
            continue
        ct = pd.crosstab(sub["question_type"], sub["is_hallucinated"])
        chi2, p, dof, _ = chi2_contingency(ct)
        cause_hr = 100 * sub[sub["question_type"] == "cause"]["is_hallucinated"].mean()
        effect_hr = 100 * sub[sub["question_type"] == "effect"]["is_hallucinated"].mean()
        row = {
            "language": lang,
            "cause_HR": round(cause_hr, 2),
            "effect_HR": round(effect_hr, 2),
            "delta_cause_minus_effect": round(cause_hr - effect_hr, 2),
            "chi2": round(chi2, 2),
            "p_chi2": round(p, 6),
            "significant": "YES" if p < 0.05 else "NO",
        }
        if ct.shape == (2, 2):
            or_, p_f = fisher_exact(ct)
            row["fisher_OR"] = round(or_, 2)
            row["p_fisher"] = round(p_f, 6)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_xcopa_labels()
    print(f"Loaded {len(df)} XCOPA labels across languages: "
          f"{sorted(df['language'].unique())}")

    bd = breakdown(df)
    print("\n=== XCOPA HR by language × question_type ===")
    print(bd.to_string(index=False))

    tests = per_lang_cause_vs_effect_test(df)
    print("\n=== Per-language cause vs effect significance ===")
    if tests.empty:
        print("  (no non-English language with both question types)")
    else:
        print(tests.to_string(index=False))

    bd.to_csv(f"{TABLE_DIR}/xcopa_cause_effect.csv", index=False)
    if not tests.empty:
        tests.to_csv(f"{TABLE_DIR}/xcopa_cause_effect_tests.csv", index=False)
    print(f"\nSaved to {TABLE_DIR}/xcopa_cause_effect*.csv")
