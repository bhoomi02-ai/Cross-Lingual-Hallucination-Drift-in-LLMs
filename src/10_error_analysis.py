"""
10_error_analysis.py
--------------------
Categorizes GPT-4o-mini judge `reason` strings into hallucination types
(incoherent, wrong_answer, fabricated, code_switching, format_violation,
refusal, other) using simple keyword heuristics. Reports type distributions
per language × task cell.

The categorization is heuristic — Italian is the only language the team can
read directly, so manually spot-check Italian categorizations after running
and tighten the rules if needed.

Reads from:  outputs/labels/*.json
Writes to:   results/tables/error_categories.csv
             results/tables/error_categories_pivot.csv

Run: python src/10_error_analysis.py
"""

import json
import os
import glob
import re
import pandas as pd

LABEL_DIR = "outputs/labels"
TABLE_DIR = "results/tables"
os.makedirs(TABLE_DIR, exist_ok=True)


# ── Heuristic category rules ──────────────────────────────────────────────────
# Order matters: first match wins. Keep rules conservative; "other" catches
# everything that doesn't fit.

CATEGORY_RULES = [
    ("incoherent",
     [r"\bincoheren", r"\bnonsens", r"\bgibberis", r"\brambl",
      r"unrelated\s+to", r"does\s+not\s+address", r"off[- ]?topic",
      r"irrelevant", r"meaningless"]),
    ("code_switching",
     [r"code[- ]?switch", r"mixed?\s+languages?", r"in\s+english\s+instead",
      r"answers?\s+in\s+\w+\s+instead", r"different\s+language"]),
    ("refusal",
     [r"\brefus", r"declines?\s+to\s+answer", r"cannot\s+answer",
      r"unable\s+to\s+(answer|provide)"]),
    ("format_violation",
     [r"\bformat", r"chose\s+neither", r"did\s+not\s+(pick|choose|select)",
      r"neither\s+a\s+nor\s+b", r"both\s+a\s+and\s+b",
      r"missing\s+(option|choice)"]),
    ("fabricated",
     [r"fabricat", r"made[- ]?up", r"hallucinat", r"invented",
      r"non-?existent", r"false\s+claim", r"unsupported\s+claim"]),
    ("wrong_answer",
     [r"incorrect", r"\bwrong", r"factually\s+(incorrect|inaccurate|false)",
      r"\binaccurate", r"misleading", r"chose\s+the\s+wrong",
      r"selected\s+the\s+(wrong|incorrect|less\s+plausible)",
      r"opposite\s+of"]),
]


def categorize(reason: str) -> str:
    if not isinstance(reason, str) or not reason.strip():
        return "no_reason"
    text = reason.lower()
    for cat, patterns in CATEGORY_RULES:
        for pat in patterns:
            if re.search(pat, text):
                return cat
    return "other"


def load_all_labels():
    rows = []
    for path in sorted(glob.glob(f"{LABEL_DIR}/*.json")):
        # skip the dual_judge subdir
        if os.path.isdir(path):
            continue
        with open(path, encoding="utf-8") as f:
            rows.extend(json.load(f))
    df = pd.DataFrame(rows)
    return df


def main():
    df = load_all_labels()
    halluc = df[df["label"] == "Hallucinated"].copy()
    halluc["category"] = halluc["reason"].apply(categorize)

    print(f"Total labels: {len(df)}  (Hallucinated: {len(halluc)})")
    print(f"Languages: {sorted(df['language'].unique())}")
    print(f"Tasks: {sorted(df['task'].unique())}")

    # Per-cell counts
    counts = (halluc.groupby(["task", "language", "category"])
                    .size().reset_index(name="count"))
    print("\n=== Hallucination categories by task × language ===")
    print(counts.to_string(index=False))

    # Wide pivot for the report table: rows = task/lang, cols = category
    pivot = (halluc.groupby(["task", "language", "category"])
                   .size().unstack(fill_value=0))
    pivot["TOTAL"] = pivot.sum(axis=1)
    print("\n=== Pivot (counts) ===")
    print(pivot)

    counts.to_csv(f"{TABLE_DIR}/error_categories.csv", index=False)
    pivot.to_csv(f"{TABLE_DIR}/error_categories_pivot.csv")
    print(f"\nSaved to {TABLE_DIR}/error_categories.csv and _pivot.csv")

    # Quick spot-check sample for manual Italian validation
    it_samples = (halluc[halluc["language"] == "it"]
                  .groupby("category")
                  .head(3)[["task", "category", "response", "reason"]])
    if len(it_samples) > 0:
        path = f"{TABLE_DIR}/error_categories_italian_samples.csv"
        it_samples.to_csv(path, index=False)
        print(f"Saved {len(it_samples)} Italian spot-check samples to {path}")


if __name__ == "__main__":
    main()
