"""
01_load_datasets.py
-------------------
Downloads TruthfulQA and XCOPA from HuggingFace and saves
raw data to data/raw/ as JSON files.

Run: python src/01_load_datasets.py
"""

import json
import os
from datasets import load_dataset

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(data)} examples → {path}")


def load_truthfulqa():
    print("\n=== Loading TruthfulQA ===")
    for lang in ["en", "es"]:
        dataset = load_dataset("alexandrainst/m_truthfulqa", lang)
        # Use 'train' split (the only split available)
        examples = [dict(ex) for ex in dataset["train"]]
        save_json(examples, f"{RAW_DIR}/truthfulqa_{lang}.json")


def load_xcopa():
    print("\n=== Loading XCOPA ===")
    for lang in ["en", "es", "sw"]:
        dataset = load_dataset("xcopa", lang)
        # Use 'test' split (standard eval split for XCOPA)
        examples = [dict(ex) for ex in dataset["test"]]
        save_json(examples, f"{RAW_DIR}/xcopa_{lang}.json")


if __name__ == "__main__":
    load_truthfulqa()
    load_xcopa()
    print("\nAll raw datasets saved to data/raw/")
