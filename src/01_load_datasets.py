"""
01_load_datasets.py
-------------------
Downloads TruthfulQA and XCOPA from HuggingFace and saves
raw data to data/raw/ as JSON files.

Language coverage:
  TruthfulQA : en (original), es, it (alexandrainst/m_truthfulqa)
               sw is NOT available in m_truthfulqa (confirmed configs: ar,bn,ca,da,de,es,
               eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh).
  XCOPA      : en (SuperGLUE COPA), sw, it (xcopa)
               Italian (it) is the shared non-English language across both datasets,
               allowing task and language effects to be disentangled.

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
    # English: use original TruthfulQA (alexandrainst/m_truthfulqa has no 'en' config)
    dataset_en = load_dataset("truthful_qa", "multiple_choice")
    examples_en = [{"question": ex["question"], "mc1_targets": ex["mc1_targets"], "mc2_targets": ex["mc2_targets"]}
                   for ex in dataset_en["validation"]]
    save_json(examples_en, f"{RAW_DIR}/truthfulqa_en.json")

    # Spanish: use alexandrainst/m_truthfulqa
    dataset_es = load_dataset("alexandrainst/m_truthfulqa", "es")
    examples_es = [dict(ex) for ex in dataset_es["val"]]
    save_json(examples_es, f"{RAW_DIR}/truthfulqa_es.json")

    # Italian: use alexandrainst/m_truthfulqa — shared with XCOPA to break task/language confound
    dataset_it = load_dataset("alexandrainst/m_truthfulqa", "it")
    examples_it = [dict(ex) for ex in dataset_it["val"]]
    save_json(examples_it, f"{RAW_DIR}/truthfulqa_it.json")


def load_xcopa():
    print("\n=== Loading XCOPA ===")
    # English: use original COPA from SuperGLUE (XCOPA has no 'en' config)
    dataset_en = load_dataset("super_glue", "copa")
    # Combine train + validation (test labels are -1); keep label field for consistency
    examples_en = [
        {"premise": ex["premise"], "choice1": ex["choice1"], "choice2": ex["choice2"],
         "question": ex["question"], "label": ex["label"], "idx": ex["idx"]}
        for split in ["train", "validation"]
        for ex in dataset_en[split]
    ]
    save_json(examples_en, f"{RAW_DIR}/xcopa_en.json")

    # Swahili: use xcopa sw
    dataset_sw = load_dataset("xcopa", "sw")
    examples_sw = [dict(ex) for ex in dataset_sw["test"]]
    save_json(examples_sw, f"{RAW_DIR}/xcopa_sw.json")

    # Italian: use xcopa it — shared with TruthfulQA to break task/language confound
    dataset_it = load_dataset("xcopa", "it")
    examples_it = [dict(ex) for ex in dataset_it["test"]]
    save_json(examples_it, f"{RAW_DIR}/xcopa_it.json")


if __name__ == "__main__":
    load_truthfulqa()
    load_xcopa()
    print("\nAll raw datasets saved to data/raw/")
