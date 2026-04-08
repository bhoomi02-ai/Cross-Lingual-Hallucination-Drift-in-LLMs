"""
02_sample_and_format.py
-----------------------
Samples N questions per language per task and formats them
into prompts ready for Aya inference.

Reads from:  data/raw/
Writes to:   data/sampled/  and  data/prompts/

Run: python src/02_sample_and_format.py
"""

import json
import os
import random

random.seed(42)

N_SAMPLES = 150          # questions per language per task
RAW_DIR   = "data/raw"
SAMP_DIR  = "data/sampled"
PROM_DIR  = "data/prompts"

os.makedirs(SAMP_DIR, exist_ok=True)
os.makedirs(PROM_DIR, exist_ok=True)


# ── Prompt templates ─────────────────────────────────────────────────────────

TRUTHFULQA_TEMPLATE = {
    "en": "Answer the following question truthfully and concisely.\nQuestion: {question}\nAnswer:",
    "es": "Responde la siguiente pregunta de forma veraz y concisa.\nPregunta: {question}\nRespuesta:",
    "sw": "Jibu swali lifuatalo kwa uaminifu na kwa ufupi.\nSwali: {question}\nJibu:",
}

XCOPA_TEMPLATE = {
    "en": (
        "Choose the most plausible option.\n"
        "Premise: {premise}\n"
        "Question: What was the {question}?\n"
        "A: {choice1}\n"
        "B: {choice2}\n"
        "Answer (A or B):"
    ),
    "es": (
        "Elige la opción más plausible.\n"
        "Premisa: {premise}\n"
        "Pregunta: ¿Cuál fue el {question}?\n"
        "A: {choice1}\n"
        "B: {choice2}\n"
        "Respuesta (A o B):"
    ),
    "sw": (
        "Chagua chaguo la busara zaidi.\n"
        "Msingi: {premise}\n"
        "Swali: {question} ilikuwa nini?\n"
        "A: {choice1}\n"
        "B: {choice2}\n"
        "Jibu (A au B):"
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(data)} → {path}")

def sample(data, n):
    return random.sample(data, min(n, len(data)))


# ── TruthfulQA ────────────────────────────────────────────────────────────────

def process_truthfulqa(lang):
    raw = load_json(f"{RAW_DIR}/truthfulqa_{lang}.json")
    sampled = sample(raw, N_SAMPLES)
    save_json(sampled, f"{SAMP_DIR}/truthfulqa_{lang}_{N_SAMPLES}.json")

    prompts = []
    for i, ex in enumerate(sampled):
        question = ex.get("question", ex.get("Question", ""))
        prompts.append({
            "question_id": i,
            "language": lang,
            "task": "truthfulqa",
            "question": question,
            "prompt": TRUTHFULQA_TEMPLATE[lang].format(question=question),
            # Keep correct answers for later reference
            "correct_answers": ex.get("correct_answers", []),
            "incorrect_answers": ex.get("incorrect_answers", []),
        })

    save_json(prompts, f"{PROM_DIR}/truthfulqa_{lang}_prompts.json")


# ── XCOPA ─────────────────────────────────────────────────────────────────────

def process_xcopa(lang):
    raw = load_json(f"{RAW_DIR}/xcopa_{lang}.json")
    sampled = sample(raw, N_SAMPLES)
    save_json(sampled, f"{SAMP_DIR}/xcopa_{lang}_{N_SAMPLES}.json")

    prompts = []
    for i, ex in enumerate(sampled):
        prompts.append({
            "question_id": i,
            "language": lang,
            "task": "xcopa",
            "premise": ex["premise"],
            "question": ex["question"],   # "cause" or "effect"
            "choice1": ex["choice1"],
            "choice2": ex["choice2"],
            "correct_label": ex["label"], # 0=choice1, 1=choice2
            "prompt": XCOPA_TEMPLATE[lang].format(
                premise=ex["premise"],
                question=ex["question"],
                choice1=ex["choice1"],
                choice2=ex["choice2"],
            ),
        })

    save_json(prompts, f"{PROM_DIR}/xcopa_{lang}_prompts.json")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== TruthfulQA ===")
    for lang in ["en", "es"]:
        process_truthfulqa(lang)

    print("\n=== XCOPA ===")
    for lang in ["en", "es", "sw"]:
        process_xcopa(lang)

    print("\nAll prompts saved to data/prompts/")
