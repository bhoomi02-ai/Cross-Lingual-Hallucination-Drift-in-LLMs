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
    "it": "Rispondi alla seguente domanda in modo veritiero e conciso.\nDomanda: {question}\nRisposta:",
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
    "it": (
        "Scegli l'opzione più plausibile.\n"
        "Premessa: {premise}\n"
        "Domanda: Quale è stato il {question}?\n"
        "A: {choice1}\n"
        "B: {choice2}\n"
        "Risposta (A o B):"
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

# Translate "cause"/"effect" labels for non-English XCOPA prompts
QUESTION_TRANSLATION = {
    "it": {"cause": "causa",  "effect": "effetto"},
    "sw": {"cause": "sababu", "effect": "athari"},
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
    translations = QUESTION_TRANSLATION.get(lang, {})
    for i, ex in enumerate(sampled):
        q_label = ex["question"]
        q_translated = translations.get(q_label, q_label)
        prompts.append({
            "question_id": i,
            "language": lang,
            "task": "xcopa",
            "premise": ex["premise"],
            "question": q_label,          # "cause" or "effect" (always English for downstream use)
            "choice1": ex["choice1"],
            "choice2": ex["choice2"],
            "correct_label": ex["label"], # 0=choice1, 1=choice2
            "prompt": XCOPA_TEMPLATE[lang].format(
                premise=ex["premise"],
                question=q_translated,
                choice1=ex["choice1"],
                choice2=ex["choice2"],
            ),
        })

    save_json(prompts, f"{PROM_DIR}/xcopa_{lang}_prompts.json")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Original languages — seed 42, order preserved so existing samples are unchanged
    print("\n=== TruthfulQA ===")
    for lang in ["en", "es"]:
        process_truthfulqa(lang)

    print("\n=== XCOPA ===")
    for lang in ["en", "sw"]:
        process_xcopa(lang)

    # Italian added separately with its own seed to avoid shifting existing samples
    # (addresses grader feedback: need a non-English language shared across both tasks)
    print("\n=== Italian (shared language for confound control) ===")
    random.seed(44)
    process_truthfulqa("it")
    random.seed(44)
    process_xcopa("it")

    print("\nAll prompts saved to data/prompts/")
