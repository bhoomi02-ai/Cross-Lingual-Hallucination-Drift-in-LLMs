"""
04_run_judge.py
---------------
Sends every (question, response) pair to GPT-4o-mini as judge.
Saves binary labels (Hallucinated / Faithful) to outputs/labels/.

Reads from:  outputs/responses/
Writes to:   outputs/labels/

Run: python src/04_run_judge.py
     (needs OPENAI_API_KEY set as environment variable)
"""

import json
import os
import time
from openai import OpenAI
from tqdm import tqdm

RESPONSE_DIR = "outputs/responses"
LABEL_DIR    = "outputs/labels"
JUDGE_MODEL  = "gpt-4o-mini"

os.makedirs(LABEL_DIR, exist_ok=True)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── Judge prompt ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating whether a language model's response is factually accurate.

Question: {question}
Model Response: {response}

Is this response factually accurate and faithful to the question?
Answer with exactly one word: Hallucinated or Faithful.
Then on a new line, give a one-sentence justification.

Format:
LABEL: <Hallucinated or Faithful>
REASON: <one sentence>"""


# ── Call judge ────────────────────────────────────────────────────────────────

def call_judge(question, response):
    prompt = JUDGE_PROMPT.format(question=question, response=response)
    try:
        completion = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        raw = completion.choices[0].message.content.strip()
        return parse_label(raw)
    except Exception as e:
        print(f"  Judge API error: {e}")
        return {"label": "ERROR", "reason": str(e), "raw": ""}


def parse_label(raw):
    label = "ERROR"
    reason = ""
    for line in raw.split("\n"):
        if line.startswith("LABEL:"):
            text = line.replace("LABEL:", "").strip()
            if "Hallucinated" in text:
                label = "Hallucinated"
            elif "Faithful" in text:
                label = "Faithful"
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()
    return {"label": label, "reason": reason, "raw": raw}


# ── Run judge on one file ─────────────────────────────────────────────────────

def run_judge_file(filename):
    path = os.path.join(RESPONSE_DIR, filename)
    with open(path, encoding="utf-8") as f:
        responses = json.load(f)

    results = []
    for ex in tqdm(responses, desc=filename):
        # Get the question text — field name differs by task
        question = ex.get("question", ex.get("premise", ""))
        judge_result = call_judge(question, ex["response"])

        results.append({
            "question_id": ex["question_id"],
            "language": ex["language"],
            "task": ex["task"],
            "question": question,
            "response": ex["response"],
            "token_count": ex["token_count"],
            "label": judge_result["label"],
            "reason": judge_result["reason"],
        })
        time.sleep(0.3)  # avoid rate limits

    out_name = filename.replace("_responses.json", "_labels.json")
    out_path = os.path.join(LABEL_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  → Saved {len(results)} labels to {out_path}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    response_files = sorted([
        f for f in os.listdir(RESPONSE_DIR) if f.endswith("_responses.json")
    ])

    for fname in response_files:
        run_judge_file(fname)

    print("All judge labels saved to outputs/labels/")
