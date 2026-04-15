"""
05_retry_errors.py
------------------
Finds all ERROR labels across label files and retries them.
Run: python3 src/05_retry_errors.py
"""

import json
import os
import time
from openai import OpenAI
from tqdm import tqdm

LABEL_DIR = "outputs/labels"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
JUDGE_MODEL = "gpt-4o-mini"

JUDGE_PROMPT = """You are evaluating whether a language model's response is factually accurate.

Question: {question}
Model Response: {response}

Is this response factually accurate and faithful to the question?
Answer with exactly one word: Hallucinated or Faithful.
Then on a new line, give a one-sentence justification.

Your response MUST follow this exact format with no deviation:
LABEL: Hallucinated
REASON: <one sentence>

or

LABEL: Faithful
REASON: <one sentence>"""


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
        print(f"  Raw response: {raw}")
        return parse_label(raw)
    except Exception as e:
        print(f"  API error: {e}")
        return {"label": "ERROR", "reason": str(e)}


def parse_label(raw):
    label = "ERROR"
    reason = ""
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("LABEL:"):
            text = line.replace("LABEL:", "").strip()
            if "Hallucinated" in text:
                label = "Hallucinated"
            elif "Faithful" in text:
                label = "Faithful"
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()
    # fallback: check full raw text if parsing still failed
    if label == "ERROR":
        if "Hallucinated" in raw:
            label = "Hallucinated"
        elif "Faithful" in raw:
            label = "Faithful"
    return {"label": label, "reason": reason}


def retry_file(filename):
    path = os.path.join(LABEL_DIR, filename)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    errors = [i for i, x in enumerate(data) if x["label"] == "ERROR"]
    if not errors:
        print(f"{filename}: no errors, skipping")
        return

    print(f"{filename}: retrying {len(errors)} errors")
    for i in tqdm(errors):
        entry = data[i]
        result = call_judge(entry["question"], entry["response"])
        data[i]["label"] = result["label"]
        data[i]["reason"] = result["reason"]
        time.sleep(0.5)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved updated {filename}")


if __name__ == "__main__":
    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith("_labels.json")]
    for fname in sorted(label_files):
        retry_file(fname)
    print("Done. Re-check for errors with:")
    print('  grep -r \'"label": "ERROR"\' outputs/labels/')