"""
03_run_inference.py
-------------------
Runs Aya Expanse 8B on all formatted prompts.
Saves responses with token counts to outputs/responses/.

Reads from:  data/prompts/
Writes to:   outputs/responses/

NOTE: Run this on Google Colab with a GPU (T4 free tier works).
      Runtime → Change runtime type → T4 GPU

Run: python src/03_run_inference.py
"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

PROMPT_DIR   = "data/prompts"
RESPONSE_DIR = "outputs/responses"
MODEL_ID     = "CohereLabs/aya-expanse-8b"
MAX_NEW_TOKENS = 200

os.makedirs(RESPONSE_DIR, exist_ok=True)


# ── Load model (once) ─────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_ID} with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.\n")
    return tokenizer, model


# ── Generate response ─────────────────────────────────────────────────────────

def generate(prompt, tokenizer, model):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,       # greedy — reproducible
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Only decode newly generated tokens
    new_tokens = outputs[0][inputs.shape[-1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    token_count = len(new_tokens)
    return response_text, token_count


# ── Run inference on one file ─────────────────────────────────────────────────

def run_file(filename, tokenizer, model):
    path = os.path.join(PROMPT_DIR, filename)
    with open(path, encoding="utf-8") as f:
        prompts = json.load(f)

    results = []
    for ex in tqdm(prompts, desc=filename):
        response, token_count = generate(ex["prompt"], tokenizer, model)
        results.append({
            **ex,
            "response": response,
            "token_count": token_count,
        })

    out_name = filename.replace("_prompts.json", "_responses.json")
    out_path = os.path.join(RESPONSE_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  → Saved {len(results)} responses to {out_path}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tokenizer, model = load_model()

    prompt_files = sorted([
        f for f in os.listdir(PROMPT_DIR) if f.endswith("_prompts.json")
    ])

    for fname in prompt_files:
        run_file(fname, tokenizer, model)

    print("All inference complete. Responses saved to outputs/responses/")
