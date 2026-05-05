"""
03_run_inference.py
-------------------
Runs Aya Expanse 8B on all formatted prompts.
Saves responses with token counts to outputs/responses/.
Logs GPU info and wall-clock runtime to results/gpu_runtime.txt.

Reads from:  data/prompts/
Writes to:   outputs/responses/
             results/gpu_runtime.txt

NOTE: Run this on Google Colab with a GPU (T4 free tier works).
      Runtime → Change runtime type → T4 GPU

Model config:
  Model:          CohereLabs/aya-expanse-8b
  Quantization:   4-bit NF4 (BitsAndBytes, double quant, compute dtype=float16)
  Decoding:       greedy (do_sample=False)
  Max new tokens: 200
  Hardware:       Google Colab T4 GPU (16 GB VRAM)

Run: python src/03_run_inference.py
"""

import json
import os
import time
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

PROMPT_DIR   = "data/prompts"
RESPONSE_DIR = "outputs/responses"
RESULTS_DIR  = "results"
MODEL_ID     = "CohereLabs/aya-expanse-8b"
MAX_NEW_TOKENS = 200

os.makedirs(RESPONSE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── GPU / runtime logging ─────────────────────────────────────────────────────

def get_gpu_info():
    if not torch.cuda.is_available():
        return "No GPU detected"
    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{name} ({vram_gb:.1f} GB VRAM)"


def write_runtime_log(file_runtimes: dict, total_seconds: float):
    gpu_info = get_gpu_info()
    lines = [
        "# GPU Runtime Log",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Hardware",
        f"GPU:              {gpu_info}",
        "",
        "## Model Config",
        f"Model:            {MODEL_ID}",
        "Quantization:     4-bit NF4 (BitsAndBytes, double quant)",
        "Compute dtype:    float16",
        "Decoding:         greedy (do_sample=False)",
        f"Max new tokens:   {MAX_NEW_TOKENS}",
        "",
        "## Per-File Runtime",
    ]
    for fname, secs in file_runtimes.items():
        lines.append(f"  {fname:<45} {secs/60:.1f} min")
    lines += [
        "",
        f"Total wall-clock time: {total_seconds/60:.1f} min ({total_seconds/3600:.2f} hr)",
    ]
    log_path = os.path.join(RESULTS_DIR, "gpu_runtime.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Runtime log saved to {log_path}")


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

    file_runtimes = {}
    total_start = time.time()

    for fname in prompt_files:
        t0 = time.time()
        run_file(fname, tokenizer, model)
        file_runtimes[fname] = time.time() - t0

    total_elapsed = time.time() - total_start
    write_runtime_log(file_runtimes, total_elapsed)
    print("All inference complete. Responses saved to outputs/responses/")
