# Cross-Lingual Hallucination Drift in LLMs
### Does It Depend on Task Type?

**Team:** Bhoomika Monthy Rajashekar, Devinn Chi, Chun Hsu, Anagha P Krishna  
**Course:** CS505 — Natural Language Processing, Boston University  
**Advisor:** Aaron Mueller

---

## Overview

We investigate whether cross-lingual hallucination drift in LLMs is **task-dependent**. We evaluate **Aya Expanse 8B** across English, Spanish, Italian, and Swahili on two structurally different tasks — factual QA (TruthfulQA) and commonsense reasoning (XCOPA) — using GPT-4o-mini as an LLM-as-a-Judge evaluator.

Italian appears in **both** benchmarks, enabling a clean within-language confound test: any difference in hallucination pattern between the two tasks, holding language constant, must be attributable to task type alone.

---

## Key Results

### Hallucination Rates

| Task | Language | HR (%) | ΔHR vs EN |
|------|----------|-------:|----------:|
| TruthfulQA | English  | 27.33 | —          |
| TruthfulQA | Spanish  | 24.67 | −2.66 pp   |
| TruthfulQA | Italian  | 28.67 | +1.34 pp   |
| XCOPA      | English  |  8.00 | —          |
| XCOPA      | Italian  |  9.33 | +1.33 pp   |
| XCOPA      | Swahili  | 98.67 | +90.67 pp  |

### Drift Interaction Score (Φ)

| Comparison | Φ (pp) | χ² | p |
|---|---:|---:|---|
| es/TruthfulQA vs sw/XCOPA | −93.33 | 45.44 | < 0.001 |
| it/TruthfulQA vs it/XCOPA (within-language) | +0.01 | 16.98 | < 0.001 |

**Interpretation:** XCOPA Swahili shows catastrophic drift (+90.67 pp) while TruthfulQA drift is negligible across all languages. The within-Italian test (same language, both tasks) confirms this is driven by task type, not language identity.

### Error Analysis

| Task | Language | Incoherent | Wrong Answer | Total Hallucinated |
|------|----------|-----------:|-------------:|-------------------:|
| TruthfulQA | English  |  0 | 14 |  41 |
| TruthfulQA | Spanish  |  1 | 11 |  37 |
| TruthfulQA | Italian  |  0 | 15 |  43 |
| XCOPA      | English  |  0 |  5 |  12 |
| XCOPA      | Italian  |  0 |  6 |  14 |
| XCOPA      | Swahili  | 40 |  5 | 148 |

Swahili XCOPA hallucinations are predominantly **incoherent** (27%) with responses averaging 185 tokens — nearly 2× the English baseline (89 tokens).

### Dual-Judge Agreement (Italian, 50 samples)

| Cell | κ GPT/Claude | κ GPT/Human | κ Claude/Human |
|------|---:|---:|---:|
| TruthfulQA / Italian | 0.662 | 0.597 | 0.712 |
| XCOPA / Italian      | 0.065 | 0.256 | 0.065 |

---

## Streamlit Dashboard

An interactive dashboard is included for exploring all results.

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**. Five pages:

| Page | Contents |
|------|----------|
| **Dashboard** | KPI metrics, full HR table, Φ scores, token verbosity chart |
| **Charts** | HR by language/task, drift bar chart, heatmap, statistical tests |
| **New Analyses** | XCOPA cause/effect, dual-judge κ, error category breakdown |
| **Example Browser** | Browse individual responses and judge labels with filters |
| **Reason Analysis** | Hallucination category distribution, cross-cell heatmap |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"   # required for dual-judge script only
```

---

## Pipeline

Run scripts in order. Scripts 03 requires a GPU; all others run on CPU.

```bash
# 1. Download datasets from HuggingFace
python src/01_load_datasets.py

# 2. Sample 150 examples/cell, build zero-shot prompts
python src/02_sample_and_format.py

# 3. Run Aya Expanse 8B inference  ← GPU required (Google Colab T4 recommended)
python src/03_run_inference.py

# 4. GPT-4o-mini judge + retry any failed labels
python src/04_run_judge.py
python src/04_retry_errors.py

# 5. Compute hallucination rates and Φ scores
python src/05_compute_metrics.py

# 6. Statistical significance tests
python src/06_statistical_tests.py

# 7. Generate figures
python src/07_visualize.py

# 8. XCOPA cause vs effect breakdown
python src/08_xcopa_cause_effect.py

# 9. Dual-judge (Claude) + κ on Italian samples
python src/09_dual_judge_italian.py

# 10. Error categorization across all cells
python src/10_error_analysis.py

# Dashboard
streamlit run app.py
```

---

## Model & Hardware Config

| Setting | Value |
|---------|-------|
| Target model | CohereLabs/aya-expanse-8b |
| Quantization | 4-bit NF4 (BitsAndBytes, double quant) |
| Compute dtype | float16 |
| Decoding | Greedy (do_sample=False) |
| Max new tokens | 200 |
| Hardware | Google Colab T4 GPU (15.8 GB VRAM) |
| Estimated runtime | ~1.5–2 hours for all 6 cells |
| Primary judge | GPT-4o-mini (OpenAI API) |
| Secondary judge | claude-sonnet-4-5 (Anthropic API) |

---

## Repository Structure

```
cross_lingual_drift/
├── data/
│   ├── raw/              # Downloaded HuggingFace datasets (6 files)
│   ├── sampled/          # 150-example samples per cell (6 files)
│   └── prompts/          # Formatted zero-shot prompts (6 files)
│
├── outputs/
│   ├── responses/        # Aya Expanse 8B responses with token counts (6 files)
│   └── labels/
│       ├── *.json        # GPT-4o-mini judge labels (6 files, 150 items each)
│       └── dual_judge/   # Claude second-judge outputs + κ summary
│
├── results/
│   ├── tables/           # CSV outputs (HR, Φ, stats, error categories, cause/effect)
│   ├── figures/          # PNG charts (hr_by_language_task, drift_by_task)
│   └── gpu_runtime.txt   # Model config and estimated runtime
│
├── src/
│   ├── 01_load_datasets.py
│   ├── 02_sample_and_format.py
│   ├── 03_run_inference.py       # GPU — logs runtime to results/gpu_runtime.txt
│   ├── 04_run_judge.py
│   ├── 04_retry_errors.py
│   ├── 05_compute_metrics.py
│   ├── 06_statistical_tests.py
│   ├── 07_visualize.py
│   ├── 08_xcopa_cause_effect.py
│   ├── 09_dual_judge_italian.py
│   └── 10_error_analysis.py
│
├── paper/
│   ├── midway_report.tex         # Midterm report (15/15)
│   ├── references.bib
│   └── acl.sty / acl_natbib.bst
│
├── app.py                        # Streamlit dashboard
└── requirements.txt
```

---

## Datasets

| Benchmark | Task Type | Languages | Source |
|-----------|-----------|-----------|--------|
| TruthfulQA | Factual QA | English, Spanish, Italian | `alexandrainst/m_truthfulqa` |
| XCOPA | Commonsense reasoning | English, Italian, Swahili | `xcopa` (HuggingFace) |

150 examples sampled per cell (random seed 42). Spanish is not available in XCOPA; Italian was added as a **shared language** across both tasks to serve as a confound control.
