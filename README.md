# Cross-Lingual Hallucination Drift in LLMs
### Does It Depend on Task Type?

**Team:**  Bhoomika Monthy Rajashekar, Devinn Chi, Chun Hsu, Anagha P Krishna
**Course:** CS505 — Natural Language Processing, Boston University  
**Advisor:** Aaron Mueller

---

## Project Summary

We investigate whether cross-lingual hallucination drift in LLMs is
**task-dependent** — comparing factual QA (TruthfulQA) vs. commonsense
reasoning (XCOPA) using Aya Expanse 8B as the target model and GPT-4o-mini as judge.

**Language coverage per task:**
- TruthfulQA: English, Spanish (`alexandrainst/m_truthfulqa`)
- XCOPA: English, Swahili (`super_glue/copa` + `xcopa/sw`)

> **Note:** Spanish is not available in the XCOPA dataset (which covers 11 languages
> but not Spanish). Cross-task drift is compared as en→es drift (TruthfulQA)
> vs. en→sw drift (XCOPA).

---

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

---

## How to Run (in order)

```bash
python src/01_load_datasets.py       # download + save raw data
python src/02_sample_and_format.py   # sample 150 per split, build prompts
python src/03_run_inference.py       # run Aya Expanse 8B (needs GPU)
python src/04_run_judge.py           # GPT-4o-mini judge (needs API key)
python src/05_compute_metrics.py     # compute HR, ΔHR, Φₗ
python src/06_statistical_tests.py   # t-test / Wilcoxon tests
python src/07_visualize.py           # bar charts and tables
```

---

## Folder Structure

```
data/raw/          → original downloaded data (never edit)
                      truthfulqa_en.json  (817 examples)
                      truthfulqa_es.json  (789 examples)
                      xcopa_en.json       (500 examples, SuperGLUE COPA train+val)
                      xcopa_sw.json       (500 examples, XCOPA Swahili test)
data/sampled/      → 150-per-language samples (random seed 42)
data/prompts/      → formatted prompts ready for inference
outputs/responses/ → raw Aya model responses
outputs/labels/    → GPT-4o-mini judge labels
outputs/combined/  → responses + labels merged
results/tables/    → HR results table, p-values
results/figures/   → bar charts, drift plots
paper/             → LaTeX source files
notebooks/         → exploratory Jupyter notebooks
```

---

## Key Metrics

- **HR(l,t)** — Hallucination Rate per language per task
- **ΔHR(l,t)** — Drift vs English baseline
- **Φₗ** — Drift Interaction Score (TruthfulQA drift − XCOPA drift)
