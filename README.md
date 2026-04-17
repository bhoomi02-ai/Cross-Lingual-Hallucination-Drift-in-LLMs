# Cross-Lingual Hallucination Drift in LLMs
### Does It Depend on Task Type?

**Team:** Bhoomika Monthy Rajashekar, Devinn Chi, Chun Hsu, Anagha P Krishna  
**Course:** CS505 — Natural Language Processing, Boston University  
**Advisor:** Aaron Mueller

---

## Project Summary

We investigate whether cross-lingual hallucination drift in LLMs is **task-dependent** — comparing factual QA (TruthfulQA) vs. commonsense reasoning (XCOPA) using Aya Expanse 8B as the target model and GPT-4o-mini as the judge.

**Language coverage per task:**
- TruthfulQA: English, Spanish (`alexandrainst/m_truthfulqa`)
- XCOPA: English, Swahili (`super_glue/copa` + `xcopa/sw`)

> **Note:** Spanish is not available in XCOPA (which covers 11 languages but not Spanish).
> Cross-task drift is compared as en→es drift (TruthfulQA) vs. en→sw drift (XCOPA),
> using the cross-task aggregate Φ formula.

---

## Results

### Hallucination Rates

| Task | Language | HR (%) | ΔHR vs English (pp) |
|------|----------|-------:|--------------------:|
| TruthfulQA | English | 27.33 | — |
| TruthfulQA | Spanish | 24.67 | −2.66 |
| XCOPA | English | 8.00 | — |
| XCOPA | Swahili | 98.67 | +90.67 |

### Drift Interaction Score (Φ)

**Φ = ΔHR(es, TruthfulQA) − ΔHR(sw, XCOPA) = −93.33 pp**

Swahili XCOPA drift (+90.67 pp) completely dominates Spanish TruthfulQA drift (−2.66 pp), indicating that cross-lingual hallucination is heavily task-dependent.

### Statistical Tests

| Test | Comparison | Statistic | p-value | Significant |
|------|------------|----------:|--------:|:-----------:|
| Mann-Whitney U | XCOPA: en vs sw | U = 1050.0 | < 0.001 | YES |
| Mann-Whitney U | TruthfulQA: en vs es | U = 11550.0 | 0.600 | NO |
| Chi-squared (task × outcome) | es/TruthfulQA vs sw/XCOPA | χ² = 170.62 | < 0.001 | YES |
| Fisher's exact (robustness) | es/TruthfulQA vs sw/XCOPA | OR = 226.0 | < 0.001 | YES |

### Token Verbosity Finding

When hallucinating in Swahili XCOPA, Aya generates nearly 2× more tokens (186 vs 100 avg), suggesting the model rambles when it lacks grounding in the target language.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
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
python src/04_retry_errors.py        # retry any ERROR labels
python src/05_compute_metrics.py     # compute HR, ΔHR, Φ
python src/06_statistical_tests.py   # Mann-Whitney, chi-squared, Fisher's exact
python src/07_visualize.py           # bar charts and drift plots
```

### Interactive Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with four pages: Dashboard, Charts, Example Browser, and Reason Analysis.

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
results/tables/    → HR results, Φ scores, statistical tests
results/figures/   → bar charts, drift plots
paper/             → LaTeX source files
```

---

## Key Metrics

- **HR(l,t)** — Hallucination Rate for language `l` on task `t`
- **ΔHR(l,t)** — Drift vs. English baseline for the same task
- **Φ** — Cross-task Drift Interaction Score: `ΔHR(es, TruthfulQA) − ΔHR(sw, XCOPA)`
