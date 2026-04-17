# Cross-Lingual Hallucination Drift in LLMs

**Team:** Bhoomika Monthy Rajashekar, Devinn Chi, Chun Hsu, Anagha P Krishna  
**Course:** CS505 — NLP, Boston University | **Advisor:** Aaron Mueller

Does hallucination drift in LLMs depend on task type? We compare Aya Expanse 8B across factual QA (TruthfulQA) and commonsense reasoning (XCOPA), using GPT-4o-mini as a judge across English, Spanish, and Swahili.

---

## Key Results

| Task | Language | HR (%) | ΔHR vs EN |
|------|----------|-------:|----------:|
| TruthfulQA | English | 27.33 | — |
| TruthfulQA | Spanish | 24.67 | −2.66 pp |
| XCOPA | English | 8.00 | — |
| XCOPA | Swahili | 98.67 | +90.67 pp |

**Φ = −93.33 pp** — drift is strongly task-dependent (χ² = 170.62, p < 0.001).

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

## Run

```bash
python src/01_load_datasets.py      # download data
python src/02_sample_and_format.py  # sample 150/split, build prompts
python src/03_run_inference.py      # Aya Expanse 8B inference (GPU required)
python src/04_run_judge.py          # GPT-4o-mini judge
python src/04_retry_errors.py       # retry any failed labels
python src/05_compute_metrics.py    # HR, ΔHR, Φ
python src/06_statistical_tests.py  # significance tests
python src/07_visualize.py          # figures
streamlit run app.py                # interactive dashboard → http://localhost:8501
```

## Structure

```
data/          → raw, sampled, and formatted prompts
outputs/       → model responses and judge labels
results/       → tables (CSV) and figures (PNG)
src/           → pipeline scripts (01–07)
app.py         → Streamlit dashboard
```
