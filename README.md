# Cross-Lingual-Hallucination-Drift-in-LLMs

## Team
Bhoomika Monthy Rajashekar, Devinn Chi, Chun Hsu, Anagha P Krishna — Boston University CS505

## Project Summary
We investigate whether cross-lingual hallucination drift is 
task-dependent, comparing factual QA (TruthfulQA) vs. commonsense 
reasoning (XCOPA) across English, Spanish, and Swahili using 
Aya Expanse 8B as the target model and GPT-4o-mini as judge.

## How to Run (in order)
1. python src/01_load_datasets.py
2. python src/02_sample_and_format.py
3. python src/03_run_inference.py        # needs GPU
4. python src/04_run_judge.py            # needs OpenAI API key
5. python src/05_compute_metrics.py
6. python src/06_statistical_tests.py
7. python src/07_visualize.py

## Setup
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
