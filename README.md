# Multilingual LLM Evaluation & Failure Analysis

Compare Qwen3-0.6B and Qwen2.5-0.5B-Instruct on Belebele reading-comprehension questions in English, Simplified Chinese and Traditional Chinese. This is a learning/portfolio benchmark, not an official leaderboard result.

## 1. Prerequisites
Python 3.10+; internet access to download Hugging Face datasets/models; CPU or CUDA GPU. GPU recommended. Use only computational resources authorized for personal projects. A fresh virtual environment is strongly recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print('CUDA?', torch.cuda.is_available())"
```

## 2. Small smoke test: 10 questions * 3 languages * 2 models

```bash
python src/run_evaluation.py --model qwen3 --n 10 --seed 42
python src/run_evaluation.py --model qwen25 --n 10 --seed 42
python src/analyze.py
```

Files created in `results/`: `qwen3_predictions.csv`, `qwen25_predictions.csv`, `paired_results.csv`, `summary.csv`, `accuracy.png`, `failure_review.csv`.

## 3. Larger run

```bash
python src/run_evaluation.py --model qwen3 --n 300 --seed 42
python src/run_evaluation.py --model qwen25 --n 300 --seed 42
python src/analyze.py
```

Run with the **same n and seed** for both models. Files are overwritten at the beginning of each run. Each answer is checkpointed, but automatic resume is not implemented. For final results, run all 900 questions per language if feasible, and record versions/hardware.

## 4. Inspect and annotate errors

Open `results/failure_review.csv`, review model disagreements against the original question and passage. Fill `error_category`, `evidence`, and `recommendation`. Suggested categories: reasoning, context misunderstanding, language-specific, instruction-following, invalid output, other/uncertain. Do not infer failure mechanisms from the wrong answer alone; check manually. Review a random sample of cases both models got wrong too.

## 5. Write final report

Use `reports/REPORT_TEMPLATE.md`. Include methodology, actual sample sizes, observed accuracy, 95% paired-bootstrap CIs, exact McNemar p-values and limitations. Do not claim causal evidence of differences in training or architecture from this observational comparison.

## Notes / limitations
- `facebook/belebele` has **test** data and should not be used to fine-tune or tune prompts. Treat this as zero-shot evaluation. Model contamination cannot be ruled out.
- Belebele is parallel across languages; taking the same shuffled row indices gives comparable item sets. The 900 questions per language can share passages, so naive question-level bootstrap may underestimate uncertainty; for publication-grade inference use passage-cluster resampling and disclose this limitation.
- The models use different native chat templates and differ slightly in model size. Qwen3 thinking is explicitly disabled; greedy decoding is used for deterministic engineering comparison, although the Qwen3 model card recommends sampling for non-thinking generation. This setting is a controlled experimental choice, not necessarily optimal model performance.
- `parse_answer` intentionally requires a single choice; verbose responses count as invalid. Report invalid outputs separately and consider sensitivity analysis with a separately justified parser.
- Keep the benchmark license and attribution (CC BY-SA 4.0).

Sources: https://huggingface.co/datasets/facebook/belebele ; https://huggingface.co/Qwen/Qwen3-0.6B ; https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
