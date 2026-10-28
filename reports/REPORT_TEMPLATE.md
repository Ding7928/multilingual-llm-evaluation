# Multilingual LLM Evaluation Report

## Executive summary
- What models and languages were compared?
- Which model was better on which language? State observed results, not assumptions.
- What practical actions follow from findings?

## Setup and reproducibility
- Date; device; Python, CUDA, PyTorch and Transformers versions
- Model IDs and revisions; dataset revision
- 0-shot prompt, seed, sample size per language, decoding settings
- Missing, invalid or failed predictions

## Quantitative findings
| Language | N | Qwen3 accuracy | Qwen2.5 accuracy | Delta (pp) | 95% paired bootstrap CI | Raw McNemar p | Bonferroni p |
|---|---:|---:|---:|---:|---|---:|---:|
| English | | | | | | | |
| Simplified Chinese | | | | | | | |
| Traditional Chinese | | | | | | | |

## Qualitative failure analysis
- Annotation scheme and sampling method
- Observed error categories and counts
- 3-5 anonymized/examples with evidence and caveats
- Actionable model/evaluation recommendations

## Limitations
- Sampling uncertainty and passage-level dependence
- Prompt and decoding dependence
- Possible benchmark contamination
- Differences in training/parameter count
- No causal attribution to model architecture or post-training process

## Next steps
- Robustness checks, alternate prompts, larger model comparisons
- Cluster bootstrap and independent manual annotation
