import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import matplotlib.pyplot as plt


def paired_bootstrap(a, b, seed=42, iterations=10000):
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(iterations)
    for i in range(iterations):
        idx = rng.integers(0, n, n)
        diffs[i] = np.mean(a[idx] - b[idx])
    return np.quantile(diffs, [0.025, 0.975])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results")
    args = parser.parse_args()
    directory = Path(args.dir)
    a = pd.read_csv(directory / "qwen3_predictions.csv")
    b = pd.read_csv(directory / "qwen25_predictions.csv")
    key = ["language", "item_index"]
    assert not a.duplicated(key).any() and not b.duplicated(key).any(), "Duplicate question keys"
    merged = a.merge(b, on=key, suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError("Predictions differ in number of questions; run both with same --n/--seed")
    if not (merged.gold_a == merged.gold_b).all() or not (merged.question_a == merged.question_b).all():
        raise ValueError("Mismatched question alignment: check identical seed and dataset revision")
    results = []
    for lang, g in merged.groupby("language", sort=True):
        x, y = g.correct_a.to_numpy(dtype=int), g.correct_b.to_numpy(dtype=int)
        ci = paired_bootstrap(x, y)
        a_only = int(((x == 1) & (y == 0)).sum())
        b_only = int(((x == 0) & (y == 1)).sum())
        discordant = a_only + b_only
        # Exact two-sided McNemar is a binomial test on discordant pairs.
        p = float(binomtest(a_only, discordant, 0.5).pvalue) if discordant else 1.0
        results.append({
            "language": lang, "n": len(g), "qwen3_accuracy": x.mean(),
            "qwen25_accuracy": y.mean(), "difference": x.mean()-y.mean(),
            "ci_lower": ci[0], "ci_upper": ci[1],
            "qwen3_only_correct": a_only, "qwen25_only_correct": b_only,
            "mcnemar_p_raw": p,
            "qwen3_invalid": int((g.prediction_a == "INVALID").sum()),
            "qwen25_invalid": int((g.prediction_b == "INVALID").sum()),
        })
    out = pd.DataFrame(results)
    # Bonferroni correction across the 3 language-specific tests.
    out["mcnemar_p_bonferroni"] = (out.mcnemar_p_raw * len(out)).clip(upper=1.0)
    out.to_csv(directory / "summary.csv", index=False)
    merged.to_csv(directory / "paired_results.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pos = np.arange(len(out))
    ax.bar(pos - 0.18, out.qwen3_accuracy * 100, width=0.36, label="Qwen3")
    ax.bar(pos + 0.18, out.qwen25_accuracy * 100, width=0.36, label="Qwen2.5")
    ax.set_xticks(pos, out.language)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "accuracy.png", dpi=160)
    plt.close(fig)
    # Cases in which only one model is correct, prioritize for manual review.
    discordant = merged[merged.correct_a != merged.correct_b].copy()
    discordant["error_category"] = ""
    discordant["evidence"] = ""
    discordant["recommendation"] = ""
    discordant.to_csv(directory / "failure_review.csv", index=False)
    print(f"Created {directory / 'accuracy.png'} and {directory / 'failure_review.csv'}")


if __name__ == "__main__":
    main()
