import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "qwen3": "Qwen/Qwen3-0.6B",
    "qwen25": "Qwen/Qwen2.5-0.5B-Instruct",
}
LANGUAGES = ["eng_Latn", "zho_Hans", "zho_Hant"]
LETTERS = "ABCD"


def make_prompt(row):
    answers = "\n".join(f"{letter}. {row[f'mc_answer{i}']}" for i, letter in enumerate(LETTERS, 1))
    return (
        "Read the passage and answer the multiple-choice question. "
        "Reply with only one capital letter: A, B, C, or D.\n\n"
        f"Passage:\n{row['flores_passage']}\n\n"
        f"Question:\n{row['question']}\n\n"
        f"Options:\n{answers}\n\nAnswer:"
    )


def parse_answer(text):
    # Strict parsing: no cherry-picking an answer from verbose explanations.
    candidate = text.strip()
    match = re.fullmatch(r"(?:Answer\s*[:：]\s*)?([ABCD])(?:[.。])?", candidate, re.IGNORECASE)
    return match.group(1).upper() if match else None


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32
    )

    model = model.to("cpu")
    model.eval()

    print("Model device:", model.device)

    return model, tokenizer


def infer(model, tokenizer, prompt, model_id, max_new_tokens=16):
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if model_id.startswith("Qwen/Qwen3-"):
        kwargs["enable_thinking"] = False
    text = tokenizer.apply_chat_template(messages, **kwargs)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    out = generated[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(out, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--n", type=int, default=10, help="questions per language; up to 900")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()
    if not 1 <= args.n <= 900:
        parser.error("--n must be between 1 and 900")
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.model}_predictions.csv"
    model_id = MODELS[args.model]
    model, tokenizer = load_model(model_id)
    records = []
    start = time.time()
    for lang in LANGUAGES:
        dataset = load_dataset("facebook/belebele", lang, split="test")
        # Same indices/seed across models AND languages, using parallel items.
        subset = dataset.shuffle(seed=args.seed).select(range(args.n))
        for i, row in enumerate(subset):
            gold = LETTERS[int(row["correct_answer_num"]) - 1]
            raw = infer(model, tokenizer, make_prompt(row), model_id)
            pred = parse_answer(raw)
            records.append({
                "model": args.model, "model_id": model_id,
                "language": lang, "item_index": i,
                "link": row["link"], "question_number": row["question_number"],
                "passage": row["flores_passage"], "question": row["question"],
                "gold": gold, "prediction": pred or "INVALID",
                "raw_output": raw, "correct": int(pred == gold),
            })
            print(f"{args.model} {lang} {i + 1}/{args.n}: gold={gold}, pred={pred or 'INVALID'}", flush=True)
            # Checkpoint output after every example for recovery.
            pd.DataFrame(records).to_csv(output_file, index=False)
    print(f"Saved {len(records)} rows to {output_file}. Elapsed: {time.time()-start:.1f}s")
    print("Environment:", json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "model": model_id}))


if __name__ == "__main__":
    main()
