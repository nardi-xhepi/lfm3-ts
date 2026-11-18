"""Writes ordinary conversations answered by the original LFM3-1.7B, to mix into training (the `chat` stream).

    python scripts/make_chat_replay.py --out chat_replay.jsonl
    python scripts/train.py --stage align --chat-data chat_replay.jsonl --mixture ...,chat:0.15 ...

Training every weight on answers about series changes how the model answers everything else. The prompts here come
from an open instruction dataset (databricks-dolly-15k) and the answers are those of the untrained model: keeping
them in the loss holds the model to what it did before, in its own style.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def answer_all(model, tokenizer, prompts, batch_size=64, max_new_tokens=384, keep_unfinished=False):
    """Yields `(prompt, answer)` for the prompts the model finishes answering within `max_new_tokens`: an answer cut
    short would teach the model to stop in the middle."""
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    ordered = sorted(prompts, key=len)  # similar lengths together: little padding
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        inputs = tokenizer.apply_chat_template(
            [[{"role": "user", "content": prompt}] for prompt in batch],
            add_generation_prompt=True,
            enable_thinking=False,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        for prompt, generated in zip(batch, output[:, inputs["input_ids"].shape[1] :]):
            finished = bool((generated == tokenizer.eos_token_id).any())
            answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
            if answer and (finished or keep_unfinished):
                yield prompt, answer


def load_prompts(dataset: str, count: int, max_characters: int = 1500) -> list[str]:
    from datasets import load_dataset

    prompts = []
    for row in load_dataset(dataset, split="train"):
        prompt = row["instruction"].strip() + ("\n\n" + row["context"].strip() if row.get("context") else "")
        if prompt and len(prompt) <= max_characters:
            prompts.append(prompt)
        if len(prompts) == count:
            break
    return prompts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--examples", type=int, default=8000, help="questions to ask; unfinished answers are dropped")
    parser.add_argument("--model", default="LiquidAI/LFM3-1.7B", help="the model whose answers are kept")
    parser.add_argument("--dataset", default="databricks/databricks-dolly-15k", help="where the questions come from")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if device == "cuda" else None)
    model = model.to(device).eval()
    prompts = load_prompts(args.dataset, args.examples)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(args.out, "w") as output:
        for prompt, answer in answer_all(model, tokenizer, prompts, args.batch_size, args.max_new_tokens):
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
            output.write(json.dumps({"messages": messages}) + "\n")
            kept += 1
            if kept % 500 == 0:
                print(f"  {kept} conversations", flush=True)
    print(f"asked {len(prompts)} questions, kept the {kept} answered in full: {args.out}")


if __name__ == "__main__":
    main()
