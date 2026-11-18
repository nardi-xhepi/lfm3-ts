"""Turns a checkpoint written during training into a model that can be loaded and scored.

    python scripts/export.py runs/read/final runs/long/trainable-step2000.safetensors runs/long/step2000

`train.py` saves only the trained weights while it runs, and the whole model at the end. This puts the trained
weights of a run with a frozen or fully trained language model back into the model the run started from. Adapter
checkpoints cannot be exported this way: their weights only exist inside the training run.
"""

import argparse

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor

from lfm3_ts import Lfm3TsForConditionalGeneration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", help="the LFM3-TS checkpoint the run started from (its --model)")
    parser.add_argument("trainable", help="a trainable-step*.safetensors file of the run")
    parser.add_argument("output", help="where to write the model")
    args = parser.parse_args(argv)

    model = Lfm3TsForConditionalGeneration.from_pretrained(args.base, dtype=torch.float32)
    state = load_file(args.trainable)
    if any("lora_" in name for name in state):
        raise SystemExit("this is an adapter checkpoint: only the end of an adapter run gives a loadable model")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise SystemExit(f"weights the base model does not have: {result.unexpected_keys[:5]}")
    model.to(torch.bfloat16).save_pretrained(args.output)
    AutoProcessor.from_pretrained(args.base).save_pretrained(args.output)
    print(f"put {len(state)} trained tensors into {args.base}; saved {args.output}")


if __name__ == "__main__":
    main()
