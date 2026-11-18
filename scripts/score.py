"""Scores a trained LFM3-TS checkpoint on held-out reading questions.

    python scripts/score.py runs/read/final

Prints the same reading score `train.py` prints after a run: how close the stated positions, periods and block means
are to the truth, on questions generated from a seed training never uses.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

import lfm3_ts  # noqa: F401  registers the LFM3-TS classes
from lfm3_ts.training import load_chatts_holdout
from train import reading_scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="an LFM3-TS checkpoint directory")
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    processor = AutoProcessor.from_pretrained(args.model)
    scores = reading_scores(model, processor, load_chatts_holdout({"reading": 1.0}, args.examples), args.examples)
    (Path(args.model) / "reading_scores.json").write_text(json.dumps(scores, indent=1))
    return scores


if __name__ == "__main__":
    main()
