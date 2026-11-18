"""Publishes a trained LFM3-TS checkpoint to the Hugging Face Hub, with its model code, model card and licence.

    # 1. write everything that would be uploaded to a folder, and look at it (no network, no token)
    python scripts/publish_to_hub.py runs/joint/final <user>/<name> --dry-run out/hub

    # 2. upload (the token is read from the environment variable HF_TOKEN; the repository is created private)
    HF_TOKEN=... python scripts/publish_to_hub.py runs/joint/final <user>/<name>

What it does: loads the checkpoint, calls `lfm3_ts.register_for_hub()` so that `save_pretrained` writes the source
files of the model next to the weights (the checkpoint then loads with `trust_remote_code=True` on a machine without
this package), saves the model and the processor to a folder, adds the model card as `README.md` with the
placeholders filled in, adds LICENSE, NOTICE and the Apache licence, and uploads the folder in one commit.
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# shipped with the weights: the LFM Open License asks that recipients get a copy of it, and the notices with it
LEGAL_FILES = {"LICENSE": "LICENSE", "NOTICE": "NOTICE", "LICENSES/Apache-2.0.txt": "LICENSE-Apache-2.0.txt"}
PLACEHOLDER = re.compile(r"<[a-z][a-z-]*>")


def fill_model_card(text: str, repo_id: str, github_repo: str | None) -> str:
    """The model card with its placeholders replaced; refuses to publish a card that still has one."""
    text = text.replace("<hub-id>", repo_id)
    if github_repo:
        text = text.replace("<github-repo>", github_repo)
    # only outside code: `<|ts|>` and the like are not placeholders, and the pattern does not match them
    left = sorted(set(PLACEHOLDER.findall(text)))
    if left:
        raise SystemExit(f"the model card still holds placeholders: {', '.join(left)} (pass --github-repo?)")
    return text


def prepare(checkpoint: str, folder: Path, repo_id: str, model_card: Path, github_repo: str | None, dtype: str | None):
    """Writes into `folder` exactly what the Hub repository will hold."""
    import torch
    from transformers import AutoProcessor

    import lfm3_ts

    card = fill_model_card(model_card.read_text(), repo_id, github_repo)
    # from here on `save_pretrained` copies the source files of the classes and writes `auto_map` into the configs
    lfm3_ts.register_for_hub()
    kwargs = {"dtype": getattr(torch, dtype)} if dtype else {}
    model = lfm3_ts.Lfm3TsForConditionalGeneration.from_pretrained(checkpoint, **kwargs)
    processor = AutoProcessor.from_pretrained(checkpoint)
    if not isinstance(processor, lfm3_ts.Lfm3TsProcessor):
        raise SystemExit(f"{checkpoint} does not hold an LFM3-TS processor, got {type(processor).__name__}")

    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    processor.save_pretrained(folder)
    (folder / "README.md").write_text(card)
    for source, name in LEGAL_FILES.items():
        shutil.copyfile(ROOT / source, folder / name)

    expected = ["config.json", "configuration_lfm3_ts.py", "modeling_lfm3_ts.py", "processing_lfm3_ts.py"]
    expected += ["feature_extraction_lfm3_ts.py", "processor_config.json", "README.md", "LICENSE", "NOTICE"]
    missing = [name for name in expected if not (folder / name).exists()]
    if missing or not list(folder.glob("*.safetensors")):
        raise SystemExit(f"the folder to upload is incomplete: missing {missing or 'the weights'}")
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="a local LFM3-TS checkpoint directory, e.g. runs/joint/final")
    parser.add_argument("repo_id", help="the Hub repository to publish to, as <user>/<name>")
    parser.add_argument("--model-card", default=str(ROOT / "MODEL_CARD.md"))
    parser.add_argument("--github-repo", help="the name of the code repository, to fill <github-repo> in the model card")
    parser.add_argument("--dtype", help="cast the weights before saving, e.g. bfloat16 (default: as the checkpoint loads)")
    parser.add_argument("--public", action="store_true", help="create the repository public (default: private)")
    parser.add_argument("--commit-message", default="Upload LFM3-TS")
    parser.add_argument("--dry-run", metavar="FOLDER", help="write what would be uploaded to FOLDER and stop")
    args = parser.parse_args(argv)

    if "/" not in args.repo_id or PLACEHOLDER.search(args.repo_id):
        raise SystemExit(f"`{args.repo_id}` is not a Hub repository id: give <user>/<name>")
    if not Path(args.checkpoint).is_dir():
        raise SystemExit(f"{args.checkpoint} is not a directory")

    if args.dry_run:
        files = prepare(args.checkpoint, Path(args.dry_run), args.repo_id, Path(args.model_card), args.github_repo, args.dtype)
        print(f"dry run: {len(files)} files in {args.dry_run}, nothing uploaded:\n  " + "\n  ".join(files))
        return files

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set the environment variable HF_TOKEN to a token with write access (or use --dry-run)")
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary) / "hub"
        files = prepare(args.checkpoint, folder, args.repo_id, Path(args.model_card), args.github_repo, args.dtype)
        api = HfApi(token=token)
        api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(folder), commit_message=args.commit_message)
    print(f"uploaded {len(files)} files to https://huggingface.co/{args.repo_id} ({'public' if args.public else 'private'})")
    return files


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
