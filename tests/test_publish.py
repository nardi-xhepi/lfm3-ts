"""Tests the publishing script up to the upload: what it would send, on a tiny random model."""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent))
from tiny import tiny_model, tiny_processor  # noqa: E402

SCRIPT = Path(__file__).parents[1] / "scripts" / "publish_to_hub.py"
spec = importlib.util.spec_from_file_location("publish_to_hub", SCRIPT)
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


@pytest.fixture
def checkpoint(tmp_path):
    folder = tmp_path / "checkpoint"
    tiny_model().save_pretrained(folder)
    tiny_processor().save_pretrained(folder)
    return folder


def test_model_card_placeholders_are_filled_or_refused():
    card = "Load `<hub-id>` from https://github.com/nardi-xhepi/<github-repo>. The token `<|ts|>` stays."
    filled = publish.fill_model_card(card, "someone/some-model", "some-code")
    assert filled == "Load `someone/some-model` from https://github.com/nardi-xhepi/some-code. The token `<|ts|>` stays."
    with pytest.raises(SystemExit, match="<github-repo>"):
        publish.fill_model_card(card, "someone/some-model", None)


def test_the_real_model_card_has_front_matter_and_only_the_two_known_placeholders():
    card = (SCRIPT.parents[1] / "MODEL_CARD.md").read_text()
    assert card.startswith("---\nlicense: other\nlicense_name: lfm1.0\n")
    assert "base_model: LiquidAI/LFM3-1.7B" in card and "library_name: transformers" in card
    assert set(publish.PLACEHOLDER.findall(card)) == {"<hub-id>", "<github-repo>"}
    assert "<" not in publish.fill_model_card(card, "someone/some-model", "some-code").replace("<|ts|>", "")


def test_dry_run_writes_what_would_be_uploaded_and_needs_no_token(checkpoint, tmp_path):
    """Run as a subprocess: `register_for_hub()` changes the classes for the rest of a process."""
    out = tmp_path / "hub"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "HF_HUB_OFFLINE": "1"}
    env.pop("HF_TOKEN", None)
    command = [sys.executable, str(SCRIPT), str(checkpoint), "someone/some-model", "--github-repo", "some-code"]
    result = subprocess.run([*command, "--dry-run", str(out)], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "nothing uploaded" in result.stdout
    files = {path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()}
    assert {
        "config.json",
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
        "configuration_lfm3_ts.py",
        "modeling_lfm3_ts.py",
        "processing_lfm3_ts.py",
        "feature_extraction_lfm3_ts.py",
        "README.md",
        "LICENSE",
        "NOTICE",
        "LICENSE-Apache-2.0.txt",
    } <= files
    assert (out / "LICENSE").read_text() == (SCRIPT.parents[1] / "LICENSE").read_text()
    card = (out / "README.md").read_text()
    assert 'from_pretrained("someone/some-model", trust_remote_code=True)' in card and "<hub-id>" not in card
    auto_map = json.loads((out / "config.json").read_text())["auto_map"]
    assert auto_map["AutoModelForCausalLM"] == "modeling_lfm3_ts.Lfm3TsForConditionalGeneration"
    assert json.loads((out / "processor_config.json").read_text())["auto_map"]["AutoProcessor"].endswith("Lfm3TsProcessor")

    # the folder loads as the Hub repository will: through the Auto classes, without the package
    load = textwrap.dedent(
        f"""
        import sys
        sys.modules["lfm3_ts"] = None
        import numpy as np
        from transformers import AutoModelForCausalLM, AutoProcessor
        model = AutoModelForCausalLM.from_pretrained({str(out)!r}, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained({str(out)!r}, trust_remote_code=True)
        print(type(model.chat(processor, np.arange(24.0), "which way", max_new_tokens=3)).__name__)
        """
    )
    loaded = subprocess.run(
        [sys.executable, "-c", load], capture_output=True, text=True, env={**env, "HF_MODULES_CACHE": str(tmp_path / "modules")}
    )
    assert loaded.returncode == 0 and loaded.stdout.strip().endswith("str"), loaded.stderr

    # without --dry-run and without a token it stops before anything else
    refused = subprocess.run(command, capture_output=True, text=True, env=env)
    assert refused.returncode != 0 and "HF_TOKEN" in refused.stderr


def test_bad_arguments_are_refused(checkpoint, tmp_path):
    for repo_id in ("<hub-id>", "no-user"):
        with pytest.raises(SystemExit, match="not a Hub repository id"):
            publish.main([str(checkpoint), repo_id, "--dry-run", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="not a directory"):
        publish.main([str(tmp_path / "missing"), "someone/some-model", "--dry-run", str(tmp_path / "out")])
