"""Tests the one-call helper, and that a checkpoint saved for the Hub loads without the package installed."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

import lfm3_ts
from lfm3_ts import build_chat_messages


sys.path.insert(0, str(Path(__file__).parent))
from tiny import tiny_model, tiny_processor  # noqa: E402


def series_parts(messages):
    return [part["timeseries"] for part in messages[0]["content"] if part["type"] == "timeseries"]


def test_messages_hold_one_series_in_the_trained_wording():
    values = np.arange(12.0)
    for given in (values, values.tolist(), tuple(values), torch.tensor(values)):
        (message,) = build_chat_messages(given, "  Is there an anomaly? ")
        assert message["role"] == "user"
        assert [part["type"] for part in message["content"]] == ["text", "timeseries", "text"]
        assert message["content"][0]["text"] == "This is a time series of length 12: "
        assert message["content"][2]["text"] == ". Is there an anomaly?"
        assert np.array_equal(message["content"][1]["timeseries"], values)


def test_messages_hold_several_series_from_a_list_or_the_rows_of_an_array():
    first, second = np.arange(8.0), np.arange(5.0)
    messages = build_chat_messages([first, second.tolist()], "Which one rises faster?")
    assert [len(series) for series in series_parts(messages)] == [8, 5]
    text = "".join(part.get("text", "<ts>") for part in messages[0]["content"])
    assert text == (
        "There are 2 time series. Series 1, of length 8: <ts>; Series 2, of length 5: <ts>. Which one rises faster?"
    )
    rows = build_chat_messages(np.stack([first, first + 1]), "Compare them.")
    assert [len(series) for series in series_parts(rows)] == [8, 8]
    # a list holding a single series is one series
    assert build_chat_messages([first], "ok")[0]["content"][0]["text"] == "This is a time series of length 8: "


def test_messages_without_a_series_and_bad_inputs():
    assert build_chat_messages(None, "Hello") == [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    for bad in ([], "1,2,3", np.zeros((2, 3, 4)), [np.zeros((2, 2))], [[]]):
        with pytest.raises(ValueError):
            build_chat_messages(bad, "question")


def test_chat_returns_the_generated_text():
    model, processor = tiny_model(), tiny_processor()
    series = np.sin(np.arange(40) / 3)
    answer = model.chat(processor, series, "which way", max_new_tokens=5)
    assert isinstance(answer, str)
    # the same as building the conversation and generating by hand, greedily
    inputs = processor.apply_chat_template(
        build_chat_messages(series, "which way"),
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    assert "input_values" in inputs
    output = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    expected = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    assert answer == expected
    assert lfm3_ts.chat(model, processor, series, "which way", max_new_tokens=5) == answer
    # several series, a list of numbers, and no series at all
    assert isinstance(model.chat(processor, [series, series[:17].tolist()], "which way", max_new_tokens=5), str)
    assert isinstance(model.chat(processor, series.tolist(), "which way", max_new_tokens=3), str)
    assert isinstance(model.chat(processor, None, "which way", max_new_tokens=3), str)


def test_chat_reads_the_series():
    """Two different series give different next-token scores: the helper really hands the series to the model."""
    model, processor = tiny_model(), tiny_processor()
    logits = []
    for series in (np.linspace(0, 1, 40), np.r_[np.zeros(20), 5.0, np.zeros(19)]):
        inputs = processor.apply_chat_template(
            build_chat_messages(series, "which way"), add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
        logits.append(model(**inputs).logits[0, -1])
    assert not torch.allclose(logits[0], logits[1])


def test_importing_the_package_does_not_import_the_training_code_or_its_dependencies():
    code = textwrap.dedent(
        """
        import sys
        for name in ("datasets", "accelerate", "peft"):
            sys.modules[name] = None  # importing any of them now raises ImportError
        import lfm3_ts
        from transformers import AutoModelForCausalLM, AutoProcessor
        assert "lfm3_ts.training" not in sys.modules
        assert hasattr(lfm3_ts.Lfm3TsForConditionalGeneration, "chat")
        print("ok")
        """
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0 and result.stdout.strip().endswith("ok"), result.stderr


def test_a_checkpoint_saved_for_the_hub_loads_with_trust_remote_code_and_without_the_package(tmp_path):
    """What `scripts/publish_to_hub.py` relies on: `register_for_hub()` puts the source files next to the weights,
    and a machine without `lfm3_ts` loads them through the Auto classes and can call `model.chat`."""
    checkpoint = tmp_path / "checkpoint"
    save = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        import lfm3_ts
        from tiny import tiny_model, tiny_processor
        lfm3_ts.register_for_hub()
        tiny_model().save_pretrained({str(checkpoint)!r})
        tiny_processor().save_pretrained({str(checkpoint)!r})
        """
    )
    load = textwrap.dedent(
        f"""
        import sys
        sys.modules["lfm3_ts"] = None  # as on a machine where the package is not installed
        import numpy as np
        from transformers import AutoModelForCausalLM, AutoProcessor
        model = AutoModelForCausalLM.from_pretrained({str(checkpoint)!r}, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained({str(checkpoint)!r}, trust_remote_code=True)
        assert type(model).__module__.startswith("transformers_modules."), type(model).__module__
        assert type(processor).__module__.startswith("transformers_modules."), type(processor).__module__
        answer = model.chat(processor, np.sin(np.arange(40) / 3), "which way", max_new_tokens=4)
        assert isinstance(answer, str)
        print("ok")
        """
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "HF_MODULES_CACHE": str(tmp_path / "modules")}
    for code in (save, load):
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
    shipped = {path.name for path in checkpoint.glob("*.py")}
    assert shipped == {
        "configuration_lfm3_ts.py",
        "feature_extraction_lfm3_ts.py",
        "modeling_lfm3_ts.py",
        "processing_lfm3_ts.py",
    }
