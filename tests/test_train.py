"""Runs the training script end to end on a tiny random model and made-up records: seconds, no download."""

import importlib.util
import itertools
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, AutoProcessor, PreTrainedTokenizerFast

from lfm3_ts import Lfm3TsConfig, Lfm3TsFeatureExtractor, Lfm3TsForConditionalGeneration, Lfm3TsProcessor
from lfm3_ts.training import (
    Lfm3TsDataCollator,
    MixedStream,
    ReadingTasks,
    ScheduledStream,
    TokenBudgetBatches,
    chatts_record_to_messages,
    fix_stated_lengths,
)


spec = importlib.util.spec_from_file_location("train", Path(__file__).parents[1] / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

TS_TOKEN_ID = 10
# a ChatML-like template with the `timeseries` content part and the thinking switch of the LFM3 one
CHAT_TEMPLATE = (
    "<bos>{% for message in messages %}<start>{{ message['role'] }} "
    "{% if message['content'] is string %}{{ message['content'] }}{% else %}"
    "{% for part in message['content'] %}"
    "{% if part['type'] == 'timeseries' %}<|ts|>{% else %}{{ part['text'] }}{% endif %}"
    "{% endfor %}{% endif %}<eos>{% endfor %}"
    "{% if add_generation_prompt %}<start>assistant {% if enable_thinking %}<think>{% endif %}{% endif %}"
)


def tiny_model() -> Lfm3TsForConditionalGeneration:
    torch.manual_seed(0)
    config = Lfm3TsConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "layer_types": ["conv", "full_attention", "conv", "sliding_attention"],
            "sliding_window": 8,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        ts_config={"patch_size": 4, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2},
        ts_token_id=TS_TOKEN_ID,
        projector_hidden_size=24,
    )
    return Lfm3TsForConditionalGeneration(config)


def tiny_processor() -> Lfm3TsProcessor:
    words = ["<pad>", "<bos>", "<eos>", "<unk>", "<start>", "<think>", "user", "assistant", "which", "way"]
    vocab = {word: index for index, word in enumerate(words)}
    vocab.update({"<|ts|>": TS_TOKEN_ID, "up": 11, "down": 12})
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        extra_special_tokens={"ts_token": "<|ts|>", "start_token": "<start>", "think_token": "<think>"},
    )
    return Lfm3TsProcessor(Lfm3TsFeatureExtractor(patch_size=4), tokenizer, chat_template=CHAT_TEMPLATE)


def records(count: int = 64) -> list[dict]:
    """Rising or falling ramps in the ChatTS record format; the answer names the direction."""
    rng = np.random.default_rng(0)
    data = []
    for index in range(count):
        rising = index % 2 == 0
        ramp = np.linspace(0, 1, int(rng.integers(8, 20))) * (1 if rising else -1) + rng.normal(scale=0.05)
        data.append({"input": "which way <ts><ts/>", "timeseries": [ramp.tolist()], "output": "up" if rising else "down"})
    return data


def run(tmp_path, *extra, model=None, processor=None):
    argv = ["--stage", "dev", "--output-dir", str(tmp_path), "--max-tokens", "400", "--grad-accum", "1"]
    argv += ["--logging-steps", "5", "--save-steps", "10", "--num-workers", "0", "--samples", "0", "--cpu", *extra]
    train.main(
        argv,
        model=model or tiny_model(),
        processor=processor or tiny_processor(),
        dataset=MixedStream([records()], [1.0]),
    )
    return [entry["loss"] for entry in json.loads((tmp_path / "log.json").read_text())["steps"] if "loss" in entry]


def test_chatts_record_to_messages():
    record = {"input": "a <ts><ts/> b <ts><ts/>", "timeseries": [[1.0, 2.0], [3.0]], "output": "fine"}
    user, assistant = chatts_record_to_messages(record)
    assert [part["type"] for part in user["content"]] == ["text", "timeseries", "text", "timeseries"]
    assert user["content"][3]["timeseries"].tolist() == [3.0]
    assert assistant == {"role": "assistant", "content": "fine"}
    with pytest.raises(ValueError, match="placeholders"):
        chatts_record_to_messages({"input": "<ts><ts/>", "timeseries": [], "output": ""})


def test_fix_stated_lengths_only_rewrites_exact_doublings():
    series = [[0.0] * 256]
    assert fix_stated_lengths('This is "CPM" with length of 512: x', series) == 'This is "CPM" with length of 256: x'
    assert fix_stated_lengths("The length of the time series is 512. Spike at 512.", series) == (
        "The length of the time series is 256. Spike at 512."
    )
    assert fix_stated_lengths("a time series of length 256", series) == "a time series of length 256"
    assert fix_stated_lengths("with length of 300", series) == "with length of 300"
    # series of different lengths give no single length to check against
    assert fix_stated_lengths("with length of 512", [[0.0] * 256, [0.0] * 100]) == "with length of 512"

    record = {"input": "with length of 8: <ts><ts/>", "timeseries": [[1.0] * 4], "output": "The length of the time series is 8."}
    user, assistant = chatts_record_to_messages(record)
    assert user["content"][0]["text"] == "with length of 4: " and assistant["content"].endswith("is 4.")
    assert chatts_record_to_messages(record, fix_lengths=False)[1]["content"].endswith("is 8.")


def test_heldout_losses_reward_reading_the_series(tmp_path):
    run(tmp_path, "--max-steps", "150", "--backbone", "full", "--lr", "3e-3", "--ts-lr", "3e-3")
    log = json.loads((tmp_path / "log.json").read_text())
    # the answer (up or down) is only in the series, so blanking them must cost
    assert log["heldout_loss"] < log["heldout_loss_zeroed_series"]


def test_reading_tasks_state_what_the_series_shows():
    import itertools
    import re

    examples = list(itertools.islice(ReadingTasks(seed=3), 300))
    assert [e["messages"][1]["content"] for e in examples] == [
        e["messages"][1]["content"] for e in itertools.islice(ReadingTasks(seed=3), 300)
    ]
    seen = set()
    for example in examples:
        user, assistant = example["messages"]
        series, answer = user["content"][1]["timeseries"], assistant["content"]
        assert f"length {len(series)}:" in user["content"][0]["text"] and np.isfinite(series).all()
        centered = series - np.median(series)
        if answer.startswith("The standardized block means are"):
            # the numbers are what the encoder is given for each patch
            values, mask, _ = Lfm3TsFeatureExtractor(patch_size=8).preprocess([series])
            expected = (values * mask).sum(-1)[0] / mask.sum(-1)[0]
            stated = [float(number) for number in re.findall(r"-?\d+\.\d", answer)]
            np.testing.assert_allclose(stated, expected, atol=0.051)
            assert "-0.0" not in answer and 16 <= len(series) <= 128
            seen.add("means")
        elif answer.startswith("There is a") and "spike at point" in answer:
            point = int(re.search(r"point (\d+)", answer).group(1))
            assert point == int(np.abs(centered).argmax())
            assert ("upward" in answer) == (centered[point] > 0)
            seen.add("spike")
        elif answer.startswith("The level shifts"):
            point = int(re.search(r"point (\d+)", answer).group(1))
            jump = series[point : point + 4].mean() - series[point - 4 : point].mean()
            assert ("upward" in answer) == (jump > 0)
            seen.add("shift")
        elif answer.startswith("The series reaches its"):
            point = int(re.search(r"point (\d+)", answer).group(1))
            assert point == int(series.argmax() if "maximum" in answer else series.argmin())
            seen.add("extremum")
        elif "spikes" in answer or answer.startswith("There is 1 spike"):
            points = [int(p) for p in re.findall(r"\d+", answer.split("at point")[1])] if "at point" in answer else []
            threshold = 4 * np.median(np.abs(centered)) / 0.6745
            assert sorted(np.flatnonzero(centered > threshold).tolist()) == points
            seen.add("count")
        elif answer.startswith("From point 0"):
            first = int(re.search(r"to point (\d+)", answer).group(1))
            assert ("increasing" in answer.split(".")[0]) == (series[first] > series[0])
            seen.add("trend")
        elif answer.startswith("Each fluctuation period"):
            seen.add("period")
    assert seen == {"means", "spike", "shift", "extremum", "count", "trend", "period"}
    share = np.mean([e["messages"][1]["content"].startswith("The standardized") for e in examples])
    assert 0.4 < share < 0.6


def test_warmup_stage_runs_offline(tmp_path, capsys):
    # the real command-line path: the stage builds its own generated stream and held-out set, with no download.
    # The tiny tokenizer maps every number to <unk>, so this cannot show reading; the up/down test above does.
    argv = ["--stage", "warmup", "--output-dir", str(tmp_path), "--max-tokens", "2000", "--grad-accum", "1"]
    argv += ["--max-steps", "4", "--sample-steps", "2", "--num-workers", "0", "--samples", "0", "--cpu"]
    argv += ["--score-examples", "6"]
    train.main(argv, model=tiny_model(), processor=tiny_processor())
    log = json.loads((tmp_path / "log.json").read_text())
    assert np.isfinite([log["heldout_loss"], log["heldout_loss_zeroed_series"]]).all()
    printed = capsys.readouterr().out
    assert "held-out loss on 300 examples" in printed and "reading score on 6 held-out questions" in printed
    assert set(log["reading_scores"]) <= {"block means", "spike", "level shift", "extremum", "period", "trend", "spike count"}


def test_score_reading():
    task, scores = train.score_reading(
        "The standardized block means are: -1.6, 0.1, 0.5.", "The standardized block means are: -1.3, 0.1, 0.8."
    )
    assert task == "block means" and scores == {"count right": 1.0, "mean error": pytest.approx(0.2)}
    assert train.score_reading("The standardized block means are: 1.0, 2.0.", "They are: 1.0.")[1] == {"count right": 0.0}

    task, scores = train.score_reading("There is a downward spike at point 6.", "There is a downward spike at point 3.")
    assert task == "spike" and scores == {
        "answered": 1.0,
        "position error": 3.0,
        "exact right": 0.0,
        "within 2 right": 0.0,
        "direction right": 1.0,
    }
    near = train.score_reading("There is a downward spike at point 6.", "There is a downward spike at point 8.")[1]
    assert near["exact right"] == 0.0 and near["within 2 right"] == 1.0
    assert train.score_reading("There is an upward spike at point 6.", "An upward spike at point 6.")[1]["exact right"] == 1.0
    assert train.score_reading("The level shifts upward at point 40.", "It shifts downward.")[1] == {
        "answered": 0.0,
        "exact right": 0.0,
        "within 2 right": 0.0,
        "direction right": 0.0,
    }
    assert train.score_reading("The series reaches its minimum at point 140.", "at point 14.")[1]["position error"] == 126
    assert train.score_reading(
        "Each fluctuation period is approximately 34 points.", "Each fluctuation period is approximately 17 points."
    ) == ("period", {"answered": 1.0, "relative error": 0.5})

    reference = "From point 0 to point 44, there is an increasing trend. From point 44 to point 92, there is a decreasing trend."
    prediction = "From point 0 to point 29, there is an increasing trend. From point 29 to point 92, there is a decreasing trend."
    assert train.score_reading(reference, prediction) == ("trend", {"directions right": 1.0, "turning point error": 15.0})
    assert train.score_reading(reference, "From point 0 to point 92, there is a decreasing trend.")[1] == {
        "directions right": 0.0
    }

    assert train.score_reading("There are 2 spikes, at points 168, 192.", "There are 2 spikes, at points 168, 190.") == (
        "spike count",
        {"count right": 1.0, "all points right": 0.0, "points right": 0.5, "points within 2 right": 1.0},
    )
    assert train.score_reading("There are 2 spikes, at points 168, 192.", "There is 1 spike, at point 30.")[1] == {
        "count right": 0.0,
        "all points right": 0.0,
        "points right": 0.0,
        "points within 2 right": 0.0,
    }
    assert train.score_reading("There are no spikes.", "There are no spikes.")[1] == {
        "count right": 1.0,
        "all points right": 1.0,
    }
    assert train.score_reading("I did not find any local characteristic fluctuation.", "anything") is None


def test_mixed_stream_restarts_exhausted_datasets_and_skips_empty_ones():
    stream = MixedStream([records(2), []], [0.5, 0.5])
    examples = [example for example, _ in zip(stream, range(10))]
    assert len(examples) == 10
    # a dataloader worker that was handed no data ends instead of raising
    assert list(MixedStream([[], []], [0.5, 0.5])) == []


def test_scheduled_stream_switches_after_the_given_number_of_examples():
    stream = ScheduledStream(itertools.repeat("first"), itertools.repeat("second"), switch_after=3)
    assert list(itertools.islice(stream, 5)) == ["first"] * 3 + ["second"] * 2


def test_a_run_can_change_its_mixture_at_a_step(tmp_path, capsys):
    argv = ["--stage", "locate", "--output-dir", str(tmp_path), "--max-tokens", "400", "--grad-accum", "1", "--max-steps", "3"]
    argv += ["--num-workers", "0", "--samples", "0", "--cpu", "--score-examples", "0"]
    argv += ["--then-mixture", "locate:1.0", "--then-step", "1"]
    train.main(argv, model=tiny_model(), processor=tiny_processor(), dataset=MixedStream([records()], [1.0]))
    assert "mixture from step 1: {'locate': 1.0}" in capsys.readouterr().out
    with pytest.raises(ValueError, match="go together"):
        train.main(argv[:-2], model=tiny_model(), processor=tiny_processor(), dataset=MixedStream([records()], [1.0]))


def test_reconstruction_loss_trains_and_leaves_the_saved_model_unchanged(tmp_path, capsys):
    argv = ["--stage", "locate", "--output-dir", str(tmp_path), "--max-tokens", "400", "--grad-accum", "1", "--max-steps", "4"]
    argv += ["--num-workers", "0", "--samples", "0", "--cpu", "--score-examples", "0", "--sample-steps", "2"]
    argv += ["--reconstruction-weight", "1.0"]
    train.main(argv, model=tiny_model(), processor=tiny_processor(), dataset=MixedStream([records()], [1.0]))
    assert "patch reconstruction error" in capsys.readouterr().out
    saved = AutoModelForCausalLM.from_pretrained(tmp_path / "final").state_dict()
    assert set(saved) == set(tiny_model().state_dict())


def test_masked_reconstruction_trains_the_encoder_on_hidden_patches(tmp_path, capsys):
    argv = ["--stage", "locate", "--output-dir", str(tmp_path), "--max-tokens", "400", "--grad-accum", "1", "--max-steps", "4"]
    argv += ["--num-workers", "0", "--samples", "0", "--cpu", "--score-examples", "0", "--sample-steps", "2"]
    argv += ["--masked-weight", "1.0", "--mask-ratio", "0.5"]
    train.main(argv, model=tiny_model(), processor=tiny_processor(), dataset=MixedStream([records()], [1.0]))
    assert "hidden patch error" in capsys.readouterr().out


def test_verdict_questions_are_answered_from_the_series():
    import re

    tasks = ReadingTasks(seed=5, profile="verdict")
    seen = set()
    for example in itertools.islice(tasks, 600):
        user, answer = example["messages"]
        series = np.asarray(user["content"][1]["timeseries"])
        question, said = user["content"][2]["text"], answer["content"]
        threshold = float(re.search(r"(?:than|above) (\d+(?:\.\d+)?)", question).group(1))
        if "normal" in question or "fluctuation" in question:
            asks_normal = "normal" in question
            normal = said.startswith("Yes") if asks_normal else said.startswith("No")
            # a normal series stays under the threshold even from its lowest to its highest point
            assert (series.max() - series.min() < threshold) == normal
            kind = "noise" if "only noise" in said else "small" if "stays below" in said else "large"
            assert normal == (kind != "large")
            seen.add((asks_normal, kind))
        else:
            # the largest move from one point to the next is what decides
            assert (np.abs(np.diff(series)).max() > threshold) == said.startswith("Yes")
            assert said.startswith("Yes") == ("exceeds" in said)
            events = said.count("There is")
            seen.add(("sudden", events, "exceeds" in said, "stays below" in said))
    assert {key for key in seen if key[0] != "sudden"} == {(a, k) for a in (True, False) for k in ("noise", "small", "large")}
    sudden = {key[1:] for key in seen if key[0] == "sudden"}
    # no event, one of either size, two of the same size, and one of each
    assert sudden >= {(0, False, False), (1, True, False), (1, False, True), (2, True, True), (2, True, False), (2, False, True)}


def test_mixed_stream_feeds_a_multi_worker_dataloader():
    loader = torch.utils.data.DataLoader(
        MixedStream([records(8)], [1.0]),
        batch_size=4,
        num_workers=2,
        collate_fn=Lfm3TsDataCollator(tiny_processor()),
    )
    batches = [batch for batch, _ in zip(loader, range(3))]
    assert all(batch["input_ids"].shape[0] == 4 for batch in batches)


def test_collator_masks_everything_but_the_answer():
    processor = tiny_processor()
    examples = [{"messages": chatts_record_to_messages(record)} for record in records(3)]
    batch = Lfm3TsDataCollator(processor)(examples)

    assert batch["input_values"].shape[0] == 3
    for row in range(3):
        kept = batch["labels"][row][batch["labels"][row] != -100].tolist()
        assert kept == [11 if row % 2 == 0 else 12, processor.tokenizer.eos_token_id]
    # padding and time-series positions are never targets
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    assert (batch["labels"][batch["input_ids"] == TS_TOKEN_ID] == -100).all()
    counts = (batch["input_ids"] == TS_TOKEN_ID).sum(-1)
    assert counts.sum() == sum(-(-len(record["timeseries"][0]) // 4) + 2 for record in records(3))


def test_collator_drops_examples_that_do_not_fit():
    processor = tiny_processor()
    examples = [{"messages": chatts_record_to_messages(record)} for record in records(4)]
    lengths = [len(Lfm3TsDataCollator(processor).encode(example["messages"])["input_ids"]) for example in examples]
    batch = Lfm3TsDataCollator(processor, max_length=min(lengths))(examples)
    assert 0 < batch["input_ids"].shape[0] < 4 or len(set(lengths)) == 1
    with pytest.raises(ValueError, match="fits"):
        Lfm3TsDataCollator(processor, max_length=3)(examples)


def test_token_budget_batches_respect_the_budget_and_keep_every_example():
    collator = Lfm3TsDataCollator(tiny_processor())
    examples = [{"messages": chatts_record_to_messages(record)} for record in records(50)]
    batches = list(TokenBudgetBatches(examples, collator, max_tokens=200, buffer_size=20))

    assert sum(batch["input_ids"].shape[0] for batch in batches) == 50
    assert all(batch["input_ids"].numel() <= 200 for batch in batches)
    assert max(batch["input_ids"].shape[0] for batch in batches) > 1
    # every batch is self-consistent: one feature row per placeholder position
    for batch in batches:
        patches = (batch["input_values_mask"].sum(-1) > 0).sum()
        assert (batch["input_ids"] == TS_TOKEN_ID).sum() == patches + 2 * batch["input_values"].shape[0]
    # an example longer than the budget still gets a batch of its own
    assert all(b["input_ids"].shape[0] == 1 for b in TokenBudgetBatches(examples, collator, max_tokens=1))
    assert TokenBudgetBatches(examples, collator, max_tokens=200, buffer_size=20).examples_per_batch() > 1


def test_answer_only_loss_equals_the_model_loss():
    model, processor = tiny_model().eval(), tiny_processor()
    batch = Lfm3TsDataCollator(processor)([{"messages": chatts_record_to_messages(r)} for r in records(4)])
    expected = model(**batch).loss
    trainer = SimpleNamespace(args=SimpleNamespace(bf16=False))
    loss = train.AnswerOnlyLossTrainer.compute_loss(trainer, model, batch)
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("backbone", ["frozen", "lora"])
def test_probe_step_runs_and_leaves_no_gradients(backbone):
    # the step that sizes batches on a GPU; here only checked to run, since a CPU reports no memory
    model = train.prepare_trainable(tiny_model(), backbone, lora_rank=4)
    train.probe_step(model, batch_size=2, length=16, use_bf16=False)
    assert all(param.grad is None for param in model.parameters())


def test_align_stage_trains_only_the_time_series_modules(tmp_path):
    model = tiny_model()
    before = {name: param.clone() for name, param in model.named_parameters()}
    losses = run(tmp_path, "--max-steps", "40", "--ts-lr", "3e-3", model=model)
    assert losses[-1] < losses[0]

    # the script leaves the model in bfloat16, so compare up to that rounding
    is_close = lambda name, param: torch.allclose(param.float(), before[name], atol=1e-2, rtol=1e-2)
    changed = {name.split(".")[1] for name, param in model.named_parameters() if not is_close(name, param)}
    assert changed == {"ts_tower", "multi_modal_projector"}
    saved = load_file(tmp_path / "trainable-step40.safetensors")
    assert {name.split(".")[1] for name in saved} == {"ts_tower", "multi_modal_projector"}
    # the two most recent small checkpoints, and a final model that loads through the Auto classes
    assert sorted(path.name for path in tmp_path.glob("trainable-step*")) == [
        "trainable-step30.safetensors",
        "trainable-step40.safetensors",
    ]
    assert isinstance(AutoModelForCausalLM.from_pretrained(tmp_path / "final"), Lfm3TsForConditionalGeneration)
    assert isinstance(AutoProcessor.from_pretrained(tmp_path / "final"), Lfm3TsProcessor)


def test_sft_stage_merges_lora_into_a_plain_checkpoint(tmp_path):
    model = tiny_model()
    embeddings = model.get_input_embeddings().weight.clone()
    layer = model.model.language_model.layers[1]
    before = {name: param.clone() for name, param in layer.named_parameters()}
    losses = run(tmp_path, "--max-steps", "40", "--backbone", "lora", "--lr", "3e-3", "--ts-lr", "3e-3", model=model)
    assert losses[-1] < losses[0]

    final = AutoModelForCausalLM.from_pretrained(tmp_path / "final", dtype=torch.float32)
    assert not any("lora" in name for name in final.state_dict())
    merged = dict(final.model.language_model.layers[1].named_parameters())
    assert any(not torch.allclose(merged[name], value, atol=1e-3) for name, value in before.items())
    # LoRA leaves the embeddings alone
    torch.testing.assert_close(final.get_input_embeddings().weight, embeddings, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("backbone", ["frozen", "lora"])
def test_live_samples_print_during_training(tmp_path, capsys, backbone):
    run(tmp_path, "--max-steps", "6", "--sample-steps", "3", "--backbone", backbone)
    printed = capsys.readouterr().out
    assert printed.count("===== step") == 2 and "===== step 6: held-out loss" in printed
    assert printed.count("REFERENCE") == 2 * 5


def test_score_script_scores_a_saved_checkpoint(tmp_path, capsys):
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import score

    run(tmp_path, "--max-steps", "2")
    scores = score.main([str(tmp_path / "final"), "--examples", "8", "--cpu"])
    assert scores and json.loads((tmp_path / "final" / "reading_scores.json").read_text()) == scores
    assert "reading score on 8 held-out questions" in capsys.readouterr().out


def test_init_trainable_resumes_from_a_small_checkpoint(tmp_path):
    run(tmp_path / "first", "--max-steps", "10", "--ts-lr", "3e-3")
    checkpoint = tmp_path / "first" / "trainable-step10.safetensors"
    model = tiny_model()
    run(tmp_path / "second", "--max-steps", "1", "--ts-lr", "0", "--init-trainable", str(checkpoint), model=model)
    for name, value in load_file(checkpoint).items():
        torch.testing.assert_close(model.state_dict()[name].float(), value, atol=1e-2, rtol=1e-2)


def test_locate_profile_asks_mostly_for_positions():
    import itertools
    import re

    with pytest.raises(ValueError, match="profile"):
        ReadingTasks(profile="other")
    examples = list(itertools.islice(ReadingTasks(seed=5, profile="locate"), 600))
    counts = {"value": 0, "spikes": 0, "spike": 0, "none": 0, "peaks": 0, "blocks": 0, "steps": 0}
    for example in examples:
        user, assistant = example["messages"]
        series, question, answer = user["content"][1]["timeseries"], user["content"][2]["text"], assistant["content"]
        numbers = [int(number) for number in re.findall(r"point[s]? ([\d, ]+)", answer) for number in number.split(",") if number.strip()]
        jump = lambda point: min(abs(series[point] - series[point - 1]), abs(series[point] - series[point + 1]))
        typical = np.median(np.abs(np.diff(series)))
        if answer.startswith("The highest point of each block"):
            counts["peaks"] += 1
            numbers = [int(number) for number in re.findall(r"\d+", answer.split(":")[1])]
            assert numbers == [start + int(np.argmax(series[start : start + 8])) for start in range(0, len(series), 8)]
        elif answer.startswith("The blocks with a spike"):
            counts["blocks"] += 1
            blocks = [int(number) for number in re.findall(r"\d+", answer.split(":")[1])]
            assert blocks == sorted(set(blocks)) and 1 <= len(blocks) <= 12
            # each named block holds a point far above both neighbours, and no other block does
            stands_out = [
                point for point in range(1, len(series) - 1)
                if jump(point) > 8 * typical and series[point] > series[point - 1]
            ]
            assert sorted({point // 8 for point in stands_out}) == blocks
        elif answer.startswith("In standardized units"):
            counts["value"] += 1
            values, _, _ = Lfm3TsFeatureExtractor(patch_size=8).preprocess([series])
            point = int(re.search(r"point (\d+)", question).group(1))
            assert numbers == [point] and abs(float(answer.split()[-1].rstrip(".")) - values.reshape(-1)[point]) <= 0.051
        elif "spikes are there" in question:
            counts["none" if answer == "There are no spikes." else "spikes"] += 1
            assert len(numbers) == (0 if answer == "There are no spikes." else int(re.search(r"\d+", answer).group()))
            assert numbers == sorted(numbers) and all(b - a >= 4 for a, b in zip(numbers, numbers[1:]))
            # every stated point stands out from both neighbours, and upward
            assert all(jump(point) > 8 * typical and series[point] > series[point - 1] for point in numbers)
        elif "is the spike" in question:
            counts["spike"] += 1
            if answer.startswith("The spike is in block"):
                counts["steps"] += 1
                block, place = map(int, re.search(r"block (\d+), at place (\d+)", answer).groups())
                assert block * 8 + place == numbers[0] and 0 <= place < 8
            assert jump(numbers[0]) > 8 * typical
            assert ("upward" in answer) == (series[numbers[0]] > series[numbers[0] - 1])
    assert counts["value"] > 15 and counts["spikes"] > 25 and counts["spike"] > 120 and 1 < counts["none"] < 40
    assert counts["peaks"] > 100 and counts["blocks"] > 60 and counts["steps"] > 40
    # spikes no longer sit only at the start of a patch
    assert any(point % 8 for example in examples for point in map(int, re.findall(r"points? (\d+)", example["messages"][1]["content"])))


def test_full_backbone_trains_the_language_model_but_not_its_embeddings():
    model = train.prepare_trainable(tiny_model(), "full", lora_rank=4)
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    assert frozen and all("embed" in name or "lm_head" in name for name in frozen)
    assert all(parameter.requires_grad for name, parameter in model.named_parameters() if ".layers." in name)


def test_locate_stage_trains_the_whole_model(tmp_path, capsys):
    before = tiny_model().state_dict()
    argv = ["--stage", "locate", "--output-dir", str(tmp_path), "--max-tokens", "400", "--grad-accum", "1", "--max-steps", "3"]
    argv += ["--num-workers", "0", "--samples", "0", "--cpu", "--score-examples", "0"]
    train.main(argv, model=tiny_model(), processor=tiny_processor(), dataset=MixedStream([records()], [1.0]))
    printed = capsys.readouterr().out
    assert "backbone=full" in printed and "lr=1e-05" in printed
    after = AutoModelForCausalLM.from_pretrained(tmp_path / "final").state_dict()
    changed = [name for name in before if not torch.equal(before[name], after[name].to(before[name].dtype))]
    assert any(".language_model.layers." in name for name in changed)
    assert not list(tmp_path.glob("trainable-step*"))  # 3 steps: below the 2000 between full checkpoints


def test_adafactor_trains_at_the_two_learning_rates_and_keeps_little_state(tmp_path, capsys):
    model = train.prepare_trainable(tiny_model(), "full", lora_rank=4)
    optimizer = train.build_optimizer(model, 1e-5, 2e-4, 0.0, "adafactor")
    assert sorted(group["lr"] for group in optimizer.param_groups) == [1e-5, 2e-4]
    argv = ["--stage", "locate", "--optimizer", "adafactor", "--output-dir", str(tmp_path), "--max-tokens", "400"]
    argv += ["--grad-accum", "1", "--max-steps", "20", "--logging-steps", "5", "--num-workers", "0", "--samples", "0"]
    argv += ["--cpu", "--score-examples", "0", "--lr", "1e-3", "--ts-lr", "1e-3"]
    train.main(argv, model=tiny_model(), processor=tiny_processor(), dataset=MixedStream([records()], [1.0]))
    losses = [entry["loss"] for entry in json.loads((tmp_path / "log.json").read_text())["steps"] if "loss" in entry]
    assert "optimizer=adafactor" in capsys.readouterr().out and losses[-1] < losses[0]


def test_spike_errors_measure_the_held_out_spike_questions(capsys):
    from lfm3_ts.training import load_chatts_holdout

    examples = load_chatts_holdout({"locate": 1.0}, 300)
    errors = train.spike_errors(tiny_model().eval(), tiny_processor(), examples, limit=3)
    assert len(errors) == 3 and "SPIKES    on 3 held-out questions" in capsys.readouterr().out


def test_export_puts_a_training_checkpoint_into_a_loadable_model(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import export
    from safetensors.torch import save_file

    base = tiny_model()
    base.save_pretrained(tmp_path / "base")
    tiny_processor().save_pretrained(tmp_path / "base")
    trained = train.prepare_trainable(tiny_model(), "full", lora_rank=4)
    with torch.no_grad():
        for parameter in trained.parameters():
            if parameter.requires_grad:
                parameter.add_(0.5)
    save_file(train.trainable_state(trained), tmp_path / "trainable-step5.safetensors")
    export.main([str(tmp_path / "base"), str(tmp_path / "trainable-step5.safetensors"), str(tmp_path / "out")])
    exported = AutoModelForCausalLM.from_pretrained(tmp_path / "out").state_dict()
    for name, value in trained.state_dict().items():
        assert torch.allclose(exported[name].float(), value.float(), atol=1e-2), name
    assert AutoProcessor.from_pretrained(tmp_path / "out") is not None


def test_chat_replay_mixes_plain_conversations_into_training(tmp_path, capsys):
    from lfm3_ts.training import ChatReplay, load_chatts_holdout, load_chatts_stream

    path = tmp_path / "chat.jsonl"
    rows = [{"messages": [{"role": "user", "content": f"which way {i}"}, {"role": "assistant", "content": "up"}]} for i in range(80)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    replay = ChatReplay(path, seed=0)
    seen = [example["messages"][0]["content"] for example in __import__("itertools").islice(replay, 60)]
    # the first 50 are held out; the rest come back shuffled, each once per pass
    assert len(set(seen[:30])) == 30 and not {f"which way {i}" for i in range(50)} & set(seen)
    held = load_chatts_holdout({"chat": 1.0}, 20, chat_path=str(path))
    assert [example["messages"][0]["content"] for example in held] == [f"which way {i}" for i in range(20)]
    with pytest.raises(ValueError, match="chat-data"):
        load_chatts_stream({"chat": 1.0})

    # text-only conversations train next to series, and their samples print
    stream = load_chatts_stream({"chat": 0.5, "reading": 0.5}, chat_path=str(path))
    long = load_chatts_stream({"locate": 1.0}, reading_max_length=2048)
    lengths = [len(part["timeseries"]) for example in itertools.islice(long, 40) for part in example["messages"][0]["content"] if part["type"] == "timeseries"]
    assert max(lengths) > 512
    argv = ["--stage", "dev", "--output-dir", str(tmp_path / "run"), "--max-tokens", "600", "--grad-accum", "1"]
    argv += ["--max-steps", "3", "--num-workers", "0", "--samples", "2", "--cpu", "--score-examples", "0"]
    train.main(argv, model=tiny_model(), processor=tiny_processor(), dataset=stream)
    assert "QUESTION" in capsys.readouterr().out


def test_make_chat_replay_keeps_only_finished_answers():
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import make_chat_replay

    model, tokenizer = tiny_model().eval(), tiny_processor().tokenizer
    tokenizer.chat_template = CHAT_TEMPLATE  # the LFM3 tokenizer carries its own
    prompts = ["which way", "way which way", "which"]
    everything = list(make_chat_replay.answer_all(model, tokenizer, prompts, batch_size=2, max_new_tokens=5, keep_unfinished=True))
    finished = list(make_chat_replay.answer_all(model, tokenizer, prompts, batch_size=2, max_new_tokens=5))
    assert {prompt for prompt, _ in everything} <= set(prompts) and len(finished) <= len(everything)
    assert all(isinstance(answer, str) and answer for _, answer in everything)


def test_holdout_of_two_generated_profiles_has_no_duplicates():
    from lfm3_ts.training import load_chatts_holdout

    mixed = load_chatts_holdout({"reading": 0.5, "locate": 0.5}, 60)
    single = load_chatts_holdout({"reading": 1.0}, 60)
    answers = [example["messages"][1]["content"] + str(example["messages"][0]["content"][1]["timeseries"][:2]) for example in mixed]
    assert len(mixed) == 60 and len(set(answers)) == 60
    assert [e["messages"][1]["content"] for e in mixed] == [e["messages"][1]["content"] for e in single]


def test_exact_profile_asks_for_the_place_inside_each_block():
    import numpy as np

    from lfm3_ts.training import ReadingTasks

    tasks = ReadingTasks(profile="exact")
    assert abs(sum(tasks.EXACT.values()) - 1) < 1e-9
    rng = np.random.default_rng(0)
    for _ in range(20):
        series, question, answer = tasks.block_places(rng, int(rng.integers(16, 129)))
        places = [int(place) for place in answer.rsplit(": ", 1)[1].rstrip(".").split(", ")]
        assert places == [int(series[start : start + 8].argmax()) for start in range(0, len(series), 8)]
        assert all(0 <= place < 8 for place in places) and "Places are numbered from 0" in question
    names = {tasks.sample(rng)["messages"][0]["content"][2]["text"][:24] for _ in range(300)}
    assert any("at which place inside" in tasks.sample(rng)["messages"][0]["content"][2]["text"] for _ in range(100))
    assert len(names) >= 8


def test_exact_mixture_shares_the_held_out_questions():
    from lfm3_ts.training import load_chatts_holdout

    exact = load_chatts_holdout({"exact": 1.0}, 40)
    reading = load_chatts_holdout({"reading": 1.0}, 40)
    assert [e["messages"][1]["content"] for e in exact] == [e["messages"][1]["content"] for e in reading]


def test_real_series_are_cut_measured_described_and_trained_on(tmp_path, capsys):
    import importlib.util

    from lfm3_ts.training import RealSeries, load_chatts_holdout, load_chatts_stream

    spec = importlib.util.spec_from_file_location("make_real_data", Path(train.__file__).with_name("make_real_data.py"))
    real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real)

    # the measurements of a spike, and of a step that does not come back
    series = np.full(300, 10.0) + np.random.default_rng(0).normal(0, 0.1, 300)
    series[100:103] = 50.0
    facts = real.measure(series, (100, 102))
    assert facts["direction"] == "upward" and facts["extreme_value"] == "50" and facts["returns"] and facts["visible"]
    step = np.r_[np.full(150, 10.0), np.full(150, 2.0)] + np.random.default_rng(1).normal(0, 0.1, 300)
    facts_step = real.measure(step, (150, 160))
    assert facts_step["direction"] == "downward" and not facts_step["returns"]
    assert not real.measure(np.random.default_rng(2).normal(0, 1, 300), (100, 110))["visible"]
    # a written description is kept only if each of its numbers was measured
    assert real.numbers_allowed(f"A spike to 50 from about {facts['level_before']} at point {facts['extreme_point']}.", facts)
    assert not real.numbers_allowed("A spike to 51.", facts) and not real.numbers_allowed("A spike.", facts)
    assert real.numbers_allowed(real.plain_description(facts), facts)
    assert "stays near" in real.plain_description(facts_step)

    # files of two trained sources and one held-out source; of each trained source one file goes to the test
    rng = np.random.default_rng(3)
    dataset = tmp_path / "TSB-AD-U"
    dataset.mkdir()
    for source, files in (("WSD", 5), ("SMD", 5), ("YAHOO", 2)):
        for index in range(files):
            values, labels = rng.normal(100, 1, 6000), np.zeros(6000, dtype=int)
            for place in range(500, 5500, 500):
                # one labelled stretch in five has nothing measurable in it
                values[place : place + 3] += 0 if place == 2500 else 40
                labels[place : place + 3] = 1
            rows = "\n".join(f"{value},{label}" for value, label in zip(values, labels))
            (dataset / f"{index:03d}_{source}_id_{index}_WebService_tr_500_1st_500.csv").write_text("Data,Label\n" + rows + "\n")
    splits = real.split_files(str(dataset))
    assert len(splits["train"]) == 8 and len(splits["test, seen sources"]) == 2 and len(splits["test, unseen sources"]) == 2
    assert not {path.name for path in splits["train"]} & {path.name for name in splits if name != "train" for path in splits[name]}
    out = tmp_path / "real"
    real.main(["windows", "--dataset", str(dataset), "--out", str(out), "--per-file", "6", "--sheet-size", "30"])
    windows = [json.loads(line) for line in (out / "train_windows.jsonl").read_text().splitlines()]
    assert {window["source"] for window in windows} == {"WSD", "SMD"}
    subtle = [window for window in windows if window["segment"] and not window["facts"]["visible"]]
    assert subtle and all(window["id"].startswith("train-subtle-") for window in subtle)
    assert [window["id"] for window in windows[-len(subtle) :]] == [window["id"] for window in subtle]
    with_anomaly = [window for window in windows if window["segment"] and window["facts"]["visible"]]
    assert with_anomaly and all(window["facts"]["extreme_value"] == real.number(max(window["series"])) for window in with_anomaly)
    questions = [json.loads(line) for line in (out / "test_questions.jsonl").read_text().splitlines()]
    train_files = {window["file"] for window in windows}
    assert questions and not {question["file"] for question in questions} & train_files
    assert {question["group"].split(",")[0] for question in questions} == {"seen sources", "unseen sources"}
    assert (out / "sheet_00.txt").read_text().startswith("### train-")

    # a description with an unmeasured number falls back to the plain sentence
    first, second = with_anomaly[0], with_anomaly[1]
    described = [
        {"id": first["id"], "description": f"A sharp spike up to {first['facts']['extreme_value']}."},
        {"id": second["id"], "description": "A spike up to 123456."},
    ]
    (tmp_path / "described.jsonl").write_text("".join(json.dumps(row) + "\n" for row in described))
    path = tmp_path / "real.jsonl"
    real.main(["build", "--windows", str(out), "--descriptions", str(tmp_path / "described.jsonl"), "--out", str(path)])
    printed = capsys.readouterr().out
    assert f"1 written descriptions kept, {len(with_anomaly) - 1} plain ones, {len(subtle)} with the place only" in printed
    answers = [json.loads(line)["messages"][1]["content"] for line in path.read_text().splitlines()]
    # a stretch with nothing measurable is taught with its place and no number but the two points
    place_only = [answer for answer in answers if answer.endswith(real.SUBTLE_ANSWERS)]
    assert len(place_only) == len(subtle) and all(sum(word.strip(".,").isdigit() for word in answer.split()) == 2 for answer in place_only)
    assert "train-subtle" not in (out / "sheet_00.txt").read_text()
    assert any("A sharp spike up to" in answer for answer in answers) and not any("123456" in answer for answer in answers)
    assert all(answer.startswith(("Yes, around points ", "No, ")) for answer in answers)

    # the file trains next to the generated questions
    example = next(iter(RealSeries(path, seed=0)))
    assert any(isinstance(part.get("timeseries"), np.ndarray) for part in example["messages"][0]["content"])
    with pytest.raises(ValueError, match="real-data"):
        load_chatts_stream({"real": 1.0})
    assert load_chatts_holdout({"real": 1.0}, 5, real_path=str(path))
    stream = load_chatts_stream({"real": 0.5, "reading": 0.5}, real_path=str(path))
    argv = ["--stage", "dev", "--output-dir", str(tmp_path / "run"), "--max-tokens", "900", "--grad-accum", "1"]
    argv += ["--max-steps", "3", "--num-workers", "0", "--samples", "2", "--cpu", "--score-examples", "0"]
    train.main(argv, model=tiny_model(), processor=tiny_processor(), dataset=stream)


def test_the_value_ruler_writes_the_levels_and_is_kept_by_a_checkpoint(tmp_path):
    processor = tiny_processor.__wrapped__() if hasattr(tiny_processor, "__wrapped__") else tiny_processor()
    series = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0])
    values, _, stats = processor.feature_extractor.preprocess([series])
    assert "levels" not in processor.format_series(stats[0])
    processor.value_ruler = True
    text = processor.format_series(stats[0])
    offset, scale = stats[0].offset, stats[0].scale
    # the values standing at -2 and +1 in the units the encoder sees, after the statistics that were there before
    assert f"last: {series[-1]:.4f} | levels -2: {offset - 2 * scale:.4f}, -1: {offset - scale:.4f}, +0: {offset:.4f}, +1: {offset + scale:.4f}" in text
    assert text.count("]") == 1 and text.endswith("<|ts|>")
    # the series itself is fed as before
    np.testing.assert_array_equal(values, processor(text="<|ts|>", timeseries=[series])["input_values"])
    processor.save_pretrained(tmp_path)
    assert type(processor).from_pretrained(tmp_path).value_ruler is True


def test_injected_events_are_an_anomaly_only_when_the_rest_of_the_series_does_not_do_the_same(tmp_path):
    from lfm3_ts.training import InjectedAnomalies, load_chatts_holdout, load_chatts_stream

    rng = np.random.default_rng(0)
    records = []
    for index in range(80):
        series = (50 + 5 * np.sin(np.arange(384) / 6) + rng.normal(size=384)).tolist()
        content = [{"type": "text", "text": "This is a time series of length 384: "}, {"type": "timeseries", "timeseries": series}, {"type": "text", "text": ". Does this time series contain an anomaly?"}]
        answer = "No, there is not an anomaly in this time series." if index % 2 else "Yes, around points 3 to 9."
        records.append({"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": answer}]})
    path = tmp_path / "real.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    stream = InjectedAnomalies(str(path))
    assert all(record["messages"][-1]["content"].startswith("No") for record in stream.records)
    cases, kinds, turns = {}, set(), set()
    for example in itertools.islice(stream, 600):
        change, messages = example["change"], example["messages"]
        series, verdict = messages[0]["content"][1]["timeseries"], messages[-1]["content"]
        cases[change["case"]] = cases.get(change["case"], 0) + 1
        # one event, or one that does not fit the others, is an anomaly; the same event recurring is not
        assert verdict.startswith("Yes" if change["case"] in ("once", "odd") else "No"), (change, verdict)
        if change["case"] == "untouched":
            assert len(messages) == 2
            continue
        kinds.add(change["kind"])
        turns.add(change["turns"])
        assert len(messages) == 2 * change["turns"]
        if change["stretch"]:
            first, last = map(int, re.match(r"Yes, around points (\d+) to (\d+)\.", verdict).groups())
            assert [first, last] == change["stretch"] and 0 <= first <= last < len(series)
        if change["case"] == "recurring":
            assert len(change["points"]) >= 3 and len(set(np.diff(change["points"]))) == 1
        if change["turns"] == 2:
            # the description gives no verdict, and the question of the anomaly comes after it
            assert not messages[1]["content"].startswith(("Yes", "No")) and messages[2]["content"][0]["text"] == InjectedAnomalies.JUDGE
        said = " ".join(message["content"] for message in messages if message["role"] == "assistant")
        for value, point in re.findall(r"(-?\d+(?:\.\d+)?) at point (\d+)", said):
            assert abs(float(value) - series[int(point)]) <= 0.06 * np.std(series) + 1e-9, said
    assert set(cases) == {"once", "recurring", "odd", "untouched"} and kinds == set(InjectedAnomalies.KINDS) and turns == {1, 2}
    assert 0.4 < (cases["once"] + cases["odd"]) / 600 < 0.7
    # a mixture can name it, with the same file as the questions on real series
    mixed = load_chatts_stream({"injected": 0.5, "real": 0.5}, real_path=str(path))
    assert sum(example["messages"][-1]["content"].startswith("Yes") for example in itertools.islice(mixed, 40)) > 10
    assert load_chatts_holdout({"injected": 0.5, "real": 0.5}, real_path=str(path))


def test_a_described_then_judged_example_trains_on_both_answers():
    from lfm3_ts.training import InjectedAnomalies

    rng = np.random.default_rng(1)
    series = 50 + 5 * np.sin(np.arange(384) / 6) + rng.normal(size=384)
    content = [{"type": "text", "text": "This is a time series of length 384: "}, {"type": "timeseries", "timeseries": series}, {"type": "text", "text": ". Is there an anomaly?"}]
    messages = [{"role": "user", "content": content}, {"role": "assistant", "content": "No."}]
    example = None
    while example is None or example["change"]["turns"] != 2:
        example = InjectedAnomalies.inject(rng, messages, case="recurring")
    processor = tiny_processor()
    batch = Lfm3TsDataCollator(processor)([example])
    kept = (batch["labels"][0] >= 0).int()
    # two stretches of the labels are trained on: the description, and the verdict after the second question
    assert int((kept.diff() == 1).sum()) + int(kept[0]) == 2
