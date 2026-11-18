import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, PreTrainedTokenizerFast

from lfm3_ts import (
    Lfm3TsConfig,
    Lfm3TsFeatureExtractor,
    Lfm3TsForConditionalGeneration,
    Lfm3TsProcessor,
    get_ts_patch_counts,
)


TS_TOKEN_ID = 10
PATCH_SIZE = 4


def tiny_config() -> Lfm3TsConfig:
    return Lfm3TsConfig(
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
        ts_config={"patch_size": PATCH_SIZE, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2},
        ts_token_id=TS_TOKEN_ID,
        projector_hidden_size=24,
    )


@pytest.fixture
def model() -> Lfm3TsForConditionalGeneration:
    torch.manual_seed(0)
    return Lfm3TsForConditionalGeneration(tiny_config()).eval()


@pytest.fixture
def processor() -> Lfm3TsProcessor:
    words = ["<pad>", "<bos>", "<eos>", "<unk>", "what", "is", "this", "[", "]", ":"]
    vocab = {word: index for index, word in enumerate(words)}
    vocab["<|ts|>"] = TS_TOKEN_ID
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        extra_special_tokens={"ts_token": "<|ts|>"},
    )
    return Lfm3TsProcessor(Lfm3TsFeatureExtractor(patch_size=PATCH_SIZE), tokenizer)


def make_inputs(lengths, prefix=(4, 5), suffix=(6,), seed=0):
    """One prompt holding one series per entry of `lengths`."""
    rng = np.random.default_rng(seed)
    extractor = Lfm3TsFeatureExtractor(patch_size=PATCH_SIZE)
    features = extractor([rng.normal(size=length) for length in lengths], return_tensors="pt")
    ids = list(prefix)
    for length in lengths:
        ids += [TS_TOKEN_ID] * (extractor.get_num_patches(length) + 2)
    ids += list(suffix)
    input_ids = torch.tensor([ids])
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), **features}


# --- configuration -----------------------------------------------------------------------------------------------


def test_config_round_trip(tmp_path):
    config = tiny_config()
    config.save_pretrained(tmp_path)
    loaded = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, Lfm3TsConfig)
    expected = {key: value for key, value in config.to_dict().items() if key != "_name_or_path"}
    assert {key: value for key, value in loaded.to_dict().items() if key != "_name_or_path"} == expected
    assert loaded.text_config.model_type == "lfm3"
    assert loaded.ts_config.patch_size == PATCH_SIZE


def test_config_defaults_match_lfm3():
    config = Lfm3TsConfig()
    assert config.text_config.hidden_size == 2048
    assert config.ts_token_id < config.text_config.vocab_size


def test_config_rejects_placeholder_outside_vocabulary():
    with pytest.raises(ValueError, match="ts_token_id"):
        Lfm3TsConfig(text_config={"vocab_size": 64, "pad_token_id": 0}, ts_token_id=64)


# --- feature extractor -------------------------------------------------------------------------------------------


def test_feature_extractor_shapes_and_padding():
    extractor = Lfm3TsFeatureExtractor(patch_size=4)
    values, mask, stats = extractor.preprocess([np.arange(10.0), np.arange(3.0)])
    assert values.shape == mask.shape == (2, 3, 4)
    assert [stat.num_patches for stat in stats] == [3, 1]
    assert [stat.length for stat in stats] == [10, 3]
    assert mask[0].reshape(-1).tolist() == [1] * 10 + [0] * 2
    assert mask[1].reshape(-1).tolist() == [1] * 3 + [0] * 9
    assert np.all(values[mask == 0] == 0)


def test_feature_extractor_normalization_is_invertible():
    series = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    extractor = Lfm3TsFeatureExtractor(patch_size=4)
    values, mask, (stat,) = extractor.preprocess([series])
    restored = values.reshape(-1)[: stat.length] * stat.scale + stat.offset
    np.testing.assert_allclose(restored, series, rtol=1e-5)
    assert stat.offset == pytest.approx(series.mean())
    assert stat.scale == pytest.approx(series.std())


def test_feature_extractor_missing_values():
    series = np.array([1.0, np.nan, 3.0, 5.0, np.nan, np.nan])
    values, mask, (stat,) = Lfm3TsFeatureExtractor(patch_size=4).preprocess([series])
    # trailing missing values are dropped, inner ones masked
    assert stat.length == 4 and stat.num_patches == 1
    assert mask.reshape(-1).tolist() == [1, 0, 1, 1]
    assert stat.offset == pytest.approx(3.0)
    assert np.isfinite(values).all()


def test_feature_extractor_constant_series_is_not_scaled():
    values, _, (stat,) = Lfm3TsFeatureExtractor(patch_size=4).preprocess([np.full(8, 1e6)])
    assert stat.scale == 1.0 and stat.offset == pytest.approx(1e6)
    assert np.all(values == 0)


@pytest.mark.parametrize("series", [np.zeros((2, 3)), np.array([np.nan, np.nan]), np.array([])])
def test_feature_extractor_rejects_bad_series(series):
    with pytest.raises(ValueError):
        Lfm3TsFeatureExtractor().preprocess([series])


def test_patch_counts_agree_between_extractor_and_model():
    extractor = Lfm3TsFeatureExtractor(patch_size=4)
    series = [np.arange(1.0, 10.0), np.array([1.0, 2.0, np.nan, np.nan, np.nan, np.nan, 3.0]), np.arange(1.0, 5.0)]
    _, mask, stats = extractor.preprocess(series)
    assert get_ts_patch_counts(torch.from_numpy(mask)).tolist() == [stat.num_patches for stat in stats]


# --- model -------------------------------------------------------------------------------------------------------


def test_forward_shapes_and_loss(model):
    inputs = make_inputs([10, 3])
    outputs = model(**inputs, labels=inputs["input_ids"])
    sequence_length = inputs["input_ids"].shape[1]
    assert outputs.logits.shape == (1, sequence_length, 64)
    assert torch.isfinite(outputs.loss)
    # 3 + 1 patches, and a start and an end embedding per series
    assert outputs.ts_hidden_states.shape == (3 + 1 + 4, 32)


def test_text_only_forward_matches_language_model(model):
    input_ids = torch.tensor([[4, 5, 6, 7]])
    logits = model(input_ids=input_ids).logits
    hidden = model.model.language_model(input_ids=input_ids).last_hidden_state
    torch.testing.assert_close(logits, model.lm_head(hidden))


def test_series_values_reach_the_logits(model):
    first, second = make_inputs([10], seed=0), make_inputs([10], seed=1)
    assert not torch.allclose(model(**first).logits[:, -1], model(**second).logits[:, -1])


def test_position_reaches_the_encoder(model):
    # the same patch of values at two places in a series must not look the same to the language model
    extractor = Lfm3TsFeatureExtractor(patch_size=PATCH_SIZE)
    features = extractor([np.tile([0.0, 1.0, 0.0, -1.0], 4)], return_tensors="pt")
    patches = model.get_ts_features(**features).pooler_output[0][1:-1]
    assert not torch.allclose(patches[0], patches[2])
    position_features = model.model.ts_tower.get_position_features(features["input_values_mask"])
    assert position_features.shape == (1, 4, PATCH_SIZE * 16)

    features = extractor([np.arange(6.0)], return_tensors="pt")
    position_features = model.model.ts_tower.get_position_features(features["input_values_mask"])
    assert (position_features.view(1, 2, PATCH_SIZE, 16)[0, 1, 2:] == 0).all()


def attention_model() -> Lfm3TsForConditionalGeneration:
    config = tiny_config()
    config.ts_config.num_attention_layers = 2
    config.ts_config.num_attention_heads = 2
    torch.manual_seed(0)
    return Lfm3TsForConditionalGeneration(config).eval()


def test_attention_layers_let_patches_read_their_series():
    extractor = Lfm3TsFeatureExtractor(patch_size=PATCH_SIZE)
    first = np.arange(12.0)
    second = first.copy()
    second[8:] = [5.0, -3.0, 7.0, 0.0]  # only the last patch differs
    patches = lambda model, series: model.get_ts_features(**extractor([series], return_tensors="pt")).pooler_output[0]

    # the header statistics change the normalization, so compare on shared statistics
    def first_patch(model, series):
        features = extractor([series], return_tensors="pt")
        features["input_values"][0, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4])
        return model.get_ts_features(**features).pooler_output[0][1]

    independent = Lfm3TsForConditionalGeneration(tiny_config()).eval()
    torch.testing.assert_close(first_patch(independent, first), first_patch(independent, second))
    contextual = attention_model()
    assert not torch.allclose(first_patch(contextual, first), first_patch(contextual, second))


def test_attention_ignores_the_padding_of_longer_series_in_the_batch():
    model = attention_model()
    extractor = Lfm3TsFeatureExtractor(patch_size=PATCH_SIZE)
    short, long = np.sin(np.arange(6.0)), np.cos(np.arange(20.0))
    alone = model.get_ts_features(**extractor([short], return_tensors="pt")).pooler_output[0]
    batched = model.get_ts_features(**extractor([short, long], return_tensors="pt")).pooler_output[0]
    torch.testing.assert_close(alone, batched, atol=1e-5, rtol=1e-4)


def test_attention_model_round_trips(tmp_path):
    model = attention_model()
    inputs = make_inputs([10, 3])
    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    assert loaded.config.ts_config.num_attention_layers == 2
    torch.testing.assert_close(loaded(**inputs).logits, model(**inputs).logits)


def test_time_series_features_start_at_the_configured_scale():
    config = tiny_config()
    config.projector_output_scale = 0.05
    model = Lfm3TsForConditionalGeneration(config).eval()
    features = Lfm3TsFeatureExtractor(patch_size=PATCH_SIZE)([np.sin(np.arange(40.0))], return_tensors="pt")
    tokens = model.get_ts_features(**features).pooler_output[0]
    # patches and both boundary embeddings
    assert tokens[1:-1].pow(2).mean().sqrt().item() == pytest.approx(0.05, rel=0.05)
    assert tokens[[0, -1]].pow(2).mean().sqrt().item() == pytest.approx(0.05, rel=0.5)


def test_from_text_pretrained_matches_the_text_model_and_its_embedding_scale(tmp_path):
    from transformers import Lfm3ForCausalLM

    torch.manual_seed(0)
    text_model = Lfm3ForCausalLM(tiny_config().text_config).eval()
    with torch.no_grad():
        text_model.get_input_embeddings().weight.mul_(3.0)
    text_model.save_pretrained(tmp_path)

    model = Lfm3TsForConditionalGeneration.from_text_pretrained(
        tmp_path, ts_config=tiny_config().ts_config, ts_token_id=TS_TOKEN_ID, projector_hidden_size=24
    ).eval()
    input_ids = torch.tensor([[4, 5, 6, 7]])
    torch.testing.assert_close(model(input_ids=input_ids).logits, text_model(input_ids=input_ids).logits)

    embedding_scale = text_model.get_input_embeddings().weight.pow(2).mean().sqrt().item()
    assert model.config.projector_output_scale == pytest.approx(embedding_scale)
    features = Lfm3TsFeatureExtractor(patch_size=PATCH_SIZE)([np.sin(np.arange(40.0))], return_tensors="pt")
    patches = model.get_ts_features(**features).pooler_output[0][1:-1]
    assert patches.pow(2).mean().sqrt().item() == pytest.approx(embedding_scale, rel=0.05)


def test_padding_values_do_not_reach_the_logits(model):
    inputs = make_inputs([10, 3])
    reference = model(**inputs).logits
    # patches past the end of a series are padding, whatever they hold
    counts = get_ts_patch_counts(inputs["input_values_mask"])
    is_padding = torch.arange(inputs["input_values"].shape[1])[None, :, None] >= counts[:, None, None]
    assert is_padding.any()
    inputs["input_values"] = torch.where(is_padding, 100.0, inputs["input_values"])
    torch.testing.assert_close(model(**inputs).logits, reference)


def test_placeholder_count_mismatch_raises(model):
    inputs = make_inputs([10])
    inputs["input_ids"] = inputs["input_ids"][:, 1:]
    inputs["attention_mask"] = inputs["attention_mask"][:, 1:]
    inputs["input_ids"][0, 1] = 4
    with pytest.raises(ValueError, match="do not match"):
        model(**inputs)


def test_inputs_embeds_forward_matches_input_ids(model):
    inputs = make_inputs([10])
    reference = model(**inputs).logits
    inputs_embeds = model.get_input_embeddings()(inputs.pop("input_ids"))
    torch.testing.assert_close(model(inputs_embeds=inputs_embeds, **inputs).logits, reference)


def test_generate_with_and_without_cache_agree(model):
    inputs = make_inputs([10, 3])
    kwargs = {"max_new_tokens": 6, "do_sample": False, "min_new_tokens": 6}
    cached = model.generate(**inputs, use_cache=True, **kwargs)
    uncached = model.generate(**inputs, use_cache=False, **kwargs)
    assert cached.shape[1] == inputs["input_ids"].shape[1] + 6
    assert cached.tolist() == uncached.tolist()


def test_batched_forward_matches_single(model):
    first, second = make_inputs([8], seed=0), make_inputs([8], seed=1)
    batch = {key: torch.cat([first[key], second[key]]) for key in first}
    logits = model(**batch).logits
    torch.testing.assert_close(logits[:1], model(**first).logits, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(logits[1:], model(**second).logits, atol=1e-5, rtol=1e-4)


def test_frozen_language_model_trains_only_time_series_modules(model):
    model.model.language_model.requires_grad_(False)
    inputs = make_inputs([10])
    model(**inputs, labels=inputs["input_ids"]).loss.backward()
    trained = {name.split(".")[1] for name, param in model.named_parameters() if param.grad is not None}
    assert trained == {"ts_tower", "multi_modal_projector"}
    assert model.model.multi_modal_projector.boundary_embeddings.grad.abs().sum() > 0


def test_save_and_load_round_trip(model, tmp_path):
    inputs = make_inputs([10])
    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    assert isinstance(loaded, Lfm3TsForConditionalGeneration)
    torch.testing.assert_close(loaded(**inputs).logits, model(**inputs).logits)
    assert loaded.lm_head.weight is loaded.get_input_embeddings().weight


# --- processor ---------------------------------------------------------------------------------------------------


def test_processor_expands_placeholders(processor, model):
    series = [np.arange(10.0), np.arange(3.0)]
    inputs = processor(text="what is this <|ts|> <|ts|>", timeseries=series, return_tensors="pt")
    assert (inputs["input_ids"] == TS_TOKEN_ID).sum() == (3 + 2) + (1 + 2)
    assert inputs["input_values"].shape == (2, 3, PATCH_SIZE)
    assert torch.isfinite(model(**inputs).logits).all()


def test_processor_writes_statistics_into_the_prompt(processor):
    stats = processor.feature_extractor.preprocess([np.array([10.0, 12.0, 14.0, 16.0, 18.0])])[2][0]
    block = processor.format_series(stats)
    assert block.startswith(
        "[offset: 14.0000 | scale: 2.8284 | length: 5 | min: 10.0000 | max: 18.0000 | first: 10.0000 | last: 18.0000]"
    )
    assert block.count("<|ts|>") == 2 + 2


def test_position_markers_sit_before_the_patch_they_name(processor, model, tmp_path):
    marked = Lfm3TsProcessor(processor.feature_extractor, processor.tokenizer, position_marker_every=8)
    stats = marked.feature_extractor.preprocess([np.arange(22.0)])[2][0]
    block = marked.format_series(stats)
    series = block[block.index("]") + 1 :]
    # start, patches at 0 and 4, marker, patches at 8 and 12, marker, patches at 16 and 20, end
    assert series == "<|ts|>" * 3 + "<8>" + "<|ts|>" * 2 + "<16>" + "<|ts|>" * 3
    assert block.count("<|ts|>") == processor.format_series(stats).count("<|ts|>")

    inputs = marked(text="what <|ts|>", timeseries=np.arange(22.0), return_tensors="pt")
    assert torch.isfinite(model(**inputs).logits).all()
    marked.save_pretrained(tmp_path)
    assert Lfm3TsProcessor.from_pretrained(tmp_path).position_marker_every == 8
    with pytest.raises(ValueError, match="multiple of the patch size"):
        Lfm3TsProcessor(processor.feature_extractor, processor.tokenizer, position_marker_every=6)


def test_differences_reach_the_encoder_across_patch_borders():
    config = tiny_config()
    config.ts_config.use_differences = True
    torch.manual_seed(0)
    model = Lfm3TsForConditionalGeneration(config).eval()
    tower = model.model.ts_tower
    assert tower.patch_embed.in_features == (3 + 16) * PATCH_SIZE

    captured = {}
    tower.patch_embed.register_forward_hook(lambda module, args, output: captured.update(patches=args[0]))
    values = torch.tensor([[[0.0, 1.0, 3.0, 6.0], [10.0, 10.0, 0.0, 0.0]]])
    mask = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]])
    tower(values, mask)
    differences = captured["patches"][0, :, 2 * PATCH_SIZE : 3 * PATCH_SIZE]
    assert differences.tolist() == [[0.0, 1.0, 2.0, 3.0], [4.0, 0.0, 0.0, 0.0]]


def test_processor_multivariate_item_takes_one_placeholder(processor):
    inputs = processor(text="what <|ts|>", timeseries=np.zeros((3, 8)) + np.arange(8.0), return_tensors="pt")
    assert inputs["input_values"].shape == (3, 2, PATCH_SIZE)
    assert (inputs["input_ids"] == TS_TOKEN_ID).sum() == 3 * (2 + 2)


def test_processor_batches_prompts_with_their_series(processor, model):
    inputs = processor(
        text=["what <|ts|>", "what is this <|ts|> <|ts|>"],
        timeseries=[[np.arange(8.0)], [np.arange(4.0), np.arange(6.0)]],
        padding=True,
        return_tensors="pt",
    )
    assert inputs["input_ids"].shape[0] == 2
    assert inputs["input_values"].shape[0] == 3
    assert torch.isfinite(model(**inputs).logits).all()


def test_processor_rejects_placeholder_mismatch(processor):
    with pytest.raises(ValueError, match="1 <|ts|> tokens in the text but received 2"):
        processor(text="what <|ts|>", timeseries=[np.arange(4.0), np.arange(4.0)])


def test_processor_text_only(processor):
    inputs = processor(text="what is this", return_tensors="pt")
    assert "input_values" not in inputs
    assert inputs["input_ids"].tolist() == [[4, 5, 6]]


def test_processor_save_and_load_round_trip(processor, tmp_path):
    processor.save_pretrained(tmp_path)
    loaded = Lfm3TsProcessor.from_pretrained(tmp_path)
    assert loaded.feature_extractor.patch_size == PATCH_SIZE
    assert loaded.ts_token_id == TS_TOKEN_ID
    kwargs = {"text": "what <|ts|>", "timeseries": np.arange(8.0), "return_tensors": "pt"}
    assert loaded(**kwargs)["input_ids"].tolist() == processor(**kwargs)["input_ids"].tolist()


# --- real LFM3 tokenizer (network) -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lfm3_processor() -> Lfm3TsProcessor:
    return Lfm3TsProcessor.from_text_pretrained("LiquidAI/LFM3-1.7B")


@pytest.mark.slow
def test_lfm3_tokenizer_has_a_free_row_for_the_placeholder(lfm3_processor):
    # the default `ts_token_id` is the first unused row of the LFM3 embedding matrix
    assert lfm3_processor.ts_token_id == Lfm3TsConfig().ts_token_id == 125_033


@pytest.mark.slow
def test_lfm3_chat_template_renders_time_series(lfm3_processor):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "timeseries", "timeseries": np.sin(np.arange(64) / 4)},
                {"type": "text", "text": "Where is the anomaly?"},
            ],
        }
    ]
    prompt = lfm3_processor.apply_chat_template(conversation, add_generation_prompt=True)
    assert prompt.count("<|ts|>") == 1 and "Where is the anomaly?" in prompt

    inputs = lfm3_processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )
    assert (inputs["input_ids"] == 125_033).sum() == 64 // 8 + 2
    assert inputs["input_values"].shape == (1, 8, 8)
    # the template writes the only BOS
    assert (inputs["input_ids"] == lfm3_processor.tokenizer.bos_token_id).sum() == 1
    decoded = lfm3_processor.tokenizer.decode(inputs["input_ids"][0])
    assert "[offset:" in decoded and "| length: 64 |" in decoded and "| last:" in decoded


@pytest.mark.slow
def test_lfm3_processor_round_trips_through_auto_processor(lfm3_processor, tmp_path):
    lfm3_processor.save_pretrained(tmp_path)
    Lfm3TsConfig().save_pretrained(tmp_path)
    loaded = AutoProcessor.from_pretrained(tmp_path)
    assert isinstance(loaded, Lfm3TsProcessor)
    assert loaded.ts_token_id == 125_033
    assert '"timeseries"' in loaded.chat_template


def test_value_compression_only_changes_outliers():
    from lfm3_ts import Lfm3TsEncoder, Lfm3TsEncoderConfig

    torch.manual_seed(0)
    plain = Lfm3TsEncoder(Lfm3TsEncoderConfig(patch_size=4, hidden_size=16, intermediate_size=32, num_hidden_layers=1))
    squashed = Lfm3TsEncoder(
        Lfm3TsEncoderConfig(
            patch_size=4, hidden_size=16, intermediate_size=32, num_hidden_layers=1, value_compression="asinh", use_differences=True
        )
    )
    values = torch.zeros(1, 2, 4)
    values[0, 1, 2] = 15.0
    seen = {}
    squashed.patch_embed.register_forward_hook(lambda module, inputs, output: seen.update(inputs=inputs[0]))
    squashed(values)
    # per step: value, mask, difference, then the position encoding of the 4 steps
    patch = seen["inputs"][0, 1]
    assert torch.allclose(patch[:4], torch.asinh(values[0, 1])) and patch[:4].max() < 3.5
    assert torch.allclose(patch[8:12], torch.asinh(torch.tensor([0.0, 0.0, 15.0, -15.0])))
    assert plain.config.value_compression == "none"
    with pytest.raises(Exception, match="value_compression"):
        Lfm3TsEncoderConfig(value_compression="log")


def test_projector_position_tells_identical_patches_apart(tmp_path):
    config = tiny_config()
    config.projector_position_dim = 8
    torch.manual_seed(0)
    model = Lfm3TsForConditionalGeneration(config).eval()
    projector = model.model.multi_modal_projector
    # the position starts at the scale of the content
    assert 0.3 < projector.position_embed.weight.std() / (config.projector_output_scale / 2) < 3
    same = torch.ones(1, 5, config.ts_config.hidden_size)
    tokens = projector(same)[0]
    assert not torch.allclose(tokens[0], tokens[1]) and not torch.allclose(tokens[1], tokens[4])
    without = Lfm3TsForConditionalGeneration(tiny_config()).model.multi_modal_projector
    assert without.position_embed is None and torch.allclose(without(same)[0, 0], without(same)[0, 4])

    model.save_pretrained(tmp_path)
    reloaded = Lfm3TsForConditionalGeneration.from_pretrained(tmp_path).eval()
    assert reloaded.config.projector_position_dim == 8
    assert torch.equal(reloaded.model.multi_modal_projector(same), projector(same))
    with pytest.raises(ValueError, match="projector_position_dim"):
        Lfm3TsConfig(projector_position_dim=7)


def _many_series_conversation(turns: int) -> list[dict]:
    rng = np.random.default_rng(0)
    content = [{"type": "text", "text": "This is a recording of 12 channels. "}]
    for lead in range(12):
        content += [{"type": "text", "text": f"Channel {lead}: "}, {"type": "timeseries", "timeseries": rng.normal(size=1000).round(3)}, {"type": "text", "text": ". "}]
    messages = []
    for turn in range(turns):
        question = {"type": "text", "text": f"Does this recording show finding {turn}? The possible answers are: no; not sure; yes. Answer with the one or the several that apply, separated by commas, and nothing else."}
        messages.append({"role": "user", "content": content + [question] if turn == 0 else [question]})
        messages.append({"role": "assistant", "content": "yes" if turn % 2 else "a level shift, none"})
    return messages


@pytest.mark.slow
def test_a_conversation_of_several_turns_is_tokenized_as_the_template_writes_it(lfm3_processor):
    from lfm3_ts.training import Lfm3TsDataCollator

    collator = Lfm3TsDataCollator(lfm3_processor)
    messages = _many_series_conversation(9)
    encoded = collator.encode(messages)
    # the same tokens as the whole conversation rendered at once, up to what the template writes after the last answer
    whole = lfm3_processor.apply_chat_template(messages, tokenize=True)["input_ids"][0]
    assert whole[: len(encoded["input_ids"])] == encoded["input_ids"]
    assert len(encoded["input_ids"]) < 4096
    # the loss is on the nine answers and on nothing else
    tokenizer = lfm3_processor.tokenizer
    labelled = [token for token in encoded["labels"] if token != -100]
    expected = []
    for message in messages[1::2]:
        expected += tokenizer(message["content"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    assert labelled == expected
    assert encoded["input_values"].shape[0] == 12
    # one turn gives what it always gave
    single = collator.encode(messages[:2])
    assert single["input_ids"] == encoded["input_ids"][: len(single["input_ids"])]
    # the turns that do not fit are left out, whole
    short = Lfm3TsDataCollator(lfm3_processor, max_length=len(single["input_ids"]) + 120).encode(messages)
    assert len(single["input_ids"]) < len(short["input_ids"]) <= len(single["input_ids"]) + 120
    assert short["input_ids"][-1] == tokenizer.eos_token_id
    print("tokens: one turn", len(single["input_ids"]), "nine turns", len(encoded["input_ids"]))
