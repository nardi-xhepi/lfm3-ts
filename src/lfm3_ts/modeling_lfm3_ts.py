# Copyright 2026 The HuggingFace Team. All rights reserved.
# Copyright 2026 Nardi Xhepi.
#
# NOTICE OF CHANGES: this file is part of LFM3-TS. It is adapted from `transformers.models.lfm3_vl`
# (Hugging Face Transformers, Apache License 2.0) and was changed by Nardi Xhepi: the vision tower and image
# inputs are replaced by a time-series encoder and time-series inputs.
#
# The portions taken from Transformers remain available under the Apache License, Version 2.0; a copy is in
# LICENSES/Apache-2.0.txt. LFM3-TS as a whole is distributed under the LFM Open License v1.0: see LICENSE and
# NOTICE at the root of the repository.
#
# Unless required by applicable law or agreed to in writing, this software is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""LFM3-TS model: a time-series tower and a projector in front of the LFM3 language model."""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.lfm3.modeling_lfm3 import Lfm3Model
from transformers.utils import can_return_tuple, torch_compilable_check

from .configuration_lfm3_ts import Lfm3TsConfig, Lfm3TsEncoderConfig


def get_ts_patch_counts(input_values_mask: torch.Tensor) -> torch.Tensor:
    """Number of patches each series occupies: up to and including its last patch holding an observed time step.

    Args:
        input_values_mask (`torch.Tensor` of shape `(num_series, max_patches, patch_size)`):
            1 for observed time steps, 0 for padding and missing values.

    Returns:
        `torch.LongTensor` of shape `(num_series,)`.
    """
    observed = input_values_mask.bool().any(-1)
    index = torch.arange(1, observed.shape[1] + 1, device=observed.device)
    return (observed * index).amax(-1)


def _as_series_list(series) -> list[np.ndarray]:
    """One 1D array per series, from a 1D array or list of numbers, a list of them, or a 2D array of rows."""
    if hasattr(series, "detach"):
        series = series.detach().cpu().numpy()
    if hasattr(series, "to_numpy"):
        series = series.to_numpy()
    if isinstance(series, np.ndarray):
        items = [series] if series.ndim == 1 else list(series)
    elif isinstance(series, (list, tuple)) and len(series) and all(
        isinstance(value, (int, float, np.number)) for value in series
    ):
        items = [series]
    elif isinstance(series, (list, tuple)):
        items = list(series)
    else:
        raise ValueError(f"`series` must be an array or a list of numbers, or a list of them, got {type(series)}.")
    arrays = []
    for index, item in enumerate(items):
        if hasattr(item, "detach"):
            item = item.detach().cpu().numpy()
        array = np.asarray(item, dtype=np.float64)
        if array.ndim != 1 or not array.size:
            raise ValueError(f"Series {index} must be one-dimensional and not empty, got shape {array.shape}.")
        arrays.append(array)
    if not arrays:
        raise ValueError("`series` is empty.")
    return arrays


def build_chat_messages(series, question: str) -> list[dict]:
    """The conversation `Lfm3TsForConditionalGeneration.chat` sends: one user message holding the series and the
    question, in the wording the released model was trained on for one series
    (`"This is a time series of length N: "`, the series, `". "`, the question).

    Args:
        series:
            A 1D array or list of numbers; or several series, as a list of them or as the rows of a 2D array; or
            `None` for a question without a series. NaN marks a missing value.
        question (`str`):
            What to ask.
    """
    question = question.strip()
    if series is None:
        return [{"role": "user", "content": [{"type": "text", "text": question}]}]
    arrays = _as_series_list(series)
    if len(arrays) == 1:
        content = [
            {"type": "text", "text": f"This is a time series of length {len(arrays[0])}: "},
            {"type": "timeseries", "timeseries": arrays[0]},
        ]
    else:
        content = [{"type": "text", "text": f"There are {len(arrays)} time series. "}]
        for index, array in enumerate(arrays):
            if index:
                content.append({"type": "text", "text": "; "})
            content.append({"type": "text", "text": f"Series {index + 1}, of length {len(array)}: "})
            content.append({"type": "timeseries", "timeseries": array})
    content.append({"type": "text", "text": f". {question}"})
    return [{"role": "user", "content": content}]


class Lfm3TsEncoderLayer(nn.Module):
    """A pre-norm residual feed-forward layer applied to each patch on its own."""

    def __init__(self, config: Lfm3TsEncoderConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.fc2(self.act(self.fc1(self.norm(hidden_states))))


def apply_rotary_position_embedding(states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotates each pair of channels of `(batch, heads, patches, head_dim)` states by its patch's angle."""
    even, odd = states[..., 0::2], states[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)


class Lfm3TsAttentionLayer(nn.Module):
    """A pre-norm bidirectional self-attention layer over the patches of one series, with rotary position
    embeddings, so that a patch can read its context: a period or a trend spans several patches."""

    def __init__(self, config: Lfm3TsEncoderConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.feed_forward = Lfm3TsEncoderLayer(config)

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        num_series, num_patches, hidden_size = hidden_states.shape
        states = self.attention_norm(hidden_states)
        split = lambda tensor: tensor.view(num_series, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        queries = apply_rotary_position_embedding(split(self.q_proj(states)), cos, sin)
        keys = apply_rotary_position_embedding(split(self.k_proj(states)), cos, sin)
        attended = F.scaled_dot_product_attention(queries, keys, split(self.v_proj(states)), attn_mask=attention_mask)
        attended = attended.transpose(1, 2).reshape(num_series, num_patches, hidden_size)
        return self.feed_forward(hidden_states + self.o_proj(attended))


class Lfm3TsPreTrainedModel(PreTrainedModel):
    config: Lfm3TsConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]

    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Lfm3TsMultiModalProjector):
            scale = module.output_scale
            init.normal_(module.boundary_embeddings, mean=0.0, std=scale)
            init.constant_(module.output_norm.weight, scale)
            init.zeros_(module.output_norm.bias)
            if module.position_embed is not None:
                # sinusoids have a mean square of 1/2: the position starts at the scale of the content
                init.normal_(module.position_embed.weight, mean=0.0, std=scale / math.sqrt(module.position_dim / 2))


class Lfm3TsEncoder(Lfm3TsPreTrainedModel):
    """The LFM3-TS time-series encoder: a linear patch embedding over the values, observation mask and position of
    each time step of a patch, followed by residual feed-forward layers. With `num_attention_layers=0` patches are
    encoded independently and only the language model relates them; attention layers let them read each other."""

    config: Lfm3TsEncoderConfig
    main_input_name = "input_values"
    _input_embed_layer = "patch_embed"

    def __init__(self, config: Lfm3TsEncoderConfig):
        super().__init__(config)
        # each time step carries its value, its observation mask and the encoding of its index in the series
        channels = 2 + int(config.use_differences) + config.position_embedding_dim
        self.patch_embed = nn.Linear(channels * config.patch_size, config.hidden_size)
        self.layers = nn.ModuleList([Lfm3TsEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.attention_layers = nn.ModuleList(
            [Lfm3TsAttentionLayer(config) for _ in range(config.num_attention_layers)]
        )
        self.post_init()

    def forward(
        self, input_values: torch.Tensor, input_values_mask: torch.Tensor | None = None, **kwargs
    ) -> BaseModelOutputWithPooling:
        r"""
        input_values (`torch.FloatTensor` of shape `(num_series, max_patches, patch_size)`):
            The normalized values of every series, cut into patches.
        input_values_mask (`torch.Tensor` of shape `(num_series, max_patches, patch_size)`, *optional*):
            1 for observed time steps, 0 for padding and missing values. Defaults to all observed.
        """
        if input_values_mask is None:
            input_values_mask = torch.ones_like(input_values)
        dtype = self.patch_embed.weight.dtype
        # outliers are compressed so that no value outweighs the rest of its patch
        compress = torch.asinh if self.config.value_compression == "asinh" else (lambda tensor: tensor)
        patches = [compress(input_values.to(dtype)), input_values_mask.to(dtype)]
        if self.config.use_differences:
            # the change from the previous step, across patch borders; zero where either step is unobserved
            values, observed = input_values.to(dtype).flatten(1), input_values_mask.to(dtype).flatten(1)
            differences = torch.zeros_like(values)
            differences[:, 1:] = (values[:, 1:] - values[:, :-1]) * observed[:, 1:] * observed[:, :-1]
            patches.append(compress(differences).view_as(input_values))
        if self.config.position_embedding_dim:
            patches.append(self.get_position_features(input_values_mask).to(dtype))
        hidden_states = self.patch_embed(torch.cat(patches, dim=-1))
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        if self.attention_layers:
            num_patches = hidden_states.shape[1]
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            device = hidden_states.device
            frequencies = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device) / head_dim))
            angles = torch.arange(num_patches, device=device)[:, None] * frequencies[None, :]
            cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
            # a patch attends to the patches of its own series, not to the padding that aligns the batch
            is_patch = torch.arange(num_patches, device=device)[None, :] < get_ts_patch_counts(input_values_mask)[:, None]
            attention_mask = is_patch[:, None, None, :]
            for layer in self.attention_layers:
                hidden_states = layer(hidden_states, cos, sin, attention_mask)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


    def get_position_features(self, input_values_mask: torch.Tensor) -> torch.Tensor:
        """Sinusoidal encoding of the index of every observed time step, zero elsewhere.

        Without it a patch holds no trace of where it sits, and the language model could only locate an event by
        counting tokens. Sinusoids need no training and extend to any length.

        Returns:
            `torch.FloatTensor` of shape `(num_series, max_patches, patch_size * position_embedding_dim)`.
        """
        num_series, max_patches, patch_size = input_values_mask.shape
        half = self.config.position_embedding_dim // 2
        device = input_values_mask.device
        positions = torch.arange(max_patches * patch_size, device=device, dtype=torch.float32)
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / half)
        angles = positions[:, None] * frequencies[None, :]
        features = torch.cat([angles.sin(), angles.cos()], dim=-1).view(max_patches, patch_size, 2 * half)
        features = features[None] * input_values_mask[..., None].to(features.dtype)
        return features.flatten(2)


class Lfm3TsMultiModalProjector(nn.Module):
    """Projects each patch to one language token and frames every series with a learned start and end embedding.

    The boundaries live here, not in the vocabulary, so that every parameter introduced for time series sits outside
    the language model and the backbone can stay frozen."""

    def __init__(self, config: Lfm3TsConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.ts_config.hidden_size, eps=config.projector_layer_norm_eps)
        self.linear_1 = nn.Linear(
            config.ts_config.hidden_size, config.projector_hidden_size, bias=config.projector_bias
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.projector_hidden_size, config.text_config.hidden_size, bias=config.projector_bias
        )
        # brings the features to the scale of the token embeddings they sit among; its weight starts at that scale
        self.output_norm = nn.LayerNorm(config.text_config.hidden_size, eps=config.projector_layer_norm_eps)
        self.output_scale = config.projector_output_scale
        # the position of a patch joins its token after the normalization, where the content cannot outweigh it
        self.patch_size = config.ts_config.patch_size
        self.position_dim = config.projector_position_dim
        self.position_embed = (
            nn.Linear(self.position_dim, config.text_config.hidden_size, bias=False) if self.position_dim else None
        )
        self.boundary_embeddings = nn.Parameter(torch.empty(2, config.text_config.hidden_size))

    def forward(self, ts_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer_norm(ts_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.output_norm(hidden_states)
        if self.position_embed is not None:
            hidden_states = hidden_states + self.position_embed(self.get_position_features(hidden_states))
        return hidden_states

    def get_position_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Sinusoidal encoding of the index of the first time step of each patch, of shape `(num_patches, position_dim)`."""
        half = self.position_dim // 2
        device = hidden_states.device
        starts = torch.arange(hidden_states.shape[1], device=device, dtype=torch.float32) * self.patch_size
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / half)
        angles = starts[:, None] * frequencies[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1).to(hidden_states.dtype)


@dataclass
class Lfm3TsModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    ts_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_ts_tokens, hidden_size)`: the projected time-series features that
        replaced the placeholder tokens.
    """

    ts_hidden_states: torch.FloatTensor | None = None


@dataclass
class Lfm3TsCausalLMOutputWithPast(CausalLMOutputWithPast):
    r"""
    ts_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_ts_tokens, hidden_size)`: the projected time-series features that
        replaced the placeholder tokens.
    """

    ts_hidden_states: torch.FloatTensor | None = None


class Lfm3TsModel(Lfm3TsPreTrainedModel):
    """The LFM3-TS model: a time-series encoder and an LFM3 language model, without a language modeling head."""

    def __init__(self, config: Lfm3TsConfig):
        super().__init__(config)
        self.ts_tower = Lfm3TsEncoder(config.ts_config)
        self.multi_modal_projector = Lfm3TsMultiModalProjector(config)
        self.language_model = Lfm3Model(config.text_config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.language_model.set_input_embeddings(value)

    def get_ts_features(
        self, input_values: torch.Tensor, input_values_mask: torch.Tensor | None = None, **kwargs
    ) -> BaseModelOutputWithPooling:
        r"""
        Encodes and projects the series. `pooler_output` holds one tensor per series of shape
        `(num_patches + 2, hidden_size)`: a start embedding, one token per patch, and an end embedding.

        input_values (`torch.FloatTensor` of shape `(num_series, max_patches, patch_size)`):
            The normalized values of every series, cut into patches.
        input_values_mask (`torch.Tensor` of shape `(num_series, max_patches, patch_size)`, *optional*):
            1 for observed time steps, 0 for padding and missing values. Defaults to all observed.
        """
        if input_values_mask is None:
            input_values_mask = torch.ones_like(input_values)
        ts_outputs = self.ts_tower(input_values, input_values_mask, **kwargs)
        ts_features = self.multi_modal_projector(ts_outputs.last_hidden_state)
        start, end = self.multi_modal_projector.boundary_embeddings.to(ts_features.dtype).split(1)
        patch_counts = get_ts_patch_counts(input_values_mask).tolist()
        ts_outputs.pooler_output = tuple(
            torch.cat([start, features[:count], end]) for features, count in zip(ts_features, patch_counts)
        )
        return ts_outputs

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor,
        ts_features: torch.FloatTensor,
    ) -> torch.Tensor:
        """Placeholder mask from `input_ids` or `inputs_embeds`, checked against the feature count."""
        if input_ids is None:
            special_ts_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), self.config.ts_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_ts_mask = special_ts_mask.all(-1)
        else:
            special_ts_mask = input_ids == self.config.ts_token_id

        n_ts_tokens = special_ts_mask.sum()
        torch_compilable_check(
            n_ts_tokens * inputs_embeds.shape[-1] == ts_features.numel(),
            f"Time-series features and time-series tokens do not match, tokens: {n_ts_tokens}, "
            f"features: {ts_features.shape[0]}",
        )
        return special_ts_mask.unsqueeze(-1).to(inputs_embeds.device)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        input_values: torch.FloatTensor | None = None,
        input_values_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> tuple | Lfm3TsModelOutputWithPast:
        r"""
        input_values (`torch.FloatTensor` of shape `(num_series, max_patches, patch_size)`, *optional*):
            The normalized values of every series in the batch, in the order their placeholders appear.
        input_values_mask (`torch.Tensor` of shape `(num_series, max_patches, patch_size)`, *optional*):
            1 for observed time steps, 0 for padding and missing values. Defaults to all observed.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        ts_features = None
        if input_values is not None:
            ts_outputs = self.get_ts_features(input_values, input_values_mask)
            ts_features = torch.cat(ts_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            ts_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, ts_features=ts_features)
            inputs_embeds = inputs_embeds.masked_scatter(ts_mask, ts_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return Lfm3TsModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            ts_hidden_states=ts_features,
        )


class Lfm3TsForConditionalGeneration(Lfm3TsPreTrainedModel, GenerationMixin):
    """The LFM3-TS model: a time-series encoder and an LFM3 language model with a language modeling head."""

    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: Lfm3TsConfig):
        super().__init__(config)
        self.model = Lfm3TsModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def get_ts_features(
        self, input_values: torch.Tensor, input_values_mask: torch.Tensor | None = None, **kwargs
    ) -> BaseModelOutputWithPooling:
        return self.model.get_ts_features(input_values, input_values_mask, **kwargs)

    @classmethod
    def from_text_pretrained(
        cls, pretrained_model_name_or_path: str = "LiquidAI/LFM3-1.7B", ts_config=None, **kwargs
    ) -> "Lfm3TsForConditionalGeneration":
        """Builds an LFM3-TS model around a pretrained LFM3 text checkpoint.

        The language model takes the checkpoint's weights; the time-series tower and projector are freshly
        initialized and need training. The projector's output starts at the scale of the checkpoint's embeddings.
        `kwargs` go to the text model's `from_pretrained` (`dtype`, `device_map`, ...), except those naming an
        `Lfm3TsConfig` field (`ts_token_id`, `projector_hidden_size`, ...).
        """
        from transformers import AutoModelForCausalLM

        config_fields = {
            "ts_token_id",
            "projector_hidden_act",
            "projector_hidden_size",
            "projector_bias",
            "projector_position_dim",
            "projector_output_scale",
        }
        config_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in config_fields}
        text_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if "projector_output_scale" not in config_kwargs:
            # start the time-series features at the scale of the embeddings they will sit among
            embeddings = text_model.get_input_embeddings().weight
            config_kwargs["projector_output_scale"] = float(embeddings.detach().float().pow(2).mean().sqrt())
        config = Lfm3TsConfig(text_config=text_model.config, ts_config=ts_config, **config_kwargs)

        model = cls(config)
        # adopt the loaded module itself: casting a fresh copy with `.to(dtype)` would also cast the buffers that
        # `from_pretrained` keeps in full precision, and the outputs would drift from the text model's
        model.model.language_model = text_model.model
        model.lm_head = text_model.lm_head
        reference = text_model.get_input_embeddings().weight
        model.model.ts_tower.to(device=reference.device, dtype=reference.dtype)
        model.model.multi_modal_projector.to(device=reference.device, dtype=reference.dtype)
        model.generation_config = text_model.generation_config
        return model

    @torch.no_grad()
    def chat(self, processor, series, question: str, max_new_tokens: int = 160, **generate_kwargs) -> str:
        r"""
        Asks one question about one or several series and returns the answer as text.

        Args:
            processor (`Lfm3TsProcessor`):
                The processor of the checkpoint.
            series:
                A 1D array or list of numbers; or several series, as a list of them or as the rows of a 2D array;
                or `None` for a question without a series. NaN marks a missing value.
            question (`str`):
                What to ask, e.g. `"Is there an anomaly? If yes, where?"`.
            max_new_tokens (`int`, *optional*, defaults to 160):
                The longest answer, in tokens.
            generate_kwargs:
                Passed to `generate`. Decoding is greedy unless `do_sample` is given.

        Example:

        ```python
        >>> series = np.sin(np.arange(512) / 8)
        >>> series[300] += 4
        >>> model.chat(processor, series, "Is there an anomaly? If yes, where?")
        ```"""
        messages = build_chat_messages(series, question)
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, enable_thinking=False, return_tensors="pt"
        ).to(self.device)
        generate_kwargs.setdefault("do_sample", False)
        output = self.generate(**inputs, max_new_tokens=max_new_tokens, **generate_kwargs)
        return processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        input_values: torch.FloatTensor | None = None,
        input_values_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> tuple | Lfm3TsCausalLMOutputWithPast:
        r"""
        input_values (`torch.FloatTensor` of shape `(num_series, max_patches, patch_size)`, *optional*):
            The normalized values of every series in the batch, in the order their placeholders appear.
        input_values_mask (`torch.Tensor` of shape `(num_series, max_patches, patch_size)`, *optional*):
            1 for observed time steps, 0 for padding and missing values. Defaults to all observed.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for the language modeling loss; positions set to `-100` are ignored.

        Example:

        ```python
        >>> import numpy as np
        >>> from lfm3_ts import Lfm3TsForConditionalGeneration, Lfm3TsProcessor

        >>> model = Lfm3TsForConditionalGeneration.from_text_pretrained("LiquidAI/LFM3-1.7B")
        >>> processor = Lfm3TsProcessor.from_text_pretrained("LiquidAI/LFM3-1.7B")

        >>> conversation = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "timeseries", "timeseries": np.sin(np.arange(256) / 8)},
        ...             {"type": "text", "text": "Where are the anomalies in this series?"},
        ...         ],
        ...     },
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ... )
        >>> outputs = model.generate(**inputs, max_new_tokens=64)
        >>> processor.batch_decode(outputs, skip_special_tokens=True)[0]
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            input_values=input_values,
            input_values_mask=input_values_mask,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Lfm3TsCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            ts_hidden_states=outputs.ts_hidden_states,
        )


__all__ = [
    "build_chat_messages",
    "Lfm3TsEncoder",
    "Lfm3TsForConditionalGeneration",
    "Lfm3TsModel",
    "Lfm3TsPreTrainedModel",
]
