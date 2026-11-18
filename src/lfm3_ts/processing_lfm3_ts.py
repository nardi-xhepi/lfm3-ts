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
"""Processor for LFM3-TS."""

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

from .feature_extraction_lfm3_ts import Lfm3TsFeatureExtractor, Lfm3TsSeriesStats


TS_TOKEN = "<|ts|>"

# The LFM3 chat template renders `image`, `video` and `text` content parts; these add a `timeseries` part. Each pair is
# (anchor found in the LFM3 template, branch inserted before it).
_CHAT_TEMPLATE_PATCHES = (
    (
        '{%- elif content is mapping and content.get("type") == "text" -%}',
        '{%- elif content is mapping and content.get("type") == "timeseries" -%}\n'
        '        {{- "' + TS_TOKEN + '" -}}\n    ',
    ),
    (
        '{%- elif item is mapping and item.get("type") == "text" -%}',
        '{%- elif item is mapping and item.get("type") == "timeseries" -%}\n'
        '                {%- set _ns.result = _ns.result + "' + TS_TOKEN + '" -%}\n            ',
    ),
)


def add_timeseries_to_chat_template(chat_template: str) -> str:
    """Teaches an LFM3 chat template to render `{"type": "timeseries"}` content parts as the placeholder token."""
    for anchor, branch in _CHAT_TEMPLATE_PATCHES:
        if chat_template.count(anchor) != 1:
            raise ValueError(f"Cannot add time series to this chat template: expected exactly one `{anchor}`.")
        chat_template = chat_template.replace(anchor, branch + anchor)
    return chat_template


def _is_array(item) -> bool:
    return hasattr(item, "ndim") and hasattr(item, "shape")


def _flatten_timeseries(timeseries) -> list[np.ndarray]:
    """Flat list of items, one per placeholder: 1D series or 2D `(num_channels, length)` multivariate series."""
    if _is_array(timeseries):
        return [np.asarray(timeseries, dtype=np.float64)]
    if not isinstance(timeseries, (list, tuple)):
        raise ValueError(f"`timeseries` must be arrays or lists of numbers, got {type(timeseries)}.")
    if len(timeseries) > 0 and all(isinstance(value, (int, float, np.number)) for value in timeseries):
        return [np.asarray(timeseries, dtype=np.float64)]
    return [item for nested in timeseries for item in _flatten_timeseries(nested)]


class Lfm3TsProcessor(ProcessorMixin):
    r"""
    Wraps an `Lfm3TsFeatureExtractor` and an LFM3 tokenizer.

    Each `<|ts|>` in the text stands for one item of `timeseries`. The processor replaces it with the item's
    normalization statistics in plain text, followed by as many `<|ts|>` tokens as the model will fill with features:

    ```text
    [offset: 12.5000 | scale: 3.2000 | length: 256 | min: 4.1000 | max: 21.7000 | first: 9.8000 | last: 15.2000]<|ts|>...<|ts|>
    ```

    The statistics keep the original values recoverable (the encoder only sees standardized values), and the
    extremes and end points spare the model the arithmetic for the values most often asked about. A 2D item of
    shape `(num_channels, length)` expands to one such block per channel, each on its own line.

    Args:
        feature_extractor (`Lfm3TsFeatureExtractor`):
            The feature extractor.
        tokenizer (`PreTrainedTokenizerBase`):
            An LFM3 tokenizer holding the `<|ts|>` token.
        chat_template (`str`, *optional*):
            A chat template that renders `{"type": "timeseries"}` content parts as `<|ts|>`.
        position_marker_every (`int`, *optional*, defaults to 0):
            Writes the index of a time step, as text like `<64>`, before the token of the patch that starts there,
            every this many time steps. Must
            be a multiple of the patch size; 0 writes none.
        value_ruler (`bool`, *optional*, defaults to `False`):
            Also writes, in the statistics, the values that stand at -2, -1, 0, +1 and +2 in the standardized units
            the encoder sees. A checkpoint keeps the setting it was trained with.
    """

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        position_marker_every: int = 0,
        value_ruler: bool = False,
        **kwargs,
    ):
        if position_marker_every % feature_extractor.patch_size:
            raise ValueError(
                f"position_marker_every={position_marker_every} must be a multiple of the patch size "
                f"{feature_extractor.patch_size}."
            )
        self.position_marker_every = position_marker_every
        self.value_ruler = value_ruler
        self.ts_token = getattr(tokenizer, "ts_token", TS_TOKEN)
        self.ts_token_id = tokenizer.convert_tokens_to_ids(self.ts_token)
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    @classmethod
    def from_text_pretrained(
        cls,
        pretrained_model_name_or_path: str = "LiquidAI/LFM3-1.7B",
        patch_size: int = 8,
        position_marker_every: int = 0,
        value_ruler: bool = False,
        **kwargs,
    ) -> "Lfm3TsProcessor":
        """Builds a processor from an LFM3 text checkpoint: its tokenizer gains the `<|ts|>` token and its chat
        template a `timeseries` content part. `kwargs` go to the tokenizer's `from_pretrained`."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, extra_special_tokens={"ts_token": TS_TOKEN}, **kwargs
        )
        chat_template = add_timeseries_to_chat_template(tokenizer.chat_template)
        tokenizer.chat_template = chat_template
        return cls(
            Lfm3TsFeatureExtractor(patch_size=patch_size),
            tokenizer,
            chat_template=chat_template,
            position_marker_every=position_marker_every,
            value_ruler=value_ruler,
        )

    def format_series(self, stats: Lfm3TsSeriesStats) -> str:
        """The text standing for one series: its statistics, then a start, per-patch and end placeholder."""
        header = (
            f"[offset: {stats.offset:.4f} | scale: {stats.scale:.4f} | length: {stats.length}"
            f" | min: {stats.minimum:.4f} | max: {stats.maximum:.4f}"
            f" | first: {stats.first:.4f} | last: {stats.last:.4f}"
        )
        if self.value_ruler:
            marks = ", ".join(f"{level:+d}: {stats.offset + level * stats.scale:.4f}" for level in (-2, -1, 0, 1, 2))
            header += f" | levels {marks}"
        header += "]"
        if not self.position_marker_every:
            return header + self.ts_token * (stats.num_patches + 2)
        patch_size = self.feature_extractor.patch_size
        pieces = [header, self.ts_token]
        for patch in range(stats.num_patches):
            index = patch * patch_size
            if patch and index % self.position_marker_every == 0:
                pieces.append(f"<{index}>")
            pieces.append(self.ts_token)
        return "".join(pieces) + self.ts_token

    def __call__(self, text=None, timeseries=None, return_tensors=None, **kwargs) -> BatchFeature:
        """
        Args:
            text (`str` or `list[str]`):
                The prompts, with one `<|ts|>` per item of `timeseries`.
            timeseries (*optional*):
                A series, or a list of them (or one list per prompt), in the order of their placeholders. A 1D array
                or list of numbers is one series; a 2D array of shape `(num_channels, length)` is a multivariate
                series. NaN marks a missing value.
            return_tensors (`str`, *optional*):
                The tensor type to return, e.g. `"pt"`.
            kwargs:
                Passed to the tokenizer (`padding`, `truncation`, `add_special_tokens`, ...).

        Returns:
            `BatchFeature` with `input_ids`, `attention_mask` and, when `timeseries` is given, `input_values` and
            `input_values_mask` of shape `(num_series, max_patches, patch_size)`.
        """
        if text is None:
            raise ValueError("You must provide `text`.")
        prompts = [text] if isinstance(text, str) else list(text)
        items = [] if timeseries is None else _flatten_timeseries(timeseries)

        num_placeholders = sum(prompt.count(self.ts_token) for prompt in prompts)
        if num_placeholders != len(items):
            raise ValueError(
                f"Found {num_placeholders} {self.ts_token} tokens in the text but received {len(items)} time series."
            )

        data = {}
        if items:
            channels, channels_per_item = [], []
            for index, item in enumerate(items):
                if item.ndim not in (1, 2):
                    raise ValueError(f"Time series {index} must be 1D or 2D (num_channels, length), got {item.shape}.")
                item_channels = [item] if item.ndim == 1 else list(item)
                channels.extend(item_channels)
                channels_per_item.append(len(item_channels))

            input_values, input_values_mask, stats = self.feature_extractor.preprocess(channels)
            data.update(input_values=input_values, input_values_mask=input_values_mask)

            blocks = iter(self.format_series(stat) for stat in stats)
            replacements = iter("\n".join(next(blocks) for _ in range(count)) for count in channels_per_item)
            expanded = []
            for prompt in prompts:
                pieces = prompt.split(self.ts_token)
                expanded.append("".join(piece + next(replacements) for piece in pieces[:-1]) + pieces[-1])
            prompts = expanded

        text_inputs = self.tokenizer(prompts, **kwargs)
        return BatchFeature(data={**text_inputs, **data}, tensor_type=return_tensors)

    def apply_chat_template(
        self,
        conversation,
        chat_template: str | None = None,
        tokenize: bool = False,
        return_dict: bool = True,
        add_generation_prompt: bool = False,
        **kwargs,
    ):
        """
        Renders one conversation, or a batch of them, and optionally tokenizes it together with its time series.

        A time series is a content part `{"type": "timeseries", "timeseries": array}`. Template arguments (`tools`,
        `documents`, `continue_final_message`, `reasoning_effort`, `enable_thinking`, `preserve_thinking`) are
        forwarded to the template; other `kwargs` go to `__call__` when `tokenize=True`.
        """
        is_batched = isinstance(conversation, (list, tuple)) and isinstance(conversation[0], (list, tuple))
        conversations = conversation if is_batched else [conversation]

        timeseries, rendered_conversations = [], []
        for messages in conversations:
            rendered_messages = []
            for message in messages:
                content = message.get("content")
                if isinstance(content, (list, tuple)):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "timeseries":
                            timeseries.append(part["timeseries"])
                            part = {"type": "timeseries"}
                        parts.append(part)
                    message = {**message, "content": parts}
                rendered_messages.append(message)
            rendered_conversations.append(rendered_messages)

        template_keys = (
            "tools",
            "documents",
            "continue_final_message",
            "reasoning_effort",
            "enable_thinking",
            "preserve_thinking",
        )
        template_kwargs = {key: kwargs.pop(key) for key in template_keys if key in kwargs}
        prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                chat_template=chat_template or self.chat_template,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **template_kwargs,
            )
            for messages in rendered_conversations
        ]
        if not tokenize:
            return prompts if is_batched else prompts[0]

        # the template writes the special tokens itself
        kwargs.setdefault("add_special_tokens", False)
        inputs = self(text=prompts, timeseries=timeseries or None, **kwargs)
        return inputs if return_dict else inputs["input_ids"]

    @property
    def model_input_names(self):
        names = self.tokenizer.model_input_names + self.feature_extractor.model_input_names
        return list(dict.fromkeys(names))


__all__ = ["Lfm3TsProcessor", "add_timeseries_to_chat_template"]
