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
"""LFM3-TS configuration."""

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig


@strict
class Lfm3TsEncoderConfig(PreTrainedConfig):
    r"""
    Configuration of the LFM3-TS time-series encoder.

    Args:
        patch_size (`int`, *optional*, defaults to 8):
            The number of consecutive time steps packed into one patch, i.e. one language token.
        position_embedding_dim (`int`, *optional*, defaults to 16):
            The size of the sinusoidal encoding of each time step's index, given to the encoder next to its value so
            that a patch knows where in the series it sits. Must be even; 0 disables it.
        use_differences (`bool`, *optional*, defaults to `False`):
            Whether each time step also carries its change from the previous step, which makes spikes, shifts and
            slopes explicit.
        value_compression (`str`, *optional*, defaults to `"none"`):
            `"asinh"` passes the standardized values (and differences) through `asinh` before the patch embedding:
            close to the identity for ordinary values, logarithmic for outliers. A spike of 15 standard deviations
            otherwise dominates its patch and drowns the position of the patch along with it.
        hidden_size (`int`, *optional*, defaults to 1024):
            The width of the encoder.
        intermediate_size (`int`, *optional*, defaults to 2560):
            The width of the feed-forward layer inside each encoder layer.
        num_hidden_layers (`int`, *optional*, defaults to 4):
            The number of residual feed-forward layers applied to each patch.
        num_attention_layers (`int`, *optional*, defaults to 0):
            The number of bidirectional self-attention layers, with rotary position embeddings, applied across the
            patches of a series after the feed-forward layers. 0 keeps the patches independent, as in ChatTS.
        num_attention_heads (`int`, *optional*, defaults to 8):
            The number of heads of the attention layers.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation of the encoder layers.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the weight initializer.
    """

    model_type = "lfm3_ts_encoder"
    base_config_key = "ts_config"

    patch_size: int = 8
    position_embedding_dim: int = 16
    use_differences: bool = False
    value_compression: str = "none"
    hidden_size: int = 1024
    intermediate_size: int = 2560
    num_hidden_layers: int = 4
    num_attention_layers: int = 0
    num_attention_heads: int = 8
    hidden_act: str = "gelu"
    norm_eps: float = 1e-5
    initializer_range: float = 0.02

    def validate_architecture(self):
        """Part of ``@strict``-powered validation."""
        if self.patch_size < 1:
            raise ValueError(f"`patch_size` must be positive, got {self.patch_size}.")
        if self.position_embedding_dim < 0 or self.position_embedding_dim % 2:
            raise ValueError(f"`position_embedding_dim` must be even and >= 0, got {self.position_embedding_dim}.")
        if self.value_compression not in ("none", "asinh"):
            raise ValueError(f'`value_compression` must be "none" or "asinh", got {self.value_compression!r}.')
        if self.num_attention_layers and (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("`hidden_size // num_attention_heads` must be even for rotary position embeddings.")
        if self.num_attention_layers and self.hidden_size % self.num_attention_heads:
            raise ValueError("`hidden_size` must be a multiple of `num_attention_heads`.")


@strict
class Lfm3TsConfig(PreTrainedConfig):
    r"""
    Configuration of an LFM3-TS model: an LFM3 language model reading time series through a patch encoder.

    Args:
        text_config (`dict` or `PreTrainedConfig`, *optional*):
            The configuration of the LFM3 language model. Defaults to `Lfm3Config()`.
        ts_config (`dict` or `Lfm3TsEncoderConfig`, *optional*):
            The configuration of the time-series encoder. Defaults to `Lfm3TsEncoderConfig()`.
        ts_token_id (`int`, *optional*, defaults to 125033):
            The id of the placeholder token whose embeddings are replaced by time-series features. 125033 is the
            first unused row of the LFM3 embedding matrix, so adding the token does not resize the embeddings.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation of the multimodal projector.
        projector_hidden_size (`int`, *optional*, defaults to 4096):
            The hidden width of the multimodal projector.
        projector_bias (`bool`, *optional*, defaults to `True`):
            Whether the projector's linear layers have a bias.
        projector_layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon of the layer normalizations of the projector.
        projector_position_dim (`int`, *optional*, defaults to 0):
            The size of a sinusoidal encoding of each patch's first index, projected and added to the patch's token
            after the content path and its normalization. Whatever the patch holds, its token then states where it
            sits. Must be even; 0 disables it.
        projector_output_scale (`float`, *optional*, defaults to 0.02):
            The per-dimension scale the projected features start at. It should equal that of the language model's
            token embeddings (0.0207 for LFM3-1.7B), which enter the residual stream unnormalized: features a few
            times larger drown the layers' updates at their positions.
    """

    model_type = "lfm3_ts"
    sub_configs = {"text_config": AutoConfig, "ts_config": Lfm3TsEncoderConfig}

    text_config: dict | PreTrainedConfig | None = None
    ts_config: dict | PreTrainedConfig | None = None
    ts_token_id: int = 125_033
    projector_hidden_act: str = "gelu"
    projector_hidden_size: int = 4096
    projector_bias: bool = True
    projector_layer_norm_eps: float = 1e-6
    projector_position_dim: int = 0
    projector_output_scale: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if "lfm3" not in CONFIG_MAPPING:
            import transformers

            raise ImportError(
                f"LFM3-TS needs transformers 5.23 or later, the first release that includes LFM3, and found "
                f"{transformers.__version__}. Run `pip install -U \"transformers>=5.23\"` and restart the runtime."
            )
        if isinstance(self.ts_config, dict):
            self.ts_config = self.sub_configs["ts_config"](**self.ts_config)
        elif self.ts_config is None:
            self.ts_config = self.sub_configs["ts_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = {"model_type": "lfm3", **self.text_config}
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["lfm3"]()
        self.validate_ts_fields()

        super().__post_init__(**kwargs)

    def validate_ts_fields(self):
        """Part of ``@strict``-powered validation, so `validate()` and `save_pretrained` refuse later edits too."""
        # an `inputs_embeds` forward embeds the placeholder id to find it
        vocab_size = self.text_config.vocab_size
        if self.projector_position_dim < 0 or self.projector_position_dim % 2:
            raise ValueError(f"`projector_position_dim` must be even and >= 0, got {self.projector_position_dim}.")
        if not 0 <= self.ts_token_id < vocab_size:
            raise ValueError(f"ts_token_id={self.ts_token_id} must lie in [0, text_config.vocab_size={vocab_size})")


__all__ = ["Lfm3TsConfig", "Lfm3TsEncoderConfig"]
