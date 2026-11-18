# Copyright 2026 Nardi Xhepi.
#
# This file is part of LFM3-TS, distributed under the LFM Open License v1.0: see LICENSE and NOTICE at the root
# of the repository.
"""LFM3-TS: an LFM3 language model that reads time series through a patch encoder.

Importing the package registers its classes with the transformers Auto classes, so a saved LFM3-TS checkpoint loads
with `AutoConfig`, `AutoModel`, `AutoModelForCausalLM`, `AutoFeatureExtractor` and `AutoProcessor`.

Only numpy, torch and transformers are needed here. What training needs on top (data streams, the collator, token
budget batches) lives in `lfm3_ts.training`, which this module does not import.
"""

from transformers import AutoConfig, AutoFeatureExtractor, AutoModel, AutoModelForCausalLM, AutoProcessor

from .configuration_lfm3_ts import Lfm3TsConfig, Lfm3TsEncoderConfig
from .feature_extraction_lfm3_ts import Lfm3TsFeatureExtractor, Lfm3TsSeriesStats
from .modeling_lfm3_ts import (
    Lfm3TsEncoder,
    Lfm3TsForConditionalGeneration,
    Lfm3TsModel,
    Lfm3TsPreTrainedModel,
    build_chat_messages,
    get_ts_patch_counts,
)
from .processing_lfm3_ts import Lfm3TsProcessor, add_timeseries_to_chat_template


__version__ = "0.1.0"

AutoConfig.register("lfm3_ts_encoder", Lfm3TsEncoderConfig, exist_ok=True)
AutoConfig.register("lfm3_ts", Lfm3TsConfig, exist_ok=True)
AutoModel.register(Lfm3TsEncoderConfig, Lfm3TsEncoder, exist_ok=True)
AutoModel.register(Lfm3TsConfig, Lfm3TsModel, exist_ok=True)
AutoModelForCausalLM.register(Lfm3TsConfig, Lfm3TsForConditionalGeneration, exist_ok=True)
AutoFeatureExtractor.register(Lfm3TsConfig, Lfm3TsFeatureExtractor, exist_ok=True)
AutoProcessor.register(Lfm3TsConfig, Lfm3TsProcessor, exist_ok=True)


def register_for_hub():
    """Makes `save_pretrained` and `push_to_hub` ship this package's source files with the checkpoint, so that it
    loads with `trust_remote_code=True` on machines without `lfm3_ts` installed."""
    Lfm3TsConfig.register_for_auto_class()
    Lfm3TsModel.register_for_auto_class("AutoModel")
    Lfm3TsForConditionalGeneration.register_for_auto_class("AutoModelForCausalLM")
    Lfm3TsFeatureExtractor.register_for_auto_class("AutoFeatureExtractor")
    Lfm3TsProcessor.register_for_auto_class("AutoProcessor")


def chat(model, processor, series, question: str, **kwargs) -> str:
    """Asks `question` about `series` and returns the answer: the same as `model.chat(processor, series, question)`."""
    return model.chat(processor, series, question, **kwargs)


__all__ = [
    "Lfm3TsConfig",
    "Lfm3TsEncoder",
    "Lfm3TsEncoderConfig",
    "Lfm3TsFeatureExtractor",
    "Lfm3TsForConditionalGeneration",
    "Lfm3TsModel",
    "Lfm3TsPreTrainedModel",
    "Lfm3TsProcessor",
    "Lfm3TsSeriesStats",
    "add_timeseries_to_chat_template",
    "build_chat_messages",
    "chat",
    "get_ts_patch_counts",
    "register_for_hub",
]
