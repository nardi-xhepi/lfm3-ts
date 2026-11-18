"""A tiny random LFM3-TS model and a word-level processor: what the tests of inference run on, offline, in seconds."""

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from lfm3_ts import Lfm3TsConfig, Lfm3TsFeatureExtractor, Lfm3TsForConditionalGeneration, Lfm3TsProcessor


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
    return Lfm3TsForConditionalGeneration(config).eval()


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
