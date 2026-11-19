---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM3-1.7B
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
language:
- en
tags:
- time-series
- time-series-question-answering
- anomaly-detection
- multimodal
- lfm3
- liquid
- chatts
- custom_code
datasets:
- ChatTSRepo/ChatTS-Training-Dataset
- databricks/databricks-dolly-15k
---

# LFM3-TS

A small language model you can chat with about time series. You give it one or several series and a question, and
it answers in text.

LFM3-TS is [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) from Liquid AI with a time-series encoder in front,
built on the ideas and the synthetic data of [ChatTS](https://github.com/NetManAIOps/ChatTS) (Xie et al., VLDB 2025). It has 1.8 billion
parameters.

- **Code and training recipe:** https://github.com/nardi-xhepi/<github-repo>
- **Blog post:** https://nardi-xhepi.github.io/blog/posts/lfm3-ts.html
- **Author:** Nardi Xhepi

This is an independent project, not affiliated with Liquid AI.

What you can ask about:

- **Trend**: the direction, the starting value, and how much it changes.
- **Seasonality**: whether there is a period, how long it is, and its amplitude.
- **Noise**: whether the series is noisy, and how much.
- **Local changes**: spikes, drops, level shifts and the like, with their position and amplitude.
- **Values**: the value at a point, or the lowest and highest value of a stretch.
- **Several series**: which ones have the same shape, or change at the same time.
- **Anomalies**: whether something is unusual for this series, where, and what it looks like.

## How to use

```bash
pip install -U "transformers>=5.23"
```

You also need PyTorch and NumPy. LFM3 was added to `transformers` in version 5.23; on Colab, which already has
PyTorch, run only the line above and restart the runtime.

```python
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained("<hub-id>", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("<hub-id>", trust_remote_code=True)

series = np.sin(np.arange(512) / 8)
series[300] += 4  # a spike

print(model.chat(processor, series, "Is there an anomaly in this time series? If so, where is it and what does it look like?"))
```

`series` is a 1D array or list, or a list of them for several series (NaN marks a missing value). Points are
numbered from 0. The model was trained on
series of up to 2,048 points, in English.

## Model

| | |
|---|---|
| Language model | LFM3-1.7B, every weight fine-tuned except the token embeddings |
| Time-series encoder | patches of 8 points, 4 feed-forward and 2 attention layers, width 1024 |
| Projector | MLP to the width of the language model (2048) |
| Tokens per series | one per patch, plus 2: a series of 512 points costs 66 tokens |
| Parameters | 1.8 billion |

The series is standardized for the encoder, and its offset and scale are written in the prompt as text, so the
answers can give real values.

## Results

**General understanding: the ChatTS benchmark** ([sets A and B](https://zenodo.org/records/14349206)). It asks
about trend, seasonality, noise, local changes, and the relations between several series. Scored with the scoring
code of the ChatTS repository:

| | Parameters | Set A, categorical | Set A, numerical | Set B, categorical | Set B, numerical |
|---|---|---|---|---|---|
| **LFM3-TS** | 1.8B | 82.0% | 77.5% | 79.5% | 76.8% |
| ChatTS-8B | 8B | 83.8% | 82.3% | 84.6% | 82.9% |
| LFM3-1.7B, numbers as text | 1.7B | 10.4% | 4.0% | 14.2% | 3.0% |

**One task in detail: anomalies in real series.** 234 windows of 512 points from seven sources of
[TSB-AD-U](https://github.com/TheDatumOrg/TSB-AD) that neither model was trained on (105 with a labelled anomaly, 129
without). An answer is correct when it says "yes" and names a point within 16 points of the labelled stretch.

| | Parameters | Precision | Recall | F1 | False alarms |
|---|---|---|---|---|---|
| **LFM3-TS** | 1.8B | 66% | 64% | 0.65 | 28 of 129 |
| ChatTS-8B | 8B | 40% | 53% | 0.46 | 61 of 129 |
| LFM3-1.7B, numbers as text | 1.7B | 1% | 1% | 0.01 | 111 of 129 |

**Reading values**: asked for the value at a point of a real series, LFM3-TS is within a quarter of a standard
deviation 57% of the time (median error 0.18 standard deviations).

## Training

Three stages, each starting from the weights of the one before: generated questions that can only be answered by
reading the series; the same questions on series of up to 2,048 points; then a mixture of ChatTS synthetic data
(48%), generated questions (19%), plain chat (15%), real series with labelled anomalies (10%) and real series with
an event put in (8%). The commands are in the code repository.

## Licence

LFM3-TS is released under the [LFM Open License v1.0](LICENSE), the licence of LFM3-1.7B, which the weights derive
from. See `NOTICE` for what was changed and for the sources and licences of the training data.

## Credits

[LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) (Liquid AI) ·
[ChatTS](https://github.com/NetManAIOps/ChatTS) (Xie et al.) ·
[TSB-AD](https://github.com/TheDatumOrg/TSB-AD) (Liu and Paparrizos) ·
[databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) ·
[Hugging Face Transformers](https://github.com/huggingface/transformers)
