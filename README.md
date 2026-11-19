# LFM3-TS

A small language model you can chat with about time series. You give it one or several series and a question, and
it answers in text.

LFM3-TS is [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) from Liquid AI with a time-series encoder in front,
built on the ideas and the synthetic data of [ChatTS](https://arxiv.org/abs/2412.03104). It has 1.8 billion
parameters. The full story is in the
[blog post](https://nardi-xhepi.github.io/blog/posts/lfm3-ts.html).

This is an independent project, not affiliated with Liquid AI.

What you can ask about:

- **Trend**: the direction, the starting value, and how much it changes.
- **Seasonality**: whether there is a period, how long it is, and its amplitude.
- **Noise**: whether the series is noisy, and how much.
- **Local changes**: spikes, drops, level shifts and the like, with their position and amplitude.
- **Values**: the value at a point, or the lowest and highest value of a stretch.
- **Several series**: which ones have the same shape, or change at the same time.
- **Anomalies**: whether something is unusual for this series, where, and what it looks like.

## Quickstart

```bash
pip install -U "transformers>=5.23"
```

You also need PyTorch and NumPy. LFM3 was added to `transformers` in version 5.23; on Colab, which already has
PyTorch, run only the line above and restart the runtime.

```python
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained("nardi/LFM3-TS", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("nardi/LFM3-TS", trust_remote_code=True)

series = np.sin(np.arange(512) / 8)
series[300] += 4  # a spike

print(model.chat(processor, series, "Is there an anomaly in this time series? If so, where is it and what does it look like?"))
```

`series` is a 1D array or list, or a list of them for several series (NaN marks a missing value). Points are
numbered from 0.

To skip `trust_remote_code`, or to train, install the package and `import lfm3_ts` first:

```bash
pip install "git+https://github.com/nardi-xhepi/lfm3-ts"                    # inference
pip install "lfm3-ts[train] @ git+https://github.com/nardi-xhepi/lfm3-ts"   # training
```

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

### Reproduce

```bash
# ChatTS benchmark, from the published answers
git clone --depth 1 https://github.com/NetManAIOps/ChatTS
python scripts/chatts_score.py --chatts ChatTS --dataset dataset_b.json eval/chatts/b-lfm3-ts.jsonl eval/chatts/b-chatts-8b.jsonl

# real series: rebuild the test windows from your copy of TSB-AD-U, then score the published answers
python scripts/make_real_data.py windows --dataset <TSB-AD-U folder> --out real
python eval/score_tsb.py --questions real/test_questions.jsonl

# or answer the questions again
python scripts/baseline.py run ours --model nardi/LFM3-TS --questions real/test_questions.jsonl --out my/lfm3-ts.jsonl
```

## How it works

- **One token per patch.** The series is standardized and cut into patches of 8 points; an encoder and a projector
  turn each patch into one token of the language model. 512 points cost 66 tokens.
- **Each point knows its position.** Every point enters the encoder with its index, so the model reads "point 105"
  instead of counting tokens.
- **The scale travels as text.** The offset and scale removed by standardization are written in the prompt, so the
  answer can give real values.
- **No new vocabulary.** The placeholder `<|ts|>` uses a free row of the LFM3 vocabulary.

## Training

Three stages, each starting from the weights of the one before, on one GPU with 95 GB of memory.

```bash
pip install -e ".[train]"

# 1. learn to read: generated questions whose answers can only come from the series
python scripts/train.py --stage locate --mixture exact:0.70,values:0.30 \
    --model LiquidAI/LFM3-1.7B --encoder-attention-layers 2 \
    --reconstruction-weight 1.0 --max-tokens 6000 --grad-accum 1 --output-dir runs/read

# 2. read long series: the same questions, up to 2,048 points
python scripts/train.py --stage locate --mixture exact:0.70,values:0.30 \
    --model runs/read/final --reading-max-length 2048 \
    --reconstruction-weight 1.0 --max-tokens 6000 --grad-accum 1 --max-steps 2000 --output-dir runs/long

# 3. learn to answer
python scripts/train.py --stage align --backbone full \
    --model runs/long/final --reading-max-length 2048 \
    --mixture align_256:0.15,sft:0.18,ift:0.10,align_random:0.05,exact:0.07,values:0.07,verdict:0.05,chat:0.15,real:0.10,injected:0.08 \
    --chat-data chat_replay.jsonl --real-data real_train.jsonl \
    --reconstruction-weight 1.0 --max-tokens 8000 --grad-accum 1 --max-steps 20000 --output-dir runs/joint
```

| Data of stage 3 | Share |
|---|---|
| [ChatTS synthetic data](https://huggingface.co/datasets/ChatTSRepo/ChatTS-Training-Dataset) | 48% |
| Generated reading questions | 19% |
| Plain chat (prompts of [dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), answered by LFM3-1.7B) | 15% |
| Real series from TSB-AD-U, with labelled anomalies | 10% |
| Real series with an event put in: an anomaly once, normal when it recurs | 8% |

The two files of stage 3 are built with:

```bash
python scripts/make_chat_replay.py --out chat_replay.jsonl
python scripts/make_real_data.py windows --dataset <TSB-AD-U folder> --out real
python scripts/make_real_data.py build --windows real --out real_train.jsonl
```

Tests: `pip install -e ".[dev,train]" && pytest -m "not slow"`.

## Data and licences

No dataset is redistributed here. The real training windows come from these sources of TSB-AD-U:

| Source | Windows | Licence |
|---|---|---|
| WSD | 1,113 | none stated |
| SVDB, LTDB, MITDB ([PhysioNet](https://physionet.org/)) | 919 | Open Data Commons Attribution v1.0 |
| SED | 104 | none stated |
| SMD | 103 | MIT |
| OPPORTUNITY, Daphnet (UCI) | 33 | CC BY 4.0 |
| Exathlon | 26 | CC BY-NC-SA 4.0 |
| CATSv2, Power | 2 | CC BY 4.0, none stated |

## Licence

LFM3-TS is released under the [LFM Open License v1.0](LICENSE), the licence of LFM3-1.7B, which the weights derive
from. See [NOTICE](NOTICE) for what was changed. Parts of the code are adapted from Hugging Face Transformers
(Apache License 2.0).

## Credits

[LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) (Liquid AI) ·
[ChatTS](https://github.com/NetManAIOps/ChatTS) (Xie et al., method, training data, evaluation sets and the model
compared with) · [TSB-AD](https://github.com/TheDatumOrg/TSB-AD) (Liu and Paparrizos, the real series) ·
[databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) ·
[Hugging Face Transformers](https://github.com/huggingface/transformers)

Author: Nardi Xhepi.
