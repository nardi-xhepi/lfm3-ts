"""Runs LFM3-TS and its baselines on the same held-out questions, and compares them.

    python scripts/baseline.py questions --set values --out questions.jsonl
    python scripts/baseline.py run ours --model nardi/LFM3-TS --questions questions.jsonl --out answers/lfm3-ts.jsonl
    python scripts/baseline.py score questions.jsonl answers/*.jsonl

The baselines:

    ours        an LFM3-TS checkpoint (`--model` is required)
    lfm3-text   LFM3-1.7B, the series written as numbers in the prompt
    lfm3-plot   LFM3-VL-2B, the series drawn as a line chart
    chatts      ChatTS-8B (`--model bytedance-research/ChatTS-14B` for the larger one)
    opentslm    an OpenTSLM checkpoint
    claude      a Claude model through the Anthropic API, given the numbers, the chart, or both (`--input`)

`run` appends one answer per line and skips the questions already answered, so an interrupted run continues where it
stopped. ChatTS and OpenTSLM need library versions that conflict with this project's: `run` imports nothing from
`lfm3_ts` unless the baseline is `ours`, so it can be started from another Python environment.

The models that were not trained on these questions may reason first and are told to end with a line of a given
form, because the score reads numbers and direction words out of it. They are also given the mean and standard
deviation of each series, which LFM3-TS reads from the header its processor writes.
"""

import argparse
import base64
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np


# The form of each `ReadingTasks` answer, shown to the models that were not trained on them.
ANSWER_FORMS = {
    "block means": '"The standardized block means are: A, B, C." with one number per block, rounded to one decimal',
    "spike": '"There is an upward spike at point N." or "There is a downward spike at point N."',
    "level shift": '"The level shifts upward at point N." or "The level shifts downward at point N."',
    "trend": (
        '"From point 0 to point N, there is an increasing trend." with one such sentence per segment of the series, '
        'each saying "increasing" or "decreasing"'
    ),
    "period": '"Each fluctuation period is approximately N points."',
    "extremum": '"The series reaches its maximum at point N." or "The series reaches its minimum at point N."',
    "spike count": '"There are K spikes, at points A, B, C." or "There are no spikes."',
}
ANSWER_MARKER = "ANSWER:"
# Raised when the instructions change: `run` then discards the answers given under the old ones.
PROMPT_VERSION = 2
DEFAULT_MODELS = {
    "lfm3-text": "LiquidAI/LFM3-1.7B",
    "lfm3-plot": "LiquidAI/LFM3-VL-2B",
    "chatts": "bytedance-research/ChatTS-8B",
    "opentslm": "OpenTSLM/llama-3.2-1b-m4-sp",
    "claude": "claude-opus-5-5",
}


def all_series(question: dict) -> list[np.ndarray]:
    return [np.asarray(part["series"], dtype=np.float64) for part in question["parts"] if "series" in part]


def render(question: dict, series_text) -> str:
    """The question as one string; `series_text(index, series)` gives what stands for each series."""
    index, pieces = 0, []
    for part in question["parts"]:
        if "series" in part:
            pieces.append(series_text(index, np.asarray(part["series"], dtype=np.float64)))
            index += 1
        else:
            pieces.append(part["text"])
    return "".join(pieces)


def instructions(question: dict) -> str:
    """What a model that was not trained on the questions needs to answer in a form the score can read. A ChatTS
    evaluation record states its own answer format and is asked as it is."""
    if question.get("task") == "chatts":
        return ""
    lines = ["Points are numbered from 0."]
    series = all_series(question)
    for index, values in enumerate(series):
        name = "The series" if len(series) == 1 else f"Series {index + 1}"
        lines.append(f"{name} has mean {values.mean():.6g} and standard deviation {values.std():.6g}.")
    if "standardized" in render(question, lambda *_: ""):
        lines.append("A standardized value is the value minus the mean, divided by the standard deviation.")
    if question.get("format"):
        lines.append(
            f"You may reason first. Finish with a last line that starts with {ANSWER_MARKER} followed by the answer "
            f"in exactly this form: {question['format']}"
        )
    return " ".join(lines)


def values_text(series: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.4g}" for value in series) + "]"


def plot_png(series: list[np.ndarray]) -> bytes:
    """A line chart of the series, one panel each, with a grid fine enough to read positions from."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    figure, axes = plt.subplots(len(series), 1, figsize=(12, 4.5 * len(series)), dpi=100, squeeze=False)
    for index, (axis, values) in enumerate(zip(axes[:, 0], series)):
        axis.plot(np.arange(len(values)), values, color="#1f4e9c", linewidth=1.5)
        axis.set_xlim(0, len(values) - 1)
        axis.xaxis.set_minor_locator(AutoMinorLocator(5))
        axis.grid(True, which="major", color="#b0b0b0", linewidth=0.8)
        axis.grid(True, which="minor", color="#e0e0e0", linewidth=0.5)
        axis.set_xlabel("point")
        axis.set_ylabel("value" if len(series) == 1 else f"series {index + 1}")
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    return buffer.getvalue()


def strip_thinking(text: str) -> str:
    """Drops a `<think>...</think>` block; an unclosed one means the answer was never reached."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.split("<think>")[0].strip()


ANSWER_PATTERN = re.compile(r"(?:ANSWER|the answer is)\s*:\s*", re.IGNORECASE)


def final_answer(text: str) -> str:
    """What follows the last answer marker, or the last "the answer is:", which some models write in its place.
    Without either, a short reply is taken as the answer itself, and a long one as reasoning that never concluded."""
    pieces = ANSWER_PATTERN.split(text)
    if len(pieces) > 1:
        return pieces[-1].strip().strip('"')
    return text.strip() if len(text) <= 300 else ""


def torch_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Ours:
    """An LFM3-TS checkpoint, prompted as in training: no instructions unless `--hint`."""

    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        import lfm3_ts  # noqa: F401  registers the LFM3-TS classes

        if not args.model:
            raise SystemExit("`run ours` needs --model <checkpoint directory>")
        self.torch, self.hint, self.max_new_tokens = torch, args.hint, args.max_new_tokens
        self.instructed = args.hint
        self.model = AutoModelForCausalLM.from_pretrained(args.model).to(torch_device()).eval()
        self.processor = AutoProcessor.from_pretrained(args.model)

    def answer(self, question: dict) -> str:
        content = [
            {"type": "timeseries", "timeseries": np.asarray(part["series"])} if "series" in part else {"type": "text", **part}
            for part in question["parts"]
        ]
        if self.hint:
            content.append({"type": "text", "text": " " + instructions(question)})
        return self.generate([{"role": "user", "content": content}], self.max_new_tokens)

    def generate(self, messages: list[dict], max_new_tokens: int) -> str:
        device = self.model.device
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(device)
        with self.torch.no_grad(), self.torch.autocast(device.type, dtype=self.torch.bfloat16, enabled=device.type == "cuda"):
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


class Lfm3Text:
    """LFM3-1.7B without a time-series encoder: the values are written in the prompt."""

    instructed = True

    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.thinking = torch, args.thinking
        self.max_new_tokens = args.max_new_tokens + (2048 if args.thinking else 0)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(torch_device()).eval()

    def answer(self, question: dict) -> str:
        prompt = render(question, lambda _, series: values_text(series)) + "\n" + instructions(question)
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=self.thinking,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return strip_thinking(self.tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))


class Lfm3Plot:
    """LFM3-VL-2B looking at a line chart of the series."""

    instructed = True

    def __init__(self, args):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch, self.thinking = torch, args.thinking
        self.max_new_tokens = args.max_new_tokens + (2048 if args.thinking else 0)
        # the model card loads both with the repository's own processing code
        self.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True).to(torch_device()).eval()

    def answer(self, question: dict) -> str:
        from PIL import Image

        image = Image.open(io.BytesIO(plot_png(all_series(question)))).convert("RGB")
        prompt = render(question, lambda *_: "(shown in the chart)") + "\n" + instructions(question)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=self.thinking,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return strip_thinking(self.processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))


class ChatTS:
    """ChatTS-8B or ChatTS-14B, called as on their model cards. Needs the transformers version they were saved with."""

    instructed = True

    TEMPLATE = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{prompt}<|im_end|>"
        "<|im_start|>assistant\n"
    )

    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        self.torch, self.max_new_tokens = torch, args.max_new_tokens
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, device_map="auto", torch_dtype="float16"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, tokenizer=self.tokenizer)

    def answer(self, question: dict) -> str:
        return self.answer_batch([question])[0]

    def answer_batch(self, questions: list[dict]) -> list[str]:
        """Several questions in one generation: the processor pads the prompts on the left and takes the series of
        all of them in one list, in the order of their placeholders."""
        prompts = [
            self.TEMPLATE.format(prompt="\n".join(filter(None, [render(question, lambda *_: "<ts><ts/>"), instructions(question)])))
            for question in questions
        ]
        series = [values for question in questions for values in all_series(question)]
        inputs = self.processor(text=prompts, timeseries=series, padding=True, return_tensors="pt")
        inputs = {name: value.to(self.model.device) for name, value in inputs.items()}
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1] :]
        return [strip_thinking(self.tokenizer.decode(row, skip_special_tokens=True)) for row in generated]


class OpenTslm:
    """An OpenTSLM checkpoint, prompted like its M4 captioning stage: a standardized series next to its statistics."""

    instructed = True

    def __init__(self, args):
        from opentslm import OpenTSLM
        from opentslm.prompt.full_prompt import FullPrompt
        from opentslm.prompt.text_prompt import TextPrompt
        from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt

        self.prompt_types = FullPrompt, TextPrompt, TextTimeSeriesPrompt
        self.max_new_tokens = args.max_new_tokens
        self.model = OpenTSLM.load_pretrained(args.model, device=torch_device())

    @staticmethod
    def split(question: dict) -> tuple[str, list[tuple[str, list[float]]], str]:
        """The text before the first series, each series with the text introducing it, and the text after the last."""
        before, series, pending = "", [], ""
        for part in question["parts"]:
            if "series" not in part:
                pending += part["text"]
                continue
            values = np.asarray(part["series"], dtype=np.float64)
            scale = values.std() or 1.0
            label = f"This is the time series, it has mean {values.mean():.4f} and std {values.std():.4f}:"
            if series:
                label = pending.strip() + " " + label
            else:
                before = pending
            series.append((label.strip(), ((values - values.mean()) / scale).tolist()))
            pending = ""
        after = pending.strip(" .")
        if after and after[-1] not in "?!":
            after += "."
        return before.strip(), series, (after + " " + instructions(question)).strip()

    def answer(self, question: dict) -> str:
        full_prompt, text_prompt, series_prompt = self.prompt_types
        before, series, after = self.split(question)
        prompt = full_prompt(
            text_prompt(before or "You are an expert in time series analysis."),
            [series_prompt(label, values) for label, values in series],
            text_prompt(after),
        )
        output = self.model.eval_prompt(prompt, max_new_tokens=self.max_new_tokens)
        return strip_thinking(output if isinstance(output, str) else output[0])


class Claude:
    """A Claude model through the Anthropic API. Reads the key from ANTHROPIC_API_KEY."""

    instructed = True

    def __init__(self, args):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=5)
        self.model, self.input = args.model, args.input

    def answer(self, question: dict) -> str:
        content = []
        if self.input in ("plot", "both"):
            data = base64.standard_b64encode(plot_png(all_series(question))).decode()
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
        if self.input == "plot":
            shown = lambda *_: "(shown in the chart)"
        elif self.input == "both":
            shown = lambda _, series: values_text(series) + " (also shown in the chart)"
        else:
            shown = lambda _, series: values_text(series)
        content.append({"type": "text", "text": render(question, shown) + "\n" + instructions(question)})
        response = self.client.messages.create(
            model=self.model, max_tokens=16000, messages=[{"role": "user", "content": content}]
        )
        if response.stop_reason == "refusal":
            return ""
        return "".join(block.text for block in response.content if block.type == "text").strip()


BASELINES = {
    "ours": Ours,
    "lfm3-text": Lfm3Text,
    "lfm3-plot": Lfm3Plot,
    "chatts": ChatTS,
    "opentslm": OpenTslm,
    "claude": Claude,
}


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


# The spike test: what each group of questions is called, and the `ReadingTasks` task that makes it.
SPIKE_GROUPS = {
    "one spike, flat series": "spike",
    "one spike, moving series": "spike_on_signal",
    "several spikes": "spikes_anywhere",
}
_SPIKE_TEST_SEED = 2_000_003


def spike_examples(count: int) -> list[tuple[str, dict]]:
    """`count` spike questions in equal groups, from a seed neither training nor the reading questions use."""
    from lfm3_ts.training import ReadingTasks

    tasks, rng = ReadingTasks(), np.random.default_rng(_SPIKE_TEST_SEED)
    groups = list(SPIKE_GROUPS)
    return [(groups[index % len(groups)], tasks.named(rng, SPIKE_GROUPS[groups[index % len(groups)]])) for index in range(count)]


CHATTS_PLACEHOLDER = "<ts><ts/>"
_NORMALITY_SEED = 3_000_017
NORMALITY_GROUPS = ("nothing but noise", "one spike", "one bump", "one level shift", "one spike under the threshold")


def normality_questions(count: int) -> list[dict]:
    """Series whose content is known, asked in the words of a ChatTS question: is this normal, given a threshold?

    The threshold is ten times the noise level: noise stays under it even measured from its lowest to its highest
    point, which is about six times its level. Every feature is at least thirty times the noise level. So a series
    of noise is normal and any other is not, whatever one calls a fluctuation. The last group has a spike of nine
    times the noise level under a threshold of twenty: a model that answers "not normal" whenever it sees a feature,
    without comparing it with the threshold, gets these wrong. A model that has learned to answer from the noise level alone, as some ChatTS references do, calls
    all of them normal."""
    rng, questions = np.random.default_rng(_NORMALITY_SEED), []
    for index in range(count):
        group = NORMALITY_GROUPS[index % len(NORMALITY_GROUPS)]
        length = int(rng.integers(128, 513))
        series = rng.normal(size=length)
        height = rng.choice([-1, 1]) * rng.uniform(30, 90)
        point = int(rng.integers(32, length - 32))
        factor = 10
        if group == "one spike under the threshold":
            series[point] += np.sign(height) * rng.uniform(8, 10)
            factor = 20
        elif group == "one spike":
            series[point] += height
        elif group == "one bump":
            # a slow rise and a sudden fall
            width = int(rng.integers(8, 25))
            series[point - width : point] += height * np.arange(1, width + 1) / width
        elif group == "one level shift":
            series[point:] += height
        scale = 10 ** rng.uniform(-1, 4)
        offset = rng.choice([-1, 1]) * scale * 10 ** rng.uniform(0, 5)
        threshold = float(f"{factor * scale:.3g}")
        text = (
            f". Given that any fluctuation greater than {threshold:g} is considered abnormal, should the time series "
            "be considered normal?"
        )
        parts = [
            {"text": f"This is a metric with length of {length}: "},
            {"series": (series * scale + offset).tolist()},
            {"text": text},
        ]
        normal = group in ("nothing but noise", "one spike under the threshold")
        questions.append(
            {
                "id": index,
                "task": "normality",
                "group": group,
                "parts": parts,
                "reference": "Yes." if normal else "No.",
                "format": '"Yes" or "No", then one sentence of explanation',
            }
        )
    return questions


def score_normality(reference: str, said: str) -> dict[str, float]:
    """Whether a reply starts with the right yes or no."""
    word = re.sub(r"[^a-z]", " ", said.lower()).split()[:1]
    answered = word in (["yes"], ["no"])
    return {"answered": float(answered), "right": float(answered and word == [reference.lower().rstrip(".")])}


_VALUES_SEED = 4_000_037
VALUE_FORMS = {
    "value": '"The value at point N is about X."',
    "extremes": '"Between point A and point B, the lowest value is about X at point P and the highest is about Y at point Q."',
}


def values_questions(count: int, real: str | None = None) -> list[dict]:
    """Questions answered with values of the series, in its own unit: the value at a point, and the lowest and
    highest value of a stretch with their places. Half are on generated series, half on the windows of real series
    of the file `real` (the questions `make_real_data.py` writes), where the answer is known as exactly: nobody had
    to label anything. Without `real`, all are on generated series."""
    from lfm3_ts.training import ReadingTasks, written_value

    rng, questions = np.random.default_rng(_VALUES_SEED), []
    windows = [np.asarray(part["series"], dtype=float) for q in (read_jsonl(real) if real else []) for part in q["parts"] if "series" in part]
    windows = [w for w in windows if np.isfinite(w).all() and w.std() > 0]
    for index in range(count):
        kind = ("value", "extremes")[index % 2]
        on_real = bool(windows) and index % 4 >= 2
        if on_real:
            series = windows[int(rng.integers(len(windows)))]
        else:
            length = int(rng.integers(64, 513))
            shape = ReadingTasks.background(rng, length) if rng.random() < 0.5 else np.cumsum(rng.normal(size=length))
            series, _ = ReadingTasks.real_units(rng, shape + rng.normal(scale=0.2, size=length))
        spread, length = float(series.std()), len(series)
        number = lambda value: written_value(value, spread)
        if kind == "value":
            point = int(rng.integers(0, length))
            text, targets = f"What is the value at point {point}?", {"value": float(series[point]), "point": point}
            reference = f"The value at point {point} is about {number(series[point])}."
        else:
            first = int(rng.integers(0, length - 16))
            last = int(rng.integers(first + 8, min(length, first + 129))) - 1
            low, high = first + int(series[first : last + 1].argmin()), first + int(series[first : last + 1].argmax())
            text = f"What are the lowest and the highest value between point {first} and point {last}, and where are they?"
            targets = {"low": float(series[low]), "low point": low, "high": float(series[high]), "high point": high, "stretch": [first, last]}
            reference = (f"Between point {first} and point {last}, the lowest value is about {number(series[low])} at point {low} "
                         f"and the highest is about {number(series[high])} at point {high}.")
        questions.append({
            "id": index, "task": "values", "group": f"{'real' if on_real else 'generated'} series, {kind}",
            "parts": [{"text": f"There is a time series of length {length}: "}, {"series": [float(v) for v in series]}, {"text": f". {text}"}],
            "reference": reference, "format": VALUE_FORMS[kind], "targets": targets, "spread": spread,
        })
    return questions


def score_values(question: dict, said: str) -> dict[str, float]:
    """How far the values of a reply are from those of the series, in standard deviations of the series, and
    whether the points it names are the right ones. A value counts as read when it is within a quarter of a
    standard deviation, and as read closely within a tenth, the precision the reference is written at."""
    targets, spread = question["targets"], question["spread"]
    number = r"(-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    as_float = lambda text: float(text.replace(",", "").rstrip("."))
    scores = {}
    if "value" in targets:
        found = re.findall(number, re.sub(r"point\s+\d+", " ", said))
        if not found:
            return {"answered": 0.0, "value 0.25 sd right": 0.0, "value 0.1 sd right": 0.0}
        error = abs(as_float(found[-1]) - targets["value"]) / spread
        return {"answered": 1.0, "value 0.25 sd right": float(error <= 0.25), "value 0.1 sd right": float(error <= 0.1), "value error (sd)": error}
    for name in ("low", "high"):
        match = re.search(rf"{name}est[^.]*?{number}\D+?point\s+(\d+)", said, flags=re.I | re.S)
        if not match:
            scores.update({f"{name} value right": 0.0, f"{name} point right": 0.0})
            continue
        error = abs(as_float(match.group(1)) - targets[name]) / spread
        scores[f"{name} value right"] = float(error <= 0.25)
        scores[f"{name} point right"] = float(abs(int(match.group(2)) - targets[f"{name} point"]) <= 2)
    scores["answered"] = float(bool(re.search(number, said)))
    return scores


def chatts_questions(path: str) -> list[dict]:
    """The ChatTS evaluation sets A and B (https://zenodo.org/records/14349206) as questions every baseline can
    read. Each record asks several numbered questions about its series; `id` is its index in the file, which is
    what the ChatTS scoring script calls `idx`."""
    questions = []
    for index, record in enumerate(json.loads(Path(path).read_text())):
        texts = record["question"].split(CHATTS_PLACEHOLDER)
        if len(texts) != len(record["timeseries"]) + 1:
            raise ValueError(f"record {index}: {len(texts) - 1} placeholders for {len(record['timeseries'])} series")
        parts = [{"text": texts[0]}]
        for series, text in zip(record["timeseries"], texts[1:]):
            parts += [{"series": list(series)}, {"text": text}]
        questions.append({"id": index, "task": "chatts", "parts": parts, "reference": record["answer"]})
    return questions


_TSB_SEED = 4_000_037
# sources of TSB-AD-U left out: generated series, and price or sensor series labelled at hundreds of single points
TSB_SKIPPED = {"MGAB", "TODS", "Stock", "TAO"}
TSB_QUESTION = (
    ". Is there an anomaly in this time series? Answer Yes or No first; if Yes, say around which point it is."
)


def tsb_questions(directory: str, count: int, length: int = 512) -> list[dict]:
    """Windows of real series from TSB-AD-U (https://github.com/TheDatumOrg/TSB-AD), half of them holding exactly one
    labelled anomaly and half holding none, in equal numbers per source. The labels were set by the people who
    published each source, so this is a test our question generator had no part in. A window with an anomaly is
    "clear" when the labelled stretch leaves the rest of the window by 4 of its standard deviations or more, and
    "subtle" otherwise: many labelled anomalies are a change of shape that no single value shows. A file is `Data,Label` with a
    name like `494_UCR_id_192_Facility_tr_22500_1st_72150.csv`; the second field is the source."""
    rng = np.random.default_rng(_TSB_SEED)
    by_source: dict[str, list[Path]] = {}
    for path in sorted(Path(directory).rglob("*.csv")):
        fields = path.stem.split("_")
        if len(fields) > 4 and fields[1] not in TSB_SKIPPED and fields[4] != "Synthetic":
            by_source.setdefault(fields[1], []).append(path)
    if not by_source:
        raise SystemExit(f"no TSB-AD-U files under {directory}")
    each = max(1, count // (2 * len(by_source)))
    questions = []
    for source, paths in sorted(by_source.items()):
        found = {"with an anomaly": [], "without an anomaly": []}
        for path in rng.permutation(paths):
            if all(len(windows) >= each for windows in found.values()):
                break
            table = np.genfromtxt(path, delimiter=",", skip_header=1)
            values, labelled = table[:, 0], table[:, -1] > 0
            if len(values) < 2 * length or np.isnan(values).any():
                continue
            edges = np.flatnonzero(np.diff(np.r_[0, labelled.astype(int), 0])).reshape(-1, 2)
            # at most two windows of each kind from one file
            for begin, end in rng.permutation(edges)[:2]:
                if end - begin > length // 4 or len(found["with an anomaly"]) >= each:
                    continue
                low, high = max(0, end - length + 32), min(begin - 32, len(values) - length)
                if low > high:
                    continue
                start = int(rng.integers(low, high + 1))
                # this anomaly and no other
                if labelled[start : start + length].sum() == end - begin:
                    found["with an anomaly"].append((path, start, [int(begin - start), int(end - 1 - start)]))
            for start in rng.integers(0, len(values) - length, size=20):
                if len(found["without an anomaly"]) >= each or sum(p == path for p, *_ in found["without an anomaly"]) >= 2:
                    break
                if not labelled[start : start + length].any():
                    found["without an anomaly"].append((path, int(start), None))
            for group, windows in found.items():
                for item in windows:
                    if len(item) == 3:
                        windows[windows.index(item)] = (*item, values[item[1] : item[1] + length].tolist())
        # as many windows without an anomaly as with one, source by source
        found["without an anomaly"] = found["without an anomaly"][: len(found["with an anomaly"])]
        for group, windows in found.items():
            for path, start, segment, series in windows:
                reference = "No." if segment is None else f"Yes, around points {segment[0]} to {segment[1]}."
                if segment:
                    inside = np.asarray(series[segment[0] : segment[1] + 1])
                    rest = np.asarray(series[: segment[0]] + series[segment[1] + 1 :])
                    clear = np.abs(inside - np.median(rest)).max() >= 4 * (rest.std() + 1e-12)
                    group = "with a clear anomaly" if clear else "with a subtle anomaly"
                questions.append(
                    {
                        "id": len(questions),
                        "task": "anomaly",
                        "group": group,
                        "source": source,
                        "file": path.name,
                        "start": start,
                        "segment": segment,
                        "parts": [{"text": f"This is a time series of length {length}: "}, {"series": series}, {"text": TSB_QUESTION}],
                        "reference": reference,
                        "format": '"Yes, around point N." or "No."',
                    }
                )
    return questions


_POINTS = re.compile(
    r"points?\s+(\d+(?:(?:\s*,\s*(?:and\s+|or\s+)?|\s+(?:to|and|through|or)\s+|\s*[-–]\s*)(?:points?\s+)?\d+)*)", flags=re.I
)


def named_points(said: str) -> list[int]:
    """The points a reply names: the numbers that follow the word "point" or "points", alone or in a list or a range
    ("point 301", "points 255 to 257", "points 68-70, 178-182, and 286-288"). The other numbers of a reply are
    values, durations or counts ("in the 44 points before") and name no place."""
    return [int(number) for match in _POINTS.finditer(said) for number in re.findall(r"\d+", match.group(1))]


def score_anomaly(question: dict, said: str, tolerance: int = 16) -> dict[str, float]:
    """Whether a reply starts with the right yes or no and, for a window with an anomaly, whether a point it names
    (`named_points`) lies in the labelled stretch, give or take `tolerance` points."""
    scores = score_normality(question["reference"].split(",")[0], said)
    if question["segment"]:
        low, high = question["segment"]
        points = named_points(said)
        scores["place right"] = float(scores["right"] and any(low - tolerance <= point <= high + tolerance for point in points))
    return scores


def export_chatts(args) -> list[dict]:
    """Writes a predictions file in the form the ChatTS scoring script reads (`evaluation/evaluate_qa.py` in
    https://github.com/NetManAIOps/ChatTS): a list of `{"idx", "response"}`."""
    answers = [{"idx": record["id"], "response": record["prediction"]} for record in read_jsonl(args.predictions)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(answers, ensure_ascii=False, indent=1))
    print(f"wrote {len(answers)} answers to {args.out}")
    return answers


def make_questions(args) -> list[dict]:
    """Writes held-out questions in a form every baseline can read: the reading questions `train.py` scores a run
    on, the larger spike test, or a ChatTS evaluation set."""
    if args.set in ("normality", "tsb", "values"):
        if args.set == "tsb" and not args.dataset:
            raise SystemExit(f"`questions --set {args.set}` needs --dataset <the folder of the dataset>")
        if args.set == "values":
            questions = values_questions(args.examples, args.dataset)
        else:
            questions = normality_questions(args.examples) if args.set == "normality" else tsb_questions(args.dataset, args.examples)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("".join(json.dumps(question) + "\n" for question in questions))
        print(f"wrote {len(questions)} questions to {args.out}")
        return questions
    if args.set == "chatts":
        if not args.dataset:
            raise SystemExit("`questions --set chatts` needs --dataset <dataset_a.json or dataset_b.json>")
        questions = chatts_questions(args.dataset)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("".join(json.dumps(question) + "\n" for question in questions))
        print(f"wrote {len(questions)} questions to {args.out}")
        return questions
    from lfm3_ts.training import load_chatts_holdout
    from train import READING_TASKS

    if args.set == "spikes":
        examples = spike_examples(args.examples)
    else:
        examples = [(None, example) for example in load_chatts_holdout({"reading": 1.0}, args.examples)]
    questions = []
    for index, (group, example) in enumerate(examples):
        user, answer = example["messages"]
        task = next(name for prefix, name in READING_TASKS.items() if answer["content"].startswith(prefix))
        parts = [
            {"series": np.asarray(part["timeseries"]).tolist()} if part["type"] == "timeseries" else {"text": part["text"]}
            for part in user["content"]
        ]
        question = {"id": index, "task": task, "parts": parts, "reference": answer["content"], "format": ANSWER_FORMS[task]}
        if group:
            question["group"] = group
        questions.append(question)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(question) + "\n" for question in questions))
    print(f"wrote {len(questions)} questions to {args.out}")
    return questions


def run(args, baseline=None) -> list[dict]:
    """Answers the questions not yet in `args.out`; `baseline` replaces the one `args.baseline` names, for tests."""
    questions = read_jsonl(args.questions)[: args.limit]
    # answers given under older instructions are dropped and asked again
    if baseline is not None:
        instructed = getattr(baseline, "instructed", False)
    else:
        instructed = args.baseline != "ours" or args.hint
    version = PROMPT_VERSION if instructed else 1
    kept = [record for record in read_jsonl(args.out) if record.get("version", 1) == version]
    if len(kept) != len(read_jsonl(args.out)):
        print(f"{args.baseline}: discarding {len(read_jsonl(args.out)) - len(kept)} answers given under older instructions")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(record) + "\n" for record in kept))
    done = {record["id"] for record in kept}
    todo = [question for question in questions if question["id"] not in done]
    print(f"{args.baseline}: {len(done)} of {len(questions)} questions already answered")
    if not todo:
        return read_jsonl(args.out)
    if baseline is None:
        baseline = BASELINES[args.baseline](args)
    # a baseline that answers several questions at once gets them in batches of similar length
    size = max(1, getattr(args, "batch_size", 1) or 1) if hasattr(baseline, "answer_batch") else 1
    if size > 1:
        todo = sorted(todo, key=lambda question: sum(len(series) for series in all_series(question)))
    count = 0
    with open(args.out, "a") as output:
        for begin in range(0, len(todo), size):
            batch = todo[begin : begin + size]
            start = time.time()
            try:
                answers = baseline.answer_batch(batch) if size > 1 else [baseline.answer(batch[0])]
            except Exception as error:  # a prompt too long for the model, or for the memory of the GPU
                if not getattr(args, "keep_going", False):
                    raise
                print(f"  no answer to question {batch[0]['id']}: {type(error).__name__}: {str(error)[:120]}", flush=True)
                answers = [""] * len(batch)
            seconds = round((time.time() - start) / len(batch), 2)
            for question, raw in zip(batch, answers):
                count += 1
                record = {
                    "id": question["id"],
                    "prediction": final_answer(raw) if instructed and question.get("task") != "chatts" else raw,
                    "raw": raw,
                    "seconds": seconds,
                    "version": PROMPT_VERSION if instructed else 1,
                }
                output.write(json.dumps(record) + "\n")
                if count <= args.show:
                    shown = record["prediction"][:300] or f"(no final answer) ...{raw[-200:]}"
                    print(f"\nTASK      {question['task']}\nREFERENCE {question['reference']}\nMODEL     {shown}")
            output.flush()
            if count > args.show and (count // 10 > (count - len(batch)) // 10):
                print(f"  {count}/{len(todo)}", flush=True)
    return sorted(read_jsonl(args.out), key=lambda record: record["id"])


def score(args) -> dict[str, dict]:
    """Scores each predictions file against the questions and prints the baselines side by side."""
    from train import reading_table, score_reading

    questions = {question["id"]: question for question in read_jsonl(args.questions)}
    tables = {}
    for path in args.predictions:
        name = Path(path).stem
        collected: dict[str, dict[str, list[float]]] = {}
        records = [record for record in read_jsonl(path) if record["id"] in questions]
        for record in records:
            question = questions[record["id"]]
            # read the answer out of the saved reply again: the rule for finding it may have improved since the run
            said = final_answer(record["raw"]) if record.get("version", 1) == PROMPT_VERSION else record["prediction"]
            if question["task"] == "normality":
                task, scores = None, score_normality(question["reference"], said)
            elif question["task"] == "anomaly":
                task, scores = None, score_anomaly(question, said)
            elif question["task"] == "values":
                task, scores = None, score_values(question, said)
            else:
                task, scores = score_reading(question["reference"], said)
            task = question.get("group", task)
            for metric, value in scores.items():
                collected.setdefault(task, {}).setdefault(metric, []).append(value)
        seconds = sorted(record["seconds"] for record in records)
        tables[name] = {
            "questions": len(records),
            "median seconds": seconds[len(seconds) // 2] if seconds else None,
            "scores": reading_table(collected, f"{name}: {len(records)} questions"),
        }

    rows = sorted({(task, metric) for table in tables.values() for task, metrics in table["scores"].items() for metric in metrics})
    width = max(12, *(len(name) for name in tables)) + 2
    print("\n" + f"{'':<34}" + "".join(f"{name:>{width}}" for name in tables))
    for task, metric in rows:
        cells = []
        for table in tables.values():
            value = table["scores"].get(task, {}).get(metric)
            flag = metric.endswith(("right", "answered"))
            cells.append("-" if value is None else f"{value:.0%}" if flag else f"{value:.2f}")
        print(f"{task + ': ' + metric:<34}" + "".join(f"{cell:>{width}}" for cell in cells))
    if args.out:
        Path(args.out).write_text(json.dumps(tables, indent=1))
    return tables


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    questions = commands.add_parser("questions", help="write the held-out questions")
    questions.add_argument("--out", required=True)
    questions.add_argument("--examples", type=int, default=100)
    questions.add_argument(
        "--set",
        choices=["reading", "spikes", "chatts", "normality", "tsb", "values"],
        default="reading",
        help="`spikes`: only spike questions, in three equal groups (flat series, moving series, several spikes); "
        "`chatts`: the records of a ChatTS evaluation set, given with --dataset; `normality`: series of known "
        "content asked whether they are normal, in equal groups; `tsb`: windows of the real series of TSB-AD-U, "
        "with and without a labelled anomaly, from the folder given with --dataset; `values`: the value at a point and "
        "the lowest and highest value of a stretch, half on generated series and half on the windows of the questions "
        "file given with --dataset",
    )
    questions.add_argument(
        "--dataset", help="`--set chatts`: dataset_a.json or dataset_b.json; `--set tsb`: the unpacked TSB-AD-U folder"
    )

    export = commands.add_parser("export", help="write predictions in the form the ChatTS scoring script reads")
    export.add_argument("predictions")
    export.add_argument("--out", required=True)

    runner = commands.add_parser("run", help="answer the questions with one baseline")
    runner.add_argument("baseline", choices=list(BASELINES))
    runner.add_argument("--questions", required=True)
    runner.add_argument("--out", required=True)
    runner.add_argument("--model", help="a checkpoint or model name; each baseline has a default except `ours`")
    runner.add_argument("--limit", type=int, help="answer only the first questions")
    runner.add_argument("--show", type=int, default=5, help="how many answers to print")
    runner.add_argument("--batch-size", type=int, default=1, help="chatts: questions answered per generation")
    runner.add_argument("--max-new-tokens", type=int, help="default: 160 for `ours`, 768 for the others")
    runner.add_argument("--keep-going", action="store_true", help="record an empty answer when a question fails")
    runner.add_argument("--thinking", action="store_true", help="lfm3-text, lfm3-plot: let the model reason first (slow)")
    runner.add_argument("--hint", action="store_true", help="ours: also give the instructions the baselines get")
    runner.add_argument("--input", choices=["text", "plot", "both"], default="both", help="claude: what it is given")

    scorer = commands.add_parser("score", help="compare predictions files")
    scorer.add_argument("questions")
    scorer.add_argument("predictions", nargs="+")
    scorer.add_argument("--out", help="write the tables to this JSON file")

    args = parser.parse_args(argv)
    if args.command == "run":
        args.model = args.model or DEFAULT_MODELS.get(args.baseline)
        args.max_new_tokens = args.max_new_tokens or (160 if args.baseline == "ours" else 768)
    return args


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).parent))
    args = parse_args(argv)
    return {"questions": make_questions, "run": run, "score": score, "export": export_chatts}[args.command](args)


if __name__ == "__main__":
    main()
