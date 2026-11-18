# Copyright 2026 Nardi Xhepi.
#
# This file is part of LFM3-TS, distributed under the LFM Open License v1.0: see LICENSE and NOTICE at the root
# of the repository.
"""Training data utilities for LFM3-TS: the ChatTS record format, a mixed stream of datasets, and a collator."""

import itertools
import json
import logging
import random
import re
from dataclasses import dataclass

import numpy as np
import torch

from ..processing_lfm3_ts import Lfm3TsProcessor


logger = logging.getLogger(__name__)

CHATTS_PLACEHOLDER = "<ts><ts/>"
CHATTS_DATASET = "ChatTSRepo/ChatTS-Training-Dataset"


_STATED_LENGTH = re.compile(r"(length(?: of(?: the time series is)?)? )(\d+)")


def fix_stated_lengths(text: str, series: list) -> str:
    """Corrects series lengths that the ChatTS data states at twice their real value.

    Every `ift` question and about half the "The length of the time series is N" answers give 2N for series of N
    points, while the positions they mention count real points. Only an exact doubling of the one length all the
    record's series share is rewritten."""
    lengths = {len(item) for item in series}
    if len(lengths) != 1:
        return text
    (length,) = lengths
    return _STATED_LENGTH.sub(lambda m: m.group(1) + (str(length) if int(m.group(2)) == 2 * length else m.group(2)), text)


def chatts_record_to_messages(record: dict, fix_lengths: bool = True) -> list[dict]:
    """Turns a ChatTS record (`input` with `<ts><ts/>` placeholders, `timeseries`, `output`) into a conversation.
    `fix_lengths` applies `fix_stated_lengths` to the question and the answer."""
    series = record["timeseries"]
    if fix_lengths:
        record = {
            **record,
            "input": fix_stated_lengths(record["input"], series),
            "output": fix_stated_lengths(record["output"], series),
        }
    pieces = record["input"].split(CHATTS_PLACEHOLDER)
    if len(pieces) - 1 != len(series):
        raise ValueError(f"Record has {len(pieces) - 1} placeholders but {len(series)} time series.")

    content = []
    for index, piece in enumerate(pieces):
        if piece:
            content.append({"type": "text", "text": piece})
        if index < len(series):
            content.append({"type": "timeseries", "timeseries": np.asarray(series[index], dtype=np.float64)})
    return [{"role": "user", "content": content}, {"role": "assistant", "content": record["output"]}]


class MixedStream(torch.utils.data.IterableDataset):
    """An endless stream of conversations drawn from several datasets with fixed probabilities.

    Args:
        datasets (`list`):
            Iterables of ChatTS records. An exhausted dataset restarts; one that is empty is skipped.
        probabilities (`list[float]`):
            The probability of drawing the next example from each dataset.
        seed (`int`, *optional*, defaults to 0):
            The seed of the draw.
    """

    def __init__(self, datasets: list, probabilities: list[float], seed: int = 0):
        if len(datasets) != len(probabilities):
            raise ValueError("`datasets` and `probabilities` must have the same length.")
        self.datasets = datasets
        self.probabilities = probabilities
        self.seed = seed

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (worker.id if worker else 0))
        iterators = [iter(dataset) for dataset in self.datasets]
        weights = list(self.probabilities)
        indices = range(len(self.datasets))
        while any(weights):
            index = rng.choices(indices, weights=weights)[0]
            try:
                record = next(iterators[index])
            except StopIteration:
                iterators[index] = iter(self.datasets[index])
                record = next(iterators[index], None)
                if record is None:
                    # a streamed dataset splits its shards between dataloader workers, and a single-file dataset
                    # leaves every worker but one with nothing: stop drawing from it here
                    weights[index] = 0
                    continue
            yield record if "messages" in record else {"messages": chatts_record_to_messages(record)}


class ScheduledStream(torch.utils.data.IterableDataset):
    """Yields the first `switch_after` examples from one stream, then the rest from another.

    A model has to read the series roughly before it can learn anything fine about them, and the exercises that
    teach the first drown out the ones that teach the second: this trains on one mixture, then on the other, within
    a single run.

    Args:
        first (iterable):
            The stream of the beginning.
        second (iterable):
            The stream of the rest of the run.
        switch_after (`int`):
            How many examples come from `first`, over all dataloader workers together.
    """

    def __init__(self, first, second, switch_after: int):
        self.first = first
        self.second = second
        self.switch_after = switch_after

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        yield from itertools.islice(self.first, self.switch_after // (worker.num_workers if worker else 1))
        yield from self.second


class ReadingTasks(torch.utils.data.IterableDataset):
    """An endless stream of generated questions that cannot be answered without reading the series.

    The ChatTS answers are long and mostly phrasing, which a language model learns to produce before it learns to
    read: only a few of their tokens depend on the series. Here the answers state what appears nowhere in the
    prompt's text, so the whole loss pushes the model to use the time-series tokens.

    Half the questions ask for the mean of every block of `block_size` points, in standardized units. With
    `block_size` equal to the model's patch size, each number of the answer is the content of one time-series
    token: every token gets its own target, where a question about one event (a spike, a shift) teaches one token
    per example and leaves the others guessing. The other half ask for a position, a direction or a period.

    Args:
        seed (`int`, *optional*, defaults to 0):
            The seed of the generator.
        min_length (`int`, *optional*, defaults to 64):
            The shortest series.
        max_length (`int`, *optional*, defaults to 512):
            The longest series.
        block_size (`int`, *optional*, defaults to 8):
            The block of the block-mean questions; set it to the model's patch size.
        profile (`str`, *optional*, defaults to `"balanced"`):
            `"locate"` weights the stream toward questions answered with a position: spikes on a flat or a moving
            background, several spikes anywhere, and the value at a given point. `"balanced"` is the original
            stream, which the held-out questions are drawn from. `"exact"` is a single stage for a model that has
            never seen a series: questions with one target per time-series token (the mean of each block, the place
            of the highest value inside each block, its point) next to the questions about one event.
    """

    # the share of each `locate` question
    LOCATE = {
        "block_peaks": 0.22,
        "spike_blocks": 0.16,
        "spike_steps": 0.12,
        "block_means": 0.06,
        "spike": 0.08,
        "spike_on_signal": 0.10,
        "spikes_anywhere": 0.10,
        "value_at": 0.06,
        "level_shift": 0.04,
        "trend": 0.02,
        "period": 0.02,
        "extremum": 0.02,
    }

    # the share of each `exact` question. The first three have one target per time-series token: what the token
    # holds, where inside the token, and where in the series. Block means are what makes a model start to read; a
    # quarter of the stream, as in the mixtures that did.
    EXACT = {
        "block_means": 0.25,
        "block_places": 0.22,
        "block_peaks": 0.12,
        "spike_steps": 0.10,
        "spike": 0.05,
        "spike_on_signal": 0.07,
        "spikes_anywhere": 0.07,
        "value_at": 0.04,
        "level_shift": 0.02,
        "trend": 0.02,
        "period": 0.02,
        "extremum": 0.02,
    }
    # questions asking for a verdict on the whole series, answered from what the series holds
    VERDICT = {"normality": 0.5, "sudden_change": 0.5}

    def __init__(
        self, seed: int = 0, min_length: int = 64, max_length: int = 512, block_size: int = 8, profile: str = "balanced"
    ):
        self.PROFILES = {"locate": self.LOCATE, "exact": self.EXACT, "verdict": self.VERDICT, "values": self.VALUES}
        if profile != "balanced" and profile not in self.PROFILES:
            raise ValueError(f'`profile` must be "balanced", "locate", "exact", "verdict" or "values", got {profile!r}.')
        self.seed, self.min_length, self.max_length, self.block_size = seed, min_length, max_length, block_size
        self.profile = profile
        self.tasks = [self.spike, self.level_shift, self.trend, self.period, self.extremum, self.spike_count]

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, worker.id if worker else 0])
        while True:
            yield self.sample(rng)

    def sample(self, rng: np.random.Generator) -> dict:
        if self.profile in self.PROFILES:
            shares = self.PROFILES[self.profile]
            name = rng.choice(list(shares), p=list(shares.values()))
            if name in self.VERDICT:
                return getattr(self, name)(rng, int(rng.integers(max(self.min_length, 160), self.max_length + 1)))
            if name == "real_block_means":
                # short series: an answer of 2 to 16 numbers, as for the standardized block means
                return self.real_block_means(rng, int(rng.integers(2 * self.block_size, 16 * self.block_size + 1)), self.block_size)
            if name in self.VALUES:
                return getattr(self, name)(rng, int(rng.integers(self.min_length, self.max_length + 1)))
            if name in ("block_means", "block_peaks", "block_places"):
                length = int(rng.integers(2 * self.block_size, 16 * self.block_size + 1))
            else:
                length = int(rng.integers(self.min_length, self.max_length + 1))
            series, question, answer = getattr(self, name)(rng, length)
        elif rng.random() < 0.5:
            # short series: an answer of 2 to 16 numbers
            length = int(rng.integers(2 * self.block_size, 16 * self.block_size + 1))
            series, question, answer = self.block_means(rng, length)
        else:
            length = int(rng.integers(self.min_length, self.max_length + 1))
            series, question, answer = self.tasks[rng.integers(len(self.tasks))](rng, length)
        return self.compose(rng, series, question, answer)

    def named(self, rng: np.random.Generator, name: str) -> dict:
        """One example of the task `name`, e.g. `"spike_on_signal"`: for test sets made of chosen tasks."""
        length = int(rng.integers(self.min_length, self.max_length + 1))
        return self.compose(rng, *getattr(self, name)(rng, length))

    @staticmethod
    def compose(rng, series, question, answer) -> dict:
        length = len(series)
        # any unit and level: the statistics in the prompt must not give the answer away
        series = series * 10 ** rng.uniform(-2, 4) + rng.normal() * 10 ** rng.uniform(-1, 4)
        content = [
            {"type": "text", "text": f"There is a time series of length {length}: "},
            {"type": "timeseries", "timeseries": series},
            {"type": "text", "text": f". {question}"},
        ]
        return {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": answer}]}

    # questions answered with values in the unit of the series, where the reading questions above answer in
    # standardized units: the model has to bring what a token holds back to the offset and the scale that the
    # prompt states. `real_block_means` has one such target for each time-series token.
    VALUES = {"real_block_means": 0.4, "real_value_at": 0.3, "real_extremes": 0.3}

    @staticmethod
    def real_units(rng, series):
        """`series` in a random unit and at a random level, and the function that writes one of its values the way
        the answers do: rounded to a step between a hundredth and a tenth of its standard deviation, which is
        about the precision a time-series token can be read at."""
        series = series * 10 ** rng.uniform(-2, 4) + rng.normal() * 10 ** rng.uniform(-1, 4)
        return series, lambda value: written_value(value, float(series.std()))

    @classmethod
    def real_block_means(cls, rng, length, block_size: int = 8) -> dict:
        """The mean of every block of `block_size` points, in the unit of the series."""
        series, number = cls.real_units(rng, np.cumsum(rng.normal(size=length)) + rng.normal(scale=0.2, size=length))
        means = [number(series[start : start + block_size].mean()) for start in range(0, length, block_size)]
        question = f"Give the mean of each block of {block_size} points, in the unit of the series."
        return cls.asked(series, question, f"The block means are about: {', '.join(means)}.")

    @classmethod
    def real_value_at(cls, rng, length) -> dict:
        """The value at one point, in the unit of the series."""
        shape = cls.background(rng, length) if rng.random() < 0.5 else np.cumsum(rng.normal(size=length))
        series, number = cls.real_units(rng, shape + rng.normal(scale=0.2, size=length))
        point = int(rng.integers(0, length))
        return cls.asked(series, f"What is the value at point {point}?", f"The value at point {point} is about {number(series[point])}.")

    @classmethod
    def real_extremes(cls, rng, length) -> dict:
        """The lowest and the highest value of a stretch, and where they are."""
        shape = cls.background(rng, length) if rng.random() < 0.5 else np.cumsum(rng.normal(size=length))
        series, number = cls.real_units(rng, shape + rng.normal(scale=0.2, size=length))
        first = int(rng.integers(0, length - 16))
        last = int(rng.integers(first + 8, min(length, first + 129))) - 1
        low, high = first + int(series[first : last + 1].argmin()), first + int(series[first : last + 1].argmax())
        question = f"What are the lowest and the highest value between point {first} and point {last}, and where are they?"
        answer = (f"Between point {first} and point {last}, the lowest value is about {number(series[low])} at point {low} "
                  f"and the highest is about {number(series[high])} at point {high}.")
        return cls.asked(series, question, answer)

    @staticmethod
    def asked(series, question: str, answer: str) -> dict:
        content = [
            {"type": "text", "text": f"There is a time series of length {len(series)}: "},
            {"type": "timeseries", "timeseries": series},
            {"type": "text", "text": f". {question}"},
        ]
        return {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": answer}]}

    @staticmethod
    def normality(rng, length) -> dict:
        """Is the series normal, given a threshold? The answer comes from the series: noise alone is normal, a
        spike, a bump or a level shift above the threshold is not, however low the noise, and the same feature
        under the threshold is normal again. Question-answer data written from a list of properties sometimes
        judges by the noise level alone; this teaches the model to look at the whole series, to compare what it
        finds with the threshold, and to say so when there is nothing but noise.

        The threshold is always above the span of the series when the verdict is "normal", from its lowest to its
        highest point, and a feature that breaks it is 2 to 8 times as large."""
        series = rng.normal(size=length)
        noise = float(series.std())
        kind = rng.choice(["noise", "spike", "bump", "shift"], p=[0.25, 0.25, 0.25, 0.25])
        # a third of the features stay under the threshold
        small = kind != "noise" and rng.random() < 0.35
        if small:
            ratio = rng.uniform(16, 26)
            height = ratio * rng.uniform(0.35, 0.55)
        else:
            ratio = rng.uniform(8, 15)
            height = ratio * rng.uniform(2, 8)
        sign = rng.choice([-1, 1])
        direction = "upward" if sign > 0 else "downward"
        point = int(rng.integers(16, length - 16))
        scale = 10 ** rng.uniform(-2, 4)
        offset = rng.normal() * 10 ** rng.uniform(-1, 6)
        # four significant digits, written out in full: 48400, not 4.84e+04
        number = lambda value: np.format_float_positional(float(f"{value * scale:.4g}"), trim="-")
        if kind == "spike":
            series[point] += sign * height
            found = f"There is {_a(direction)} spike at point {point} with an amplitude of about {number(height)}"
        elif kind == "bump":
            width = int(rng.integers(6, 15))
            series[point - width : point] += sign * height * np.arange(1, width + 1) / width
            slope = "rise" if sign > 0 else "decline"
            found = (
                f"There is a slow {slope} from point {point - width} to point {point - 1}, then a sudden return, "
                f"with an amplitude of about {number(height)}"
            )
        elif kind == "shift":
            series[point:] += sign * height
            found = f"The level shifts {direction} at point {point} by about {number(height)}"
        nothing = f"The series shows only noise, with a standard deviation of about {number(noise)}"
        threshold = number(ratio)
        wording = rng.integers(3)
        if wording == 0:
            question = (
                f"Given that any fluctuation greater than {threshold} is considered abnormal, should the time series "
                "be considered normal?"
            )
        elif wording == 1:
            question = f"Any change larger than {threshold} counts as an anomaly. Is this series normal?"
        else:
            question = f"Is there any fluctuation larger than {threshold} in this series?"
        normal = kind == "noise" or small
        if wording == 2:
            verdict = "No." if normal else "Yes."
        else:
            verdict = "Yes, it should be considered normal." if normal else "No, it should not be considered normal."
        if kind == "noise":
            answer = f"{verdict} {nothing}. No fluctuation exceeds {threshold}."
        elif small:
            answer = f"{verdict} {found}, which stays below {threshold}."
        else:
            answer = f"{verdict} {found}, which exceeds {threshold}."
        content = [
            {"type": "text", "text": f"There is a time series of length {length}: "},
            {"type": "timeseries", "timeseries": series * scale + offset},
            {"type": "text", "text": f". {question}"},
        ]
        return {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": answer}]}

    @staticmethod
    def sudden_change(rng, length) -> dict:
        """Is there a sudden change above a threshold? Here the series moves: a trend several thresholds long, often
        a seasonal swing, and noise, all of them smooth from one point to the next. On top come zero, one or two
        events (a spike, a level shift, or a slow drift that snaps back), each either well above the threshold or
        well under it. The answer lists every event with its place and size and says which ones break the
        threshold, so the model has to tell an event from the pattern it sits on, and measure it.

        From one point to the next the pattern and the noise move by less than 0.9 thresholds even with a small
        event; an event above the threshold moves by at least 1.6."""
        threshold = rng.uniform(16, 24)
        time = np.arange(length)
        series = rng.normal(size=length) + rng.uniform(-6, 6) * threshold * time / length
        period = None
        if rng.random() < 0.6:
            period = int(rng.integers(24, max(25, length // 3) + 1))
            # at most 0.05 thresholds from one point to the next
            amplitude = rng.uniform(0.3, 1.0) * 0.05 * threshold * period / (2 * np.pi)
            series += amplitude * np.sin(2 * np.pi * time / period + rng.uniform(0, 2 * np.pi))
        scale = 10 ** rng.uniform(-2, 4)
        offset = rng.normal() * 10 ** rng.uniform(-1, 6)
        number = lambda value: np.format_float_positional(float(f"{value * scale:.4g}"), trim="-")
        limit = number(threshold)

        count = int(rng.choice([0, 1, 2], p=[0.3, 0.45, 0.25]))
        slots = rng.permutation(np.arange(1, (length - 64) // 48 + 1))[:count] * 48 if length >= 160 else []
        events = []
        for slot in sorted(int(slot) for slot in slots):
            # anywhere, but at least 32 points from the next event
            point = slot + int(rng.integers(-8, 9))
            large = rng.random() < 0.65
            size = threshold * (rng.uniform(2, 6) if large else rng.uniform(0.35, 0.5))
            sign = rng.choice([-1, 1])
            direction = "upward" if sign > 0 else "downward"
            kind = rng.choice(["spike", "shift", "drift"])
            if kind == "spike":
                series[point] += sign * size
                text = f"{_a(direction)} spike at point {point} of about {number(size)}"
            elif kind == "shift":
                series[point:] += sign * size
                text = f"{_a(direction)} level shift at point {point} of about {number(size)}"
            else:
                # a drift of at most a quarter of the threshold per point, then back at once
                width = int(min(30, max(8, np.ceil(size / (0.25 * threshold)))))
                series[point - width : point] += sign * size * np.arange(1, width + 1) / width
                slope, back = ("rise", "drop") if sign > 0 else ("decline", "jump")
                text = (
                    f"a slow {slope} from point {point - width} to point {point - 1} followed by a sudden {back} "
                    f"of about {number(size)} at point {point}"
                )
            events.append((large, text))

        question = [
            f"Is there a sudden change larger than {limit} in this series?",
            f"Does the series jump by more than {limit} from one point to the next anywhere?",
            f"An alert fires on any abrupt change above {limit}. Would it fire here?",
        ][rng.integers(3)]
        fires = any(large for large, _ in events)
        if not events:
            pattern = f", with a period of about {period} points" if period else ""
            answer = f"No. The series changes smoothly{pattern}; nothing changes abruptly by more than {limit}."
        else:
            parts = [f"{text}, which {'exceeds' if large else 'stays below'} {limit}" for large, text in events]
            answer = f"{'Yes' if fires else 'No'}. There is {' There is also '.join(part + '.' for part in parts)}"
        content = [
            {"type": "text", "text": f"There is a time series of length {length}: "},
            {"type": "timeseries", "timeseries": series * scale + offset},
            {"type": "text", "text": f". {question}"},
        ]
        return {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": answer}]}

    def block_means(self, rng, length):
        series = np.cumsum(rng.normal(size=length)) + rng.normal(scale=0.2, size=length)
        standardized = (series - series.mean()) / series.std()
        means = [standardized[start : start + self.block_size].mean() for start in range(0, length, self.block_size)]
        # `+ 0.0` turns -0.0 into 0.0
        listed = ", ".join(f"{round(float(mean), 1) + 0.0:.1f}" for mean in means)
        question = (
            f"Give the mean of each block of {self.block_size} points, in standardized units "
            "(the series minus its offset, divided by its scale)."
        )
        return series, question, f"The standardized block means are: {listed}."

    @staticmethod
    def spike(rng, length):
        series = rng.normal(scale=0.3, size=length)
        point, sign = int(rng.integers(1, length - 1)), rng.choice([-1, 1])
        series[point] += sign * rng.uniform(6, 12)
        direction = "upward" if sign > 0 else "downward"
        question = "At which point is the spike, and is it upward or downward?"
        return series, question, f"There is {_a(direction)} spike at point {point}."

    def block_peaks(self, rng, length):
        """The point of the highest value of every block: what block means are to values, for positions. Each
        number of the answer is the index of one time-series token times the block size, plus the place of the
        peak inside it, so every token of every example teaches the model to name a position."""
        series = rng.normal(size=length) + 0.5 * self.background(rng, length)
        peaks = [
            start + int(series[start : start + self.block_size].argmax()) for start in range(0, length, self.block_size)
        ]
        question = f"For each block of {self.block_size} points, at which point is its highest value?"
        return series, question, f"The highest point of each block is at: {', '.join(map(str, peaks))}."

    def block_places(self, rng, length):
        """The place, from 0, of the highest value inside every block: one digit per time-series token, which says
        where inside the token and nothing else. A point is the number of a block times its size plus this place;
        `block_peaks` asks for the sum, where a model can be right about the block and vague about the place for a
        long time. In half the series every block holds a clear spike, as in the questions about one spike."""
        series = rng.normal(size=length) + 0.5 * self.background(rng, length)
        if rng.random() < 0.5:
            for start in range(0, length, self.block_size):
                place = int(rng.integers(0, min(self.block_size, length - start)))
                series[start + place] += rng.uniform(4, 8)
        places = [int(series[start : start + self.block_size].argmax()) for start in range(0, length, self.block_size)]
        question = (
            f"For each block of {self.block_size} points, at which place inside the block is its highest value? "
            "Places are numbered from 0."
        )
        return series, question, f"The place of the highest value in each block is: {', '.join(map(str, places))}."

    def spike_blocks(self, rng, length):
        """Which blocks hold a spike. A point is a block number times the block size plus a place inside the block,
        and a model that is one block off gains nothing from knowing the place: this asks for the block alone, many
        times per series, so that the number of a time-series token is learned for itself."""
        moving = rng.random() < 0.5
        series = self.background(rng, length) if moving else rng.normal(scale=0.3, size=length)
        blocks = length // self.block_size
        chosen = sorted(rng.choice(blocks, size=int(rng.integers(1, min(12, blocks) + 1)), replace=False).tolist())
        steps = series.copy()
        for block in chosen:
            # away from the edges of the block, so that two spikes never touch
            point = block * self.block_size + int(rng.integers(1, self.block_size - 1))
            series[point] += self.spike_height(rng, steps) if moving else rng.uniform(6, 12)
        question = f"Which blocks of {self.block_size} points hold an upward spike? Blocks are numbered from 0."
        return series, question, f"The blocks with a spike are: {', '.join(map(str, chosen))}."

    def spike_steps(self, rng, length):
        """One spike, located in steps: the block, the place inside it, then the point. Each part is scored for
        itself, where the point alone is wrong as a whole when the block is."""
        maker = self.spike_on_signal if rng.random() < 0.5 else self.spike
        series, _, answer = maker(rng, length)
        point = int(answer.rsplit(" ", 1)[1].rstrip("."))
        question = (
            f"In which block of {self.block_size} points is the spike, at which place inside the block, "
            "and at which point of the series? Blocks and places are numbered from 0."
        )
        block, place = divmod(point, self.block_size)
        return series, question, f"The spike is in block {block}, at place {place}. {answer}"

    @staticmethod
    def background(rng, length):
        """A smooth series that moves: slow waves and a drift, with a little noise."""
        time = np.arange(length) / length
        waves = sum(
            rng.normal() * np.sin(2 * np.pi * frequency * time + rng.uniform(0, 2 * np.pi)) for frequency in (0.5, 1, 2, 4)
        )
        return waves + rng.normal() * time + rng.normal(scale=0.05, size=length)

    @staticmethod
    def spike_height(rng, series):
        """Tall next to the steps the series takes anyway: a short series with fast waves is steep everywhere."""
        return rng.uniform(1.5, 4) + 10 * float(np.median(np.abs(np.diff(series))))

    @classmethod
    def spike_on_signal(cls, rng, length):
        """One spike on a moving series: it stands out from its neighbours, not from the whole series."""
        series = cls.background(rng, length)
        point, sign = int(rng.integers(1, length - 1)), rng.choice([-1, 1])
        series[point] += sign * cls.spike_height(rng, series)
        direction = "upward" if sign > 0 else "downward"
        question = "At which point is the spike, and is it upward or downward?"
        return series, question, f"There is {_a(direction)} spike at point {point}."

    @classmethod
    def spikes_anywhere(cls, rng, length):
        """Zero to five upward spikes at any points at least four apart, on a flat or a moving series."""
        moving = rng.random() < 0.5
        series = cls.background(rng, length) if moving else rng.normal(scale=0.3, size=length)
        count, points = int(rng.integers(0, 6)), []
        for candidate in rng.permutation(np.arange(1, length - 1)):
            if len(points) == count:
                break
            if all(abs(int(candidate) - point) >= 4 for point in points):
                points.append(int(candidate))
        points.sort()
        steps = series.copy()
        for point in points:
            series[point] += cls.spike_height(rng, steps) if moving else rng.uniform(6, 12)
        question = "How many upward spikes are there, and at which points?"
        if not points:
            return series, question, "There are no spikes."
        listed = ", ".join(str(point) for point in points)
        noun = "spike, at point" if len(points) == 1 else "spikes, at points"
        return series, question, f"There {'is' if len(points) == 1 else 'are'} {len(points)} {noun} {listed}."

    @staticmethod
    def value_at(rng, length):
        """The value at one point: the reverse of locating, from an index to what the series holds there."""
        series = np.cumsum(rng.normal(size=length)) + rng.normal(scale=0.2, size=length)
        standardized = (series - series.mean()) / series.std()
        point = int(rng.integers(0, length))
        question = (
            f"What is the value at point {point}, in standardized units "
            "(the series minus its offset, divided by its scale)?"
        )
        value = round(float(standardized[point]), 1) + 0.0
        return series, question, f"In standardized units, the value at point {point} is {value:.1f}."

    @staticmethod
    def level_shift(rng, length):
        series = rng.normal(scale=0.3, size=length)
        point, sign = int(rng.integers(8, length - 8)), rng.choice([-1, 1])
        series[point:] += sign * rng.uniform(3, 8)
        direction = "upward" if sign > 0 else "downward"
        question = "At which point does the level of the series shift, and in which direction?"
        return series, question, f"The level shifts {direction} at point {point}."

    @staticmethod
    def trend(rng, length):
        sign = rng.choice([-1, 1])
        names = {1: "increasing", -1: "decreasing"}
        steps = np.full(length, sign * rng.uniform(0.02, 0.1))
        question = "Analyze the trend of the series in detail."
        if rng.random() < 0.3:
            answer = f"From point 0 to point {length - 1}, there is {_a(names[sign])} trend."
        else:
            point = int(rng.integers(length // 6, length - length // 6))
            steps[point:] = -sign * rng.uniform(0.02, 0.1)
            answer = (
                f"From point 0 to point {point}, there is {_a(names[sign])} trend. "
                f"From point {point} to point {length - 1}, there is {_a(names[-sign])} trend."
            )
        series = np.concatenate([[0.0], np.cumsum(steps[1:])]) + rng.normal(scale=0.05, size=length)
        return series, question, answer

    @staticmethod
    def period(rng, length):
        period = int(rng.integers(6, max(7, length // 3)))
        phase = rng.uniform(0, 2 * np.pi)
        series = np.sin(2 * np.pi * np.arange(length) / period + phase) + rng.normal(scale=0.1, size=length)
        question = "What is the period of the fluctuation?"
        return series, question, f"Each fluctuation period is approximately {period} points."

    @staticmethod
    def extremum(rng, length):
        time = np.arange(length) / length
        series = sum(
            rng.normal() * np.sin(2 * np.pi * frequency * time + rng.uniform(0, 2 * np.pi)) for frequency in (0.5, 1, 2)
        )
        kind = rng.choice(["maximum", "minimum"])
        point = int(series.argmax() if kind == "maximum" else series.argmin())
        question = f"At which point does the series reach its {kind}?"
        return series, question, f"The series reaches its {kind} at point {point}."

    @staticmethod
    def spike_count(rng, length):
        series = rng.normal(scale=0.3, size=length)
        count = int(rng.integers(0, 5))
        # at least 8 points apart, away from the ends
        slots = rng.choice(np.arange(1, (length - 2) // 8), size=count, replace=False) * 8 if count else []
        points = sorted(int(slot) for slot in slots)
        for point in points:
            series[point] += rng.uniform(6, 12)
        question = "How many upward spikes are there, and at which points?"
        if not points:
            return series, question, "There are no spikes."
        listed = ", ".join(str(point) for point in points)
        noun = "spike, at point" if count == 1 else "spikes, at points"
        return series, question, f"There {'is' if count == 1 else 'are'} {count} {noun} {listed}."


def _a(word: str) -> str:
    return f"{'an' if word[0] in 'aeiou' else 'a'} {word}"


def written_value(value: float, spread: float) -> str:
    """A value of a series as an answer writes it: rounded to the power of ten between a hundredth and a tenth of
    the standard deviation `spread` of the series, and written out in full (48400, not 4.84e+04)."""
    digits = 1 - int(np.floor(np.log10(spread))) if spread > 0 else 0
    rounded = round(float(value), digits) + 0.0
    return f"{rounded:.{max(digits, 0)}f}"


READING = "reading"
LOCATE = "locate"
EXACT = "exact"
VERDICT = "verdict"
VALUES = "values"
# the mixture names of the generated questions, and the profile of `ReadingTasks` each stands for
GENERATED = {READING: "balanced", LOCATE: "locate", EXACT: "exact", VERDICT: "verdict", VALUES: "values"}
CHAT = "chat"
_CHAT_HOLDOUT = 50


class ChatReplay(torch.utils.data.IterableDataset):
    """An endless shuffled stream of ordinary conversations, without series, read from a JSON-lines file of
    `{"messages": [...]}` records.

    Training every weight on templated answers about series wears down what the language model could do before:
    its answers to plain questions turn short and its arithmetic goes. Mixing in conversations answered by the
    original model (see `scripts/make_chat_replay.py`) keeps that behaviour in the loss. The first records of the
    file are left out, for `load_chatts_holdout`."""

    def __init__(self, path: str, seed: int = 0):
        self.path, self.seed = str(path), seed
        with open(path) as lines:
            self.records = [json.loads(line) for line in lines if line.strip()][_CHAT_HOLDOUT:]
        if not self.records:
            raise ValueError(f"{path} holds no conversation beyond the {_CHAT_HOLDOUT} kept for the held-out set.")

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, worker.id if worker else 0])
        while True:
            for index in rng.permutation(len(self.records)):
                yield self.records[int(index)]


REAL = "real"


class RealSeries(ChatReplay):
    """An endless shuffled stream of questions on real series, read from the JSON-lines file that
    `scripts/make_real_data.py build` writes: the windows of public series with an anomaly somebody labelled, and
    as many without. The generated questions teach the model to read values; these teach it what real series
    look like, which shapes are ordinary in them and which are not."""

    def __iter__(self):
        for record in super().__iter__():
            yield {"messages": _with_arrays(record["messages"])}


INJECTED = "injected"


class InjectedAnomalies(ChatReplay):
    """An endless stream of real series with events put into them, where the same event is an anomaly or not
    according to the rest of the series, and every number of an answer can be read in the series.

    The windows are those of the file of questions on real series (`scripts/make_real_data.py build`) that have no
    labelled anomaly. An event is a spike, a drop of a few points, a shift of the level or a stretch of larger
    swings, sized against what that series already does: 1.5 to 5 times its largest step, change of level or swing.
    Each window becomes one of four cases (`CASES`):

    - "once": the event is put in at one place. The answer is "Yes", with the place and what the series does there.
    - "recurring": the same event is put in three to six times, evenly spaced, with about the same size. The answer
      is "No": what recurs is the pattern of the series.
    - "odd": the recurring events, and one that does not fit them (much larger, the other way, or of another
      kind). The answer is "Yes", at that one.
    - "untouched": the window as it is, with its "No".

    The event of "once" and those of "recurring" are drawn in the same way, so its shape or its size cannot give
    the answer: only whether the rest of the series does the same. A model trained on single events alone learns
    that a spike or a plateau is an anomaly, and says so of the ones real series are made of.

    A share of the examples (`TWO_TURNS`) asks in two turns: first what the notable events are, which is answered
    with places and values and no verdict, then whether any of them is an anomaly. Describing and judging are then
    two answers, and a true description does not depend on the verdict."""

    KINDS = ("spike", "dropout", "shift", "burst")
    CASES = (("once", 0.40), ("recurring", 0.30), ("odd", 0.15), ("untouched", 0.15))
    TWO_TURNS = 0.35
    DESCRIBE = "What are the most notable events in this time series? Give their points and values."
    JUDGE = "Is any of this an anomaly? Answer Yes or No first; if Yes, say around which points."
    NAMES = {"spike": "spikes", "dropout": "drops", "shift": "shifts of the level", "burst": "bursts of larger swings"}

    def __init__(self, path: str, seed: int = 0):
        super().__init__(path, seed=seed)
        self.records = [record for record in self.records if record["messages"][-1]["content"].lower().startswith("no")]
        if not self.records:
            raise ValueError(f"{path} holds no series without an anomaly to put one into.")

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, worker.id if worker else 0, 2])
        for record in super().__iter__():
            example = self.inject(rng, _with_arrays(record["messages"]))
            if example is not None:
                yield example

    @classmethod
    def inject(cls, rng, messages: list[dict], kind: str | None = None, case: str | None = None) -> dict | None:
        """`messages` as one of the four cases, or `None` for a series too short or too flat to change."""
        parts = messages[0]["content"]
        index = next(k for k, part in enumerate(parts) if part["type"] == "timeseries")
        original = np.array(parts[index]["timeseries"], dtype=np.float64)
        spread, length = float(original.std()), len(original)
        if length < 96 or spread <= 0 or not np.isfinite(original).all():
            return None
        names, shares = zip(*cls.CASES)
        case = case or names[int(rng.choice(len(names), p=shares))]
        if case == "untouched":
            return {"messages": messages, "change": {"case": case}}
        if case != "once" and length < 160:
            case = "once"
        series = original.copy()
        number = lambda value: written_value(value, spread)
        kind = kind or cls.KINDS[int(rng.integers(len(cls.KINDS)))]
        # how far beyond what the series already does: 2 to 5 times for a clear event, 1.5 to 2 for a subtle one
        beyond = float(rng.uniform(2, 5) if rng.random() < 0.5 else rng.uniform(1.5, 2))
        sign = 1 if rng.random() < 0.5 else -1
        levels = original[: length // 16 * 16].reshape(-1, 16).mean(axis=1)
        # what the series already does: its largest step, its largest change of level, its range
        step = max(float(np.abs(np.diff(original)).max()), spread)
        jump = max(float(np.abs(np.diff(levels)).max()), spread)
        low, span = float(original.min()), float(np.ptp(original))

        def width_of(kind: str) -> int:
            if kind == "spike":
                return 1 if rng.random() < 0.7 else int(rng.integers(2, 4))
            return int(rng.integers(2, 9)) if kind == "dropout" else int(rng.integers(16, 41))

        def put(kind: str, point: int, width: int | None, sign: int, scale: float) -> bool:
            """Puts one event into `series`; `scale` is its size against the one drawn for this window."""
            end = length if width is None else point + width
            if kind == "spike":
                series[point:end] += sign * scale * beyond * step
            elif kind == "dropout":
                series[point:end] = low - scale * (beyond - 1) * 0.5 * span
            elif kind == "shift":
                series[point:end] += sign * scale * beyond * jump
            else:
                inside = original[point:end]
                widest = max(float(np.ptp(original[k : k + width])) for k in range(0, length - width + 1, 4))
                if np.ptp(inside) <= 0 or widest <= 0:
                    return False
                series[point:end] = inside.mean() + (inside - inside.mean()) * scale * beyond * widest / float(np.ptp(inside))
            return True

        def told(kind: str, point: int, width: int | None, sign: int) -> tuple[int, int, str]:
            """The stretch of one event and the sentence that describes it, from the values of `series`."""
            if kind == "spike":
                inside = series[point : point + width]
                peak = point + int(np.argmax(sign * inside))
                way = "upward" if sign > 0 else "downward"
                if width == 1:
                    said = (f"A one-point {way} spike takes the value from about {number(series[point - 1])} to "
                            f"{number(series[point])} at point {point}, and it returns to about {number(series[point + 1])}.")
                else:
                    said = (f"A {way} spike of {width} points takes the value from about {number(series[point - 1])} to "
                            f"{number(series[peak])} at point {peak}, and it returns to about {number(series[point + width])} at point {point + width}.")
                return point - 1, point + width, said
            if kind == "dropout":
                last = point + width - 1
                return point, last, (f"The value drops from about {number(series[point - 1])} to {number(series[point])} at point {point}, "
                                     f"stays there for {width} points, and returns to about {number(series[last + 1])} at point {last + 1}.")
            if kind == "shift":
                way, before = "up" if sign > 0 else "down", number(series[point - 16 : point].mean())
                if width is None:
                    return point, min(length - 1, point + 15), (
                        f"The level shifts {way} at point {point}: the mean of the 16 points before is about {before}, and the mean "
                        f"of the 16 points after is about {number(series[point : point + 16].mean())}. It does not come back.")
                return point, point + width - 1, (
                    f"The level shifts {way} at point {point}: the mean of the 16 points before is about {before}, and the mean from "
                    f"point {point} to point {point + width - 1} is about {number(series[point : point + width].mean())}. "
                    f"It comes back at point {point + width}.")
            last = point + width - 1
            inside, before = series[point : last + 1], series[max(0, point - width) : point]
            return point, last, (f"The swings grow: between point {point} and point {last} the values go from about {number(inside.min())} to "
                                 f"{number(inside.max())}, against {number(before.min())} to {number(before.max())} in the {len(before)} points before.")

        width = width_of(kind)
        plural = cls.NAMES[kind]
        if case == "once":
            if kind == "shift" and rng.random() < 0.5:
                width = None  # a shift that stays
            point = int(rng.integers(24, length - 48 - (width or 0)))
            if not put(kind, point, width, sign, 1.0):
                return cls.inject(rng, messages, "spike", case)
            first, last, said = told(kind, point, width, sign)
            listed, verdict = f"One event stands out. {said}", f"Yes, around points {first} to {last}."
            answer, reason, places = f"{verdict} {said}", "Nothing else in the series looks like it.", [point]
        else:
            # three to six of them, evenly spaced, with room between two of them
            count = int(rng.integers(3, 7))
            while count > 3 and (length - 48) // count < 2 * width + 24:
                count -= 1
            period = (length - 48) // count
            width = max(1, min(width, period // 2 - 6))
            if period < 32 or (kind == "burst" and width < 12):
                return cls.inject(rng, messages, kind, "once")
            start = 24 + int(rng.integers(0, max(1, period - width - 8)))
            places = [start + turn * period for turn in range(count) if start + turn * period + width < length - 16]
            odd = None
            if case == "odd":
                if rng.random() < 0.5 or period - width < 28:
                    # one of them does not fit the others: much larger, or the other way
                    slot = int(rng.integers(len(places)))
                    other_way = kind in ("spike", "shift") and rng.random() < 0.5
                    odd = (kind, places[slot], width, -sign if other_way else sign, float(rng.uniform(1, 1.5) if other_way else rng.uniform(2.2, 3.5)))
                    places = places[:slot] + places[slot + 1 :]
                else:
                    # an event of another kind between two of them
                    other = "dropout" if kind == "spike" else "spike" if kind == "dropout" else ("spike", "dropout")[int(rng.integers(2))]
                    slot, wide = int(rng.integers(len(places) - 1)), width_of(other)
                    gap = places[slot] + width + (period - width - wide) // 2
                    odd = (other, gap, wide, 1 if rng.random() < 0.5 else -1, 1.0)
            for place in places:
                if not put(kind, place, width, sign, float(rng.uniform(0.9, 1.1))):
                    return cls.inject(rng, messages, "spike", case)
            if odd and not put(*odd):
                return cls.inject(rng, messages, "spike", case)
            points = ", ".join(str(place) for place in places[:-1]) + f" and {places[-1]}"
            # what the recurring ones reach, read in the series
            if kind == "burst":
                swings = [float(np.ptp(series[place : place + width])) for place in places]
                reach = f"each with a swing of about {number(min(swings))} to {number(max(swings))}"
            else:
                reached = [float(series[place : place + width].mean()) if kind == "shift" else float(series[place + int(np.argmax(sign * series[place : place + width]))]) if kind == "spike" else float(series[place]) for place in places]
                reach = f"each {'to a level of' if kind == 'shift' else 'to'} about {number(min(reached))} to {number(max(reached))}"
            recur = f"{plural[0].upper()}{plural[1:]} of {width} point{'s' if width > 1 else ''} recur about every {period} points, near points {points}, {reach}."
            if odd is None:
                first = last = None
                listed, verdict = recur, "No."
                reason = "They recur with about the same size, so they are part of the pattern of this series."
                answer = f"No. The {plural} near points {points} recur about every {period} points with about the same size, so they are part of the pattern of this series."
            else:
                first, last, said = told(*odd[:4])
                listed, verdict = f"{recur} One event differs. {said}", f"Yes, around points {first} to {last}."
                reason = "The others recur with about the same size; this one does not fit them."
                answer = f"{verdict} {said} The {'other ' if odd[0] == kind else ''}{plural} recur regularly and are part of the pattern."
        content = [{**part, "timeseries": series} if k == index else part for k, part in enumerate(parts)]
        change = {"case": case, "kind": kind, "beyond": round(beyond, 2), "stretch": None if first is None else [first, last], "points": places, "turns": 1}
        if rng.random() < cls.TWO_TURNS:
            # what is there first, then whether it is an anomaly
            asked = [{**part, "text": ". " + cls.DESCRIBE} if k == len(content) - 1 and part["type"] == "text" else part for k, part in enumerate(content)]
            turns = [
                {**messages[0], "content": asked},
                {"role": "assistant", "content": listed},
                {"role": "user", "content": [{"type": "text", "text": cls.JUDGE}]},
                {"role": "assistant", "content": f"{verdict} {reason}"},
            ]
            return {"messages": turns, "change": change | {"turns": 2}}
        return {"messages": [{**messages[0], "content": content}, {"role": "assistant", "content": answer}], "change": change}


def _with_arrays(messages: list[dict]) -> list[dict]:
    """The series of a conversation read from JSON, as arrays."""
    converted = []
    for message in messages:
        content = message["content"]
        if not isinstance(content, str):
            content = [
                {**part, "timeseries": np.asarray(part["timeseries"], dtype=np.float64)} if part["type"] == "timeseries" else part
                for part in content
            ]
        converted.append({**message, "content": content})
    return converted


_READING_HOLDOUT_SEED = 1_000_003


HOLDOUT = 300


def load_chatts_stream(
    mixture: dict[str, float],
    seed: int = 0,
    shuffle_buffer: int = 2000,
    chat_path: str | None = None,
    reading_max_length: int = 512,
    real_path: str | None = None,
) -> MixedStream:
    """Streams a mixture of ChatTS training subsets, e.g. `{"align_256": 0.9, "ift": 0.1}`, without downloading
    them in full. The first `HOLDOUT` records of each subset are left out, for `load_chatts_holdout`. The name
    `"reading"` stands for the generated `ReadingTasks`, and `"locate"`, `"exact"`, `"verdict"` and `"values"` for
    their other profiles; `"chat"` for the plain conversations of the file `chat_path`; `"real"` for the questions on
    real series of the file `real_path`, and `"injected"` for the windows of that file without an anomaly, with
    events put into them (`InjectedAnomalies`). The generated series are up to `reading_max_length` points long."""
    datasets = []
    for name in mixture:
        if name in GENERATED:
            datasets.append(ReadingTasks(seed=seed, max_length=reading_max_length, profile=GENERATED[name]))
            continue
        if name == CHAT:
            if not chat_path:
                raise ValueError('The mixture names "chat": pass the file of conversations (`--chat-data`).')
            datasets.append(ChatReplay(chat_path, seed=seed))
            continue
        if name == REAL:
            if not real_path:
                raise ValueError('The mixture names "real": pass the file of questions on real series (`--real-data`).')
            datasets.append(RealSeries(real_path, seed=seed))
            continue
        if name == INJECTED:
            if not real_path:
                raise ValueError('The mixture names "injected": pass the file of questions on real series (`--real-data`).')
            datasets.append(InjectedAnomalies(real_path, seed=seed))
            continue
        from datasets import load_dataset

        subset = load_dataset(CHATTS_DATASET, name, split="train", streaming=True)
        datasets.append(subset.skip(HOLDOUT).shuffle(seed=seed, buffer_size=shuffle_buffer))
    return MixedStream(datasets, list(mixture.values()), seed=seed)


def load_chatts_holdout(
    mixture: dict[str, float],
    count: int = 300,
    chat_path: str | None = None,
    reading_max_length: int = 512,
    real_path: str | None = None,
) -> list[dict]:
    """About `count` conversations the training stream never yields, split between the subsets like the mixture."""
    total = sum(mixture.values())
    # both generated profiles are scored on the same balanced questions: count them as one source, or a mixture
    # naming both would hold the same questions twice
    generated = sum(weight for name, weight in mixture.items() if name in GENERATED)
    examples = []
    for name, weight in mixture.items():
        share = min(HOLDOUT, max(1, round(count * weight / total)))
        if name in GENERATED:
            if generated:
                share = min(HOLDOUT, max(1, round(count * generated / total)))
                # a seed the training stream never uses
                tasks = ReadingTasks(seed=_READING_HOLDOUT_SEED, max_length=reading_max_length)
                examples.extend(itertools.islice(tasks, share))
                generated = 0
            continue
        if name == CHAT:
            with open(chat_path) as lines:
                records = [json.loads(line) for line in lines if line.strip()]
            examples.extend(records[: min(share, _CHAT_HOLDOUT)])
            continue
        if name == REAL:
            with open(real_path) as lines:
                records = [json.loads(line) for line in itertools.islice(filter(str.strip, lines), min(share, _CHAT_HOLDOUT))]
            examples.extend({"messages": _with_arrays(record["messages"])} for record in records)
            continue
        if name == INJECTED:
            # made from the same windows as the questions on real series, which the held-out set already has
            continue
        from datasets import load_dataset

        records = load_dataset(CHATTS_DATASET, name, split="train", streaming=True).take(share)
        examples.extend({"messages": chatts_record_to_messages(record)} for record in records)
    return examples


@dataclass
class Lfm3TsDataCollator:
    """Builds training batches from conversations whose last message is the assistant's answer.

    The loss covers the answer and the end-of-turn token only; the prompt, including every time-series position, is
    masked with `-100`.

    Args:
        processor (`Lfm3TsProcessor`):
            The processor.
        max_length (`int`, *optional*, defaults to 4096):
            The longest sequence, in tokens. Longer answers are cut; an example whose prompt does not fit is
            dropped from its batch with a warning.
        enable_thinking (`bool`, *optional*, defaults to `False`):
            Whether the prompt opens a reasoning block. Off by default: the answers hold no reasoning traces.
    """

    processor: Lfm3TsProcessor
    max_length: int = 4096
    enable_thinking: bool = False

    def encode(self, messages: list[dict]) -> dict | None:
        """Tokenizes one conversation into `input_ids`, `labels` and, if it holds series, their features. Returns
        `None` when the prompt leaves no room for the answer within `max_length`.

        A conversation of several turns has the loss on each of its answers. Its series must all be in the first
        message. The turns that do not fit within `max_length` are left out."""
        if messages[-1]["role"] != "assistant" or not isinstance(messages[-1]["content"], str):
            raise ValueError("The last message must be the assistant's answer, as a string.")
        answers = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
        prompt = self.processor.apply_chat_template(
            messages[: answers[0]],
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=self.enable_thinking,
        )
        tokenizer = self.processor.tokenizer
        prompt_ids = prompt["input_ids"][0]
        encode_answer = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        answer_ids = encode_answer(messages[answers[0]]["content"])

        room = self.max_length - len(prompt_ids)
        if room < 1:
            return None
        answer_ids = answer_ids[:room]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        if len(answers) > 1:
            render = lambda end: self.processor.apply_chat_template(
                messages[:end], add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking
            )
            before = render(answers[0])
            for previous, index in zip(answers, answers[1:]):
                if any(isinstance(m["content"], list) and any(p.get("type") == "timeseries" for p in m["content"]) for m in messages[previous:index]):
                    raise ValueError("Only the first message of a conversation of several turns may hold series.")
                after = render(index)
                # what the template writes between two answers: the end of a turn, the next question, the header
                # of the next answer
                closed = before + messages[previous]["content"] + tokenizer.eos_token
                if not after.startswith(closed):
                    raise ValueError("The chat template does not write a conversation as its turns one after the other.")
                between = tokenizer(after[len(closed) :], add_special_tokens=False)["input_ids"]
                answer_ids = encode_answer(messages[index]["content"])
                if len(input_ids) + len(between) + len(answer_ids) > self.max_length:
                    break
                input_ids += between + answer_ids
                labels += [-100] * len(between) + answer_ids
                before = after

        encoded = {"input_ids": input_ids, "labels": labels}
        if "input_values" in prompt:
            encoded["input_values"] = prompt["input_values"]
            encoded["input_values_mask"] = prompt["input_values_mask"]
        return encoded

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        encoded = [self.encode(example["messages"]) for example in examples]
        dropped = sum(item is None for item in encoded)
        if dropped:
            logger.warning("Dropped %d example(s) whose prompt exceeds max_length=%d.", dropped, self.max_length)
        encoded = [item for item in encoded if item is not None]
        if not encoded:
            raise ValueError(f"No example of the batch fits in max_length={self.max_length}.")
        return self.collate(encoded)

    def collate(self, encoded: list[dict]) -> dict[str, torch.Tensor]:
        """Pads and stacks the outputs of `encode`."""
        length = max(len(item["input_ids"]) for item in encoded)
        pad_token_id = self.processor.tokenizer.pad_token_id

        batch = {
            "input_ids": torch.full((len(encoded), length), pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros((len(encoded), length), dtype=torch.long),
            "labels": torch.full((len(encoded), length), -100, dtype=torch.long),
        }
        for row, item in enumerate(encoded):
            size = len(item["input_ids"])
            batch["input_ids"][row, :size] = torch.tensor(item["input_ids"])
            batch["attention_mask"][row, :size] = 1
            batch["labels"][row, :size] = torch.tensor(item["labels"])

        series = [item for item in encoded if "input_values" in item]
        if series:
            # every example brings (num_series, num_patches, patch_size); align the patch axis and stack the series
            max_patches = max(item["input_values"].shape[1] for item in series)
            for key in ("input_values", "input_values_mask"):
                padded = [
                    np.pad(item[key], ((0, 0), (0, max_patches - item[key].shape[1]), (0, 0))) for item in series
                ]
                batch[key] = torch.from_numpy(np.concatenate(padded))
        return batch


class TokenBudgetBatches(torch.utils.data.IterableDataset):
    """Turns a stream of conversations into batches that each hold about the same number of tokens.

    Sequence lengths vary tenfold, and a batch is padded to its longest example, so a fixed batch size either
    wastes most of the memory on short examples or runs out of it on long ones. This buffers examples, sorts them
    by length, and cuts batches whose padded size stays within `max_tokens`. Use it with a dataloader batch size
    of 1 and `collate_fn=lambda batches: batches[0]`.

    Args:
        stream (iterable):
            Yields `{"messages": [...]}` examples.
        collator (`Lfm3TsDataCollator`):
            Encodes and pads the examples.
        max_tokens (`int`, *optional*, defaults to 16000):
            The largest padded batch, in tokens (examples times longest sequence).
        buffer_size (`int`, *optional*, defaults to 512):
            How many examples are sorted together. Batches leave the buffer in random order.
        seed (`int`, *optional*, defaults to 0):
            The seed of the batch order.
    """

    def __init__(self, stream, collator: Lfm3TsDataCollator, max_tokens: int = 16000, buffer_size: int = 512, seed: int = 0):
        self.stream = stream
        self.collator = collator
        self.max_tokens = max_tokens
        self.buffer_size = buffer_size
        self.seed = seed

    def group(self, encoded: list[dict]) -> list[list[dict]]:
        """Cuts encoded examples into batches within the token budget, shortest examples first."""
        batches, current = [], []
        for item in sorted(encoded, key=lambda item: len(item["input_ids"])):
            # sorted, so the newcomer is the longest and sets the padded length
            if current and (len(current) + 1) * len(item["input_ids"]) > self.max_tokens:
                batches.append(current)
                current = []
            current.append(item)
        if current:
            batches.append(current)
        return batches

    def encoded_buffers(self):
        buffer = []
        for example in self.stream:
            item = self.collator.encode(example["messages"])
            if item is None:
                continue
            buffer.append(item)
            if len(buffer) == self.buffer_size:
                yield buffer
                buffer = []
        if buffer:
            yield buffer

    def examples_per_batch(self) -> float:
        """The mean batch size over the first buffer, to convert an example budget into steps."""
        buffer = next(self.encoded_buffers())
        return len(buffer) / len(self.group(buffer))

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (worker.id if worker else 0))
        for buffer in self.encoded_buffers():
            batches = self.group(buffer)
            rng.shuffle(batches)
            for batch in batches:
                yield self.collator.collate(batch)


__all__ = [
    "CHATTS_DATASET",
    "Lfm3TsDataCollator",
    "MixedStream",
    "ChatReplay",
    "InjectedAnomalies",
    "ReadingTasks",
    "RealSeries",
    "ScheduledStream",
    "TokenBudgetBatches",
    "chatts_record_to_messages",
    "fix_stated_lengths",
    "written_value",
    "load_chatts_holdout",
    "load_chatts_stream",
]
