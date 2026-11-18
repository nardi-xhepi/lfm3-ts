"""Real series with labelled anomalies, from TSB-AD-U (https://github.com/TheDatumOrg/TSB-AD), as training
conversations and as a test set the training never touches.

    python scripts/make_real_data.py windows --dataset <unpacked TSB-AD-U> --out <folder>
    python scripts/make_real_data.py build --windows <folder> --descriptions <descriptions.jsonl> --out real.jsonl

`windows` splits the files, cuts the windows and measures each anomaly; `build` writes the conversations. The split:

- whole sources are kept for the test (`HELD_OUT`): the model never sees their kind of series;
- of every other source, one file in five is kept for the test too, so no test window shares a file with training.

The answer to a window with an anomaly is "Yes, around points A to B." followed by a description. Every number of
a description is measured here from the values (`measure`); a description written by a language model is kept only
if each number in it is one of those (`numbers_allowed`), and the plain sentence of `plain_description` is used
otherwise. A labelled stretch that nothing measurable sets apart from the rest of its window is not trained on:
answering "Yes" to it would be a guess."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

SEED = 5_000_011
# generated series, and series labelled at hundreds of single points
SKIPPED = {"MGAB", "TODS", "Stock", "TAO"}
# never trained on: sources whose licence does not clearly allow it, or that the ChatTS evaluation is built from
HELD_OUT = {"YAHOO", "UCR", "NAB", "MSL", "SMAP", "IOPS", "SWaT"}
LENGTHS = (256, 384, 512, 768, 1024)
TEST_LENGTH = 512
QUESTION = ". Is there an anomaly in this time series? Answer Yes or No first; if Yes, say around which point it is."
QUESTIONS = (
    QUESTION,
    ". Is there an anomaly in this time series? If so, where is it and what does it look like?",
    ". Does this time series contain an anomaly? Answer Yes or No first; if Yes, say where it is and describe it.",
    ". Is anything abnormal in this time series? If yes, give the points and describe what happens there.",
)
NO_ANSWERS = (
    "No, there is no anomaly in this time series.",
    "No, there is not an anomaly in this time series.",
    "No, nothing in this time series stands out as an anomaly.",
)


# for a labelled stretch whose values and swings stay in the usual range: the place, and no invented measurement
SUBTLE_ANSWERS = (
    "The values stay within their usual range there, but the pattern of the series changes.",
    "Nothing stands out in the level or the size of the swings, but the shape of the series departs from its usual pattern there.",
    "The series keeps its usual range in this stretch, yet its shape differs from the rest.",
)


def number(value: float) -> str:
    """A value at 4 significant digits, written without an exponent."""
    return np.format_float_positional(float(f"{float(value):.4g}"), trim="-")


def source_of(path: Path) -> str | None:
    fields = path.stem.split("_")
    if len(fields) <= 4 or fields[1] in SKIPPED or fields[4] == "Synthetic":
        return None
    return fields[1]


def split_files(directory: str) -> dict[str, list[Path]]:
    """The files of the training split and of the two test splits. Which files of a trained source go to the test is
    decided by a hash of their names, so the split does not depend on the order or the number of files found."""
    by_source: dict[str, list[Path]] = {}
    for path in sorted(Path(directory).rglob("*.csv")):
        if source_of(path):
            by_source.setdefault(source_of(path), []).append(path)
    splits = {"train": [], "test, seen sources": [], "test, unseen sources": []}
    for source, paths in by_source.items():
        if source in HELD_OUT:
            splits["test, unseen sources"] += paths
            continue
        paths = sorted(paths, key=lambda path: hashlib.md5(path.name.encode()).hexdigest())
        kept = max(1, round(len(paths) / 5)) if len(paths) > 1 else 0
        splits["test, seen sources"] += paths[:kept]
        splits["train"] += paths[kept:]
    return splits


def period_of(values: np.ndarray) -> int | None:
    """The length of the cycle of a series that repeats, from its autocorrelation, or None."""
    centred = values - values.mean()
    if len(centred) < 64 or not centred.any():
        return None
    full = np.correlate(centred, centred, mode="full")[len(centred) - 1 :]
    full = full / full[0]
    limit = len(centred) // 3
    below = np.flatnonzero(full[:limit] < 0)
    if not len(below):
        return None
    start = int(below[0])
    if start >= limit - 1:
        return None
    lag = start + int(np.argmax(full[start:limit]))
    return lag if full[lag] > 0.5 and lag >= 4 else None


def measure(series: np.ndarray, segment: tuple[int, int]) -> dict:
    """What can be said of a labelled stretch from the values alone. `visible` tells whether anything sets it apart."""
    first, last = segment
    inside = series[first : last + 1]
    rest = np.r_[series[: max(0, first - 3)], series[last + 4 :]]
    before = series[max(0, first - 20) : first]
    after = series[last + 1 : last + 21]
    usual, spread = float(np.median(rest)), float(rest.std())
    level_before = float(np.median(before)) if len(before) else usual
    level_after = float(np.median(after)) if len(after) else usual
    top, bottom = int(first + inside.argmax()), int(first + inside.argmin())
    upward = series[top] - level_before >= level_before - series[bottom]
    extreme = top if upward else bottom
    amplitude = float(abs(series[extreme] - level_before))
    facts = {
        "first": first,
        "last": last,
        "duration": last - first + 1,
        "direction": "upward" if upward else "downward",
        "extreme_point": extreme,
        "extreme_value": number(series[extreme]),
        "level_before": number(level_before),
        "level_after": number(level_after),
        "amplitude": number(amplitude),
        "usual_level": number(usual),
        "returns": bool(abs(level_after - level_before) < 0.25 * amplitude) if amplitude else True,
    }
    visible = amplitude >= 3 * (spread + 1e-12)
    period = period_of(rest[: max(0, first - 3)] if first > len(series) // 2 else rest[max(0, first - 3) :])
    if period:
        cycles = [rest[i : i + period] for i in range(0, len(rest) - period + 1, period)]
        swing = float(np.median([cycle.max() - cycle.min() for cycle in cycles]))
        low, high = max(0, min(first, (first + last - period) // 2)), min(len(series), max(last + 1, (first + last + period) // 2 + 1))
        stretch = series[low:high]
        swing_inside = float(stretch.max() - stretch.min())
        facts |= {"period": period, "usual_swing": number(swing), "swing_inside": number(swing_inside)}
        visible = bool(swing and abs(swing_inside - swing) >= 0.3 * swing)
    facts["visible"] = bool(visible)
    return facts


def plain_description(facts: dict) -> str:
    """One sentence built from the measurements, for when no written description is kept."""
    if "period" in facts:
        change = "larger" if float(facts["swing_inside"]) > float(facts["usual_swing"]) else "smaller"
        return (
            f"The series repeats about every {facts['period']} points with a swing of about {facts['usual_swing']}; "
            f"in this stretch the swing is {change}, about {facts['swing_inside']}."
        )
    move = "rises" if facts["direction"] == "upward" else "falls"
    end = (
        f"then comes back to about {facts['level_after']}"
        if facts["returns"]
        else f"and stays near {facts['level_after']} afterwards"
    )
    return (
        f"The value {move} from about {facts['level_before']} to {facts['extreme_value']} at point "
        f"{facts['extreme_point']}, a change of about {facts['amplitude']}, {end}."
    )


def numbers_allowed(description: str, facts: dict) -> bool:
    """Whether every number of a description is one of the measurements."""
    allowed = {str(value).lstrip("-") for value in facts.values() if not isinstance(value, bool)}
    found = re.findall(r"\d[\d,]*\.?\d*", description)
    return bool(found) and all(token.replace(",", "").rstrip(".") in allowed for token in found)


def cut_windows(path: Path, rng, lengths, per_anomaly: int, most: int) -> list[dict]:
    """Windows of one file: up to `per_anomaly` around each labelled stretch that hold that stretch and no other,
    and as many without any label."""
    table = np.genfromtxt(path, delimiter=",", skip_header=1)
    values, labelled = table[:, 0], table[:, -1] > 0
    if len(values) < 2 * max(lengths) or np.isnan(values).any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[0, labelled.astype(int), 0])).reshape(-1, 2)
    windows = []
    for begin, end in rng.permutation(edges):
        if len(windows) >= most:
            break
        for _ in range(per_anomaly):
            length = int(rng.choice(lengths))
            if end - begin > length // 4:
                continue
            low, high = max(0, end - length + 32), min(begin - 32, len(values) - length)
            if low > high:
                continue
            start = int(rng.integers(low, high + 1))
            if labelled[start : start + length].sum() == end - begin:
                windows.append((start, length, (int(begin - start), int(end - 1 - start))))
    normal = []
    for _ in range(20 * max(1, len(windows))):
        if len(normal) >= max(1, len(windows)):
            break
        length = int(rng.choice(lengths))
        start = int(rng.integers(0, len(values) - length))
        if not labelled[start : start + length].any():
            normal.append((start, length, None))
    records = []
    for start, length, segment in windows + normal:
        series = values[start : start + length]
        if series.std() == 0:
            continue
        record = {"source": source_of(path), "file": path.name, "start": start, "length": length, "segment": segment}
        if segment:
            record["facts"] = measure(series, segment)
        records.append(record | {"series": [float(value) for value in series]})
    return records


def sheet(record: dict) -> str:
    """What a describer is given for one window: the measurements, and the values of the stretch and around it."""
    facts, series = record["facts"], record["series"]
    low, high = max(0, facts["first"] - 12), min(len(series), facts["last"] + 13)
    step = max(1, (high - low) // 48)
    values = " ".join(f"{index}:{number(series[index])}" for index in range(low, high, step))
    shown = {name: value for name, value in facts.items() if name != "visible"}
    return f"### {record['id']}  (source {record['source']})\nFACTS: {json.dumps(shown)}\nVALUES (point:value): {values}"


def make_windows(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = split_files(args.dataset)
    counts = {}
    for split, paths in splits.items():
        train = split == "train"
        rng = np.random.default_rng([SEED, len(split)])
        records = []
        for path in paths:
            records += cut_windows(
                path, rng, LENGTHS if train else (TEST_LENGTH,), per_anomaly=2 if train else 1, most=args.per_file if train else 4
            )
        if train:
            hidden = [record for record in records if record["segment"] and not record["facts"]["visible"]]
            records = [record for record in records if not (record["segment"] and not record["facts"]["visible"])]
            counts["trained on with the place only (nothing measurable sets the stretch apart)"] = len(hidden)
        order = rng.permutation(len(records))
        records = [records[int(index)] for index in order]
        if not train:
            # as many windows without an anomaly as with one, source by source, and no source above `per_source`
            kept, seen = [], {}
            for record in records:
                key = (record["source"], bool(record["segment"]))
                if seen.get(key, 0) < args.per_source:
                    seen[key] = seen.get(key, 0) + 1
                    kept.append(record)
            records = kept
        for index, record in enumerate(records):
            record["id"] = f"{'train' if train else 'seen' if 'unseen' not in split else 'unseen'}-{index}"
        if train:
            # after the others, which keeps their order and their names whether or not these are there
            for index, record in enumerate(hidden):
                record["id"] = f"train-subtle-{index}"
            records += hidden
        name = "train" if train else ("test_seen" if "unseen" not in split else "test_unseen")
        with open(out / f"{name}_windows.jsonl", "w") as lines:
            for record in records:
                lines.write(json.dumps(record) + "\n")
        counts[split] = f"{len(paths)} files, {sum(bool(r['segment']) for r in records)} windows with an anomaly, {sum(not r['segment'] for r in records)} without"
        if train:
            described = [record for record in records if record["segment"] and record["facts"]["visible"]]
            for part in range(0, len(described), args.sheet_size):
                (out / f"sheet_{part // args.sheet_size:02d}.txt").write_text(
                    "\n\n".join(sheet(record) for record in described[part : part + args.sheet_size]) + "\n"
                )
    # the test questions, in the form `scripts/baseline.py run` and `score` read
    questions = []
    for name, label in (("test_seen", "seen sources"), ("test_unseen", "unseen sources")):
        with open(out / f"{name}_windows.jsonl") as lines:
            for line in lines:
                record = json.loads(line)
                segment = record["segment"]
                group = "without an anomaly"
                if segment:
                    inside = np.asarray(record["series"][segment[0] : segment[1] + 1])
                    rest = np.asarray(record["series"][: segment[0]] + record["series"][segment[1] + 1 :])
                    clear = np.abs(inside - np.median(rest)).max() >= 4 * (rest.std() + 1e-12)
                    group = "with a clear anomaly" if clear else "with a subtle anomaly"
                questions.append(
                    {
                        "id": len(questions),
                        "task": "anomaly",
                        "group": f"{label}, {group}",
                        "source": record["source"],
                        "file": record["file"],
                        "start": record["start"],
                        "segment": segment,
                        "facts": record.get("facts"),
                        "parts": [
                            {"text": f"This is a time series of length {record['length']}: "},
                            {"series": record["series"]},
                            {"text": QUESTION},
                        ],
                        "reference": f"Yes, around points {segment[0]} to {segment[1]}." if segment else "No.",
                        "format": '"Yes, around point N." or "No."',
                    }
                )
    with open(out / "test_questions.jsonl", "w") as lines:
        for question in questions:
            lines.write(json.dumps(question) + "\n")
    for name, value in counts.items():
        print(f"{name}: {value}")
    print(f"{len(questions)} test questions in {out / 'test_questions.jsonl'}")


def conversation(record: dict, answer: str, rng) -> dict:
    question = QUESTIONS[int(rng.integers(len(QUESTIONS)))]
    content = [
        {"type": "text", "text": f"This is a time series of length {record['length']}: "},
        {"type": "timeseries", "timeseries": record["series"]},
        {"type": "text", "text": question},
    ]
    return {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": answer}]}


def build(args) -> None:
    written = {}
    if args.descriptions and Path(args.descriptions).exists():
        with open(args.descriptions) as lines:
            written = {row["id"]: row["description"] for row in map(json.loads, filter(str.strip, lines))}
    rng = np.random.default_rng(SEED)
    kept = plain = subtle = 0
    with open(Path(args.windows) / "train_windows.jsonl") as lines, open(args.out, "w") as out:
        for line in lines:
            record = json.loads(line)
            if not record["segment"]:
                answer = NO_ANSWERS[int(rng.integers(len(NO_ANSWERS)))]
            elif not record["facts"]["visible"]:
                facts, subtle = record["facts"], subtle + 1
                description = SUBTLE_ANSWERS[int(rng.integers(len(SUBTLE_ANSWERS)))]
                answer = f"Yes, around points {facts['first']} to {facts['last']}. {description}"
            else:
                facts = record["facts"]
                description = written.get(record["id"], "").strip()
                if description and numbers_allowed(description, facts):
                    kept += 1
                else:
                    description, plain = plain_description(facts), plain + 1
                answer = f"Yes, around points {facts['first']} to {facts['last']}. {description}"
            out.write(json.dumps(conversation(record, answer, rng)) + "\n")
    print(f"{kept} written descriptions kept, {plain} plain ones, {subtle} with the place only, in {args.out}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    windows = commands.add_parser("windows", help="split the files, cut the windows, measure the anomalies")
    windows.add_argument("--dataset", required=True, help="the folder TSB-AD-U was unpacked in")
    windows.add_argument("--out", required=True)
    windows.add_argument("--per-file", type=int, default=40, help="training windows with an anomaly from one file, at most")
    windows.add_argument("--per-source", type=int, default=25, help="test windows of each kind from one source, at most")
    windows.add_argument("--sheet-size", type=int, default=120, help="windows on one sheet for the describers")
    builder = commands.add_parser("build", help="write the training conversations")
    builder.add_argument("--windows", required=True, help="the folder `windows` wrote")
    builder.add_argument("--descriptions", help="JSON lines of {id, description}")
    builder.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    (make_windows if args.command == "windows" else build)(args)


if __name__ == "__main__":
    main()
