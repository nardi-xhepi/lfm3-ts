"""Scores answers on the TSB-AD-U test windows and prints the table of the README.

    python eval/score_tsb.py --questions <folder>/test_questions.jsonl

The questions hold windows of third-party series and are not in this repository. Rebuild them from a copy of
TSB-AD-U (https://github.com/TheDatumOrg/TSB-AD):

    python scripts/make_real_data.py windows --dataset <unpacked TSB-AD-U> --out <folder>

`eval/windows.csv` lists the 350 windows of the published table (id, source, file, first point). The script checks the
rebuilt questions against it and stops if they differ, unless `--no-check` is given.

The rule (`score_anomaly` in scripts/baseline.py): an answer is read from its first word, yes or no. On a window with
a labelled anomaly it has the right place when it says yes and names a point at most 16 points from the labelled
stretch. Precision is right places over all yes answers, recall is right places over the windows with an anomaly.
An answer that starts with neither yes nor no is wrong; on a window without an anomaly it counts as a false alarm.
"""

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from baseline import score_anomaly  # noqa: E402


GROUPS = ["with a clear anomaly", "with a subtle anomaly", "without an anomaly"]
PARTS = {"new": "unseen sources", "known": "seen sources"}


def read_jsonl(path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as file:
        return [json.loads(line) for line in file if line.strip()]


def check_windows(questions: list[dict], manifest: Path) -> list[str]:
    """What differs between the questions and the windows of the published table."""
    with open(manifest) as file:
        expected = {int(row["id"]): (row["source"], row["file"], int(row["start"])) for row in csv.DictReader(file)}
    found = {question["id"]: (question["source"], question["file"], question["start"]) for question in questions}
    problems = []
    if len(found) != len(expected):
        problems.append(f"{len(found)} questions, the published table has {len(expected)} windows")
    wrong = [index for index in expected if index in found and found[index] != expected[index]]
    if wrong:
        problems.append(f"{len(wrong)} windows are not the published ones (first: id {wrong[0]})")
    return problems


def table_of(questions: list[dict], predictions: dict[int, str], tolerance: int = 16) -> dict:
    """Per group of windows: how many, how many with the right yes or no, how many with the right place."""
    table = {}
    for question in questions:
        result = score_anomaly(question, predictions[question["id"]], tolerance)
        row = table.setdefault(question["group"], {"n": 0, "right": 0, "place": 0})
        row["n"] += 1
        row["right"] += int(result["right"])
        row["place"] += int(result.get("place right", 0))
    return table


def summary(table: dict, part: str | None = None) -> dict:
    """Precision, recall and F1 of "an anomaly at the right place", and the false alarms, over one part of the test
    set (`"unseen sources"`, `"seen sources"`) or over all of it."""
    rows = {group: row for group, row in table.items() if part is None or group.startswith(part)}
    anomalies = sum(row["n"] for group, row in rows.items() if "without" not in group)
    places = sum(row["place"] for group, row in rows.items() if "without" not in group)
    quiet = sum(row["n"] for group, row in rows.items() if "without" in group)
    alarms = sum(row["n"] - row["right"] for group, row in rows.items() if "without" in group)
    said_yes = alarms + sum(row["right"] for group, row in rows.items() if "without" not in group)
    precision = places / max(said_yes, 1)
    recall = places / max(anomalies, 1)
    return {
        "windows with an anomaly": anomalies,
        "windows without": quiet,
        "right places": places,
        "yes answers": said_yes,
        "false alarms": alarms,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "false alarm rate": alarms / max(quiet, 1),
    }


def markdown(results: dict) -> str:
    lines = []
    for scope, title in (("new", "New sources"), ("all", "All windows")):
        first = next(iter(results.values()))[scope]
        lines += [
            f"**{title}** ({first['windows with an anomaly']} windows with an anomaly, {first['windows without']} without)",
            "",
            "| Model | Precision | Recall | F1 | False alarms |",
            "|---|---|---|---|---|",
        ]
        for name, result in results.items():
            stats = result[scope]
            lines.append(
                f"| {name} | {stats['precision']:.0%} ({stats['right places']}/{stats['yes answers']}) "
                f"| {stats['recall']:.0%} ({stats['right places']}/{stats['windows with an anomaly']}) | {stats['f1']:.2f} "
                f"| {stats['false alarms']}/{stats['windows without']} ({stats['false alarm rate']:.0%}) |"
            )
        lines.append("")
    lines += ["**By group** (share of windows with the right place; for windows without an anomaly, share with a false alarm)", ""]
    names = list(results)
    lines += ["| Group | Windows | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    groups = next(iter(results.values()))["groups"]
    for part in PARTS.values():
        for kind in GROUPS:
            group = f"{part}, {kind}"
            if group not in groups:
                continue
            cells = []
            for name in names:
                row = results[name]["groups"][group]
                count = row["n"] - row["right"] if "without" in kind else row["place"]
                cells.append(f"{count / row['n']:.0%} ({count}/{row['n']})")
            lines.append(f"| {group} | {groups[group]['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", required=True, help="test_questions.jsonl written by scripts/make_real_data.py windows")
    parser.add_argument(
        "--predictions",
        nargs="+",
        default=[f"LFM3-TS={HERE / 'predictions/lfm3-ts.jsonl'}", f"ChatTS-8B={HERE / 'predictions/chatts-8b.jsonl'}", f"LFM3-1.7B-text={HERE / 'predictions/lfm3-1.7b-text.jsonl'}"],
        help="name=file pairs; each file holds one {id, prediction} per line, as scripts/baseline.py run writes",
    )
    parser.add_argument("--tolerance", type=int, default=16, help="how far from the labelled stretch a point may be")
    parser.add_argument("--no-check", action="store_true", help="do not compare the questions with eval/windows.csv")
    parser.add_argument("--json", help="also write the numbers to this file")
    args = parser.parse_args(argv)

    questions = read_jsonl(args.questions)
    problems = [] if args.no_check else check_windows(questions, HERE / "windows.csv")
    if problems:
        raise SystemExit("the questions are not those of the published table: " + "; ".join(problems))
    results = {}
    for item in args.predictions:
        name, _, path = item.rpartition("=")
        predictions = {record["id"]: record["prediction"] for record in read_jsonl(path)}
        missing = [question["id"] for question in questions if question["id"] not in predictions]
        if missing:
            raise SystemExit(f"{path} has no answer to {len(missing)} questions (first: id {missing[0]})")
        table = table_of(questions, predictions, args.tolerance)
        results[name or Path(path).stem] = {
            "all": summary(table),
            "new": summary(table, PARTS["new"]),
            "known": summary(table, PARTS["known"]),
            "groups": table,
        }
    print(markdown(results))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    main()
