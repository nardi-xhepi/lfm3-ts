"""Precision, recall and false alarms of answers to "is there an anomaly, and where?".

    python scripts/detection.py questions.jsonl answers-a.jsonl answers-b.jsonl

A window with an anomaly is found when the answer says yes and names a point within 16 points of the labelled
stretch (`baseline.score_anomaly`). Recall is the share of those windows that are found; precision is the share of
the yes answers that find one; a false alarm is a yes on a window without an anomaly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import baseline  # noqa: E402


def detection(questions: list[dict], answers: dict) -> dict:
    found = alarms = said_yes = 0
    with_anomaly = [question for question in questions if question["segment"]]
    for question in questions:
        scores = baseline.score_anomaly(question, answers.get(question["id"]) or "")
        if question["segment"]:
            found += scores["place right"]
            said_yes += scores["right"]
        else:
            alarms += 1 - scores["right"]
    said_yes += alarms
    precision, recall = found / max(said_yes, 1), found / max(len(with_anomaly), 1)
    return {
        "windows": len(questions),
        "recall": recall,
        "precision": precision,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "false alarms": int(alarms),
        "quiet windows": len(questions) - len(with_anomaly),
    }


def main(argv=None):
    paths = argv or sys.argv[1:]
    questions = baseline.read_jsonl(paths[0])
    sets = {"all windows": questions}
    for prefix in sorted({question["group"].split(",")[0] for question in questions}):
        sets[prefix] = [question for question in questions if question["group"].startswith(prefix)]
    if len(sets) == 2:
        del sets["all windows"]
    for path in paths[1:]:
        answers = {row["id"]: row.get("prediction") for row in baseline.read_jsonl(path)}
        print(Path(path).stem)
        for name, part in sets.items():
            row = detection(part, answers)
            print(f"  {name:<16} recall {row['recall']:.0%}  precision {row['precision']:.0%}  F1 {row['f1']:.2f}  "
                  f"false alarms {row['false alarms']} of {row['quiet windows']} quiet windows")


if __name__ == "__main__":
    main()
