"""Scores answers on the ChatTS evaluation sets A and B with the scoring code of ChatTS itself.

    git clone --depth 1 https://github.com/NetManAIOps/ChatTS
    python scripts/chatts_score.py --chatts ChatTS --dataset dataset_a.json runs/chatts/a/ours.jsonl runs/chatts/a/chatts-8b.jsonl

The datasets are at https://zenodo.org/records/14349206 (CC BY 4.0). The predictions are the files that
`scripts/baseline.py run ... --questions <questions --set chatts>` writes. Each record of a set asks several numbered
questions; `evaluation/evaluate_qa.py` of the ChatTS repository cuts an answer into its numbered parts and scores
each against the attributes of the record. Its functions are used here unchanged.

Two of its three scores need no other model and are reported: the *categorical* score (the right kind of trend,
of fluctuation, of noise...) and the *numerical* score (how close a position, an amplitude, a period is). The third,
the *reasoning* score, asks a judge language model to compare free text with the reference (through RAGAS). It is
reported only for the kinds of question where the ChatTS code decides without the judge (the multiple-choice ones);
the others are counted and left out.
"""

import argparse
import json
import math
import sys
import types
from pathlib import Path


def load_scorer(repository: str):
    """`evaluate_qa` of the ChatTS repository, loaded without its judge and without the packages only its
    command-line use needs."""
    source = (Path(repository) / "evaluation" / "evaluate_qa.py").read_text()
    quiet = types.SimpleNamespace(**{name: (lambda *args, **kwargs: None) for name in ("info", "warning", "error", "debug")})
    stubs = {
        "loguru": types.SimpleNamespace(logger=quiet),
        "tqdm": types.SimpleNamespace(tqdm=lambda iterable=None, **kwargs: iterable),
        "evaluation": types.ModuleType("evaluation"),
        "evaluation.ragas": types.ModuleType("evaluation.ragas"),
        # the judge: a score that is not a number marks the questions that needed it
        "evaluation.ragas.score": types.SimpleNamespace(calculate_ragas_score=lambda **kwargs: (math.nan, {})),
    }
    kept = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        module = types.ModuleType("chatts_evaluate_qa")
        exec(compile(source, "evaluate_qa.py", "exec"), module.__dict__)
    finally:
        for name, value in kept.items():
            sys.modules.pop(name) if value is None else sys.modules.__setitem__(name, value)
    return module.evaluate_qa


def score(evaluate_qa, dataset: list[dict], answers: dict[int, str]) -> dict:
    """The mean of each score, per kind of question and over all of them, as the ChatTS script computes them."""
    import contextlib
    import io

    kinds = ("categorical", "numerical", "reason")
    collected = {kind: {} for kind in kinds}
    failed = 0
    for index, sample in enumerate(dataset):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate_qa(answers.get(index, ""), sample)
        except Exception:  # the ChatTS script skips the records its parser fails on, too
            failed += 1
            continue
        for ability, values in result.items():
            for kind, scores in zip(kinds, values):
                collected[kind].setdefault(ability, []).extend(float(value) for value in scores)
    table = {"records": len(dataset), "answered": sum(index in answers for index in range(len(dataset))), "failed": failed}
    mean = lambda values: round(sum(values) / len(values), 4) if values else None
    for kind in kinds:
        per_ability = {}
        for ability, values in collected[kind].items():
            if kind == "reason" and any(math.isnan(value) for value in values):
                per_ability[ability] = f"needs a judge model ({len(values)} questions)"
            elif values:
                per_ability[ability] = mean(values)
        numbers = [value for values in collected[kind].values() for value in values if not math.isnan(value)]
        table[kind] = {"overall": mean(numbers), "questions": len(numbers), "by kind": per_ability}
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--chatts", required=True, help="a copy of https://github.com/NetManAIOps/ChatTS")
    parser.add_argument("--dataset", required=True, help="dataset_a.json or dataset_b.json")
    parser.add_argument("--out", help="write the tables to this JSON file")
    args = parser.parse_args(argv)
    evaluate_qa = load_scorer(args.chatts)
    dataset = json.loads(Path(args.dataset).read_text())
    tables = {}
    for path in args.predictions:
        with open(path) as lines:
            answers = {record["id"]: record.get("raw") or record["prediction"] for record in map(json.loads, filter(str.strip, lines))}
        tables[Path(path).stem] = score(evaluate_qa, dataset, answers)
    names = list(tables)
    print(f"{Path(args.dataset).name}: {len(dataset)} records\n")
    print(f"{'':44s}" + "".join(f"{name[:16]:>18s}" for name in names))
    for kind in ("categorical", "numerical"):
        rows = ["overall"] + sorted({ability for table in tables.values() for ability in table[kind]["by kind"]})
        for row in rows:
            cells = [table[kind]["overall"] if row == "overall" else table[kind]["by kind"].get(row) for table in tables.values()]
            print(f"{kind + ': ' + row:44s}" + "".join(f"{'-' if cell is None else format(cell, '.1%'):>18s}" for cell in cells))
        print()
    for name, table in tables.items():
        print(f"{name}: answered {table['answered']} of {table['records']}; reasoning score: {table['reason']['by kind']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(tables, indent=1))
    return tables


if __name__ == "__main__":
    main()
