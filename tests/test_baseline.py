"""Tests the baseline runner without any real model: questions, prompts, the resumable run loop and the comparison."""

import itertools
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import baseline  # noqa: E402


QUESTION = {
    "id": 0,
    "task": "spike",
    "parts": [{"text": "There is a time series of length 4: "}, {"series": [1.0, 2.0, 9.0, 2.0]}, {"text": ". Where is the spike?"}],
    "reference": "There is an upward spike at point 2.",
    "format": baseline.ANSWER_FORMS["spike"],
}


class Echo:
    """Answers with the reference, or with a fixed sentence."""

    def __init__(self, fixed=None):
        self.fixed, self.calls = fixed, 0

    def answer(self, question):
        self.calls += 1
        return self.fixed or question["reference"]


@pytest.fixture
def questions(tmp_path):
    path = tmp_path / "questions.jsonl"
    made = baseline.main(["questions", "--out", str(path), "--examples", "40"])
    return path, made


def run_args(questions_path, out, **extra):
    values = {"baseline": "echo", "questions": str(questions_path), "out": str(out), "limit": None, "show": 0}
    return SimpleNamespace(**{**values, **extra})


def test_questions_are_the_held_out_reading_questions(questions):
    path, made = questions
    assert len(made) == 40 and baseline.read_jsonl(path) == made
    assert {question["task"] for question in made} <= set(baseline.ANSWER_FORMS)
    for question in made:
        assert [("series" in part) for part in question["parts"]] == [False, True, False]
        assert question["format"] == baseline.ANSWER_FORMS[question["task"]]
    # the same questions on every call: baselines run at different times must see the same ones
    assert baseline.main(["questions", "--out", str(path), "--examples", "40"]) == made


def test_render_and_instructions():
    assert baseline.render(QUESTION, lambda _, series: baseline.values_text(series)) == (
        "There is a time series of length 4: [1, 2, 9, 2]. Where is the spike?"
    )
    text = baseline.instructions(QUESTION)
    assert "numbered from 0" in text and "mean 3.5 and standard deviation 3.2" in text
    assert text.endswith(baseline.ANSWER_FORMS["spike"]) and "standardized" not in text
    blocks = {**QUESTION, "parts": [*QUESTION["parts"][:2], {"text": ". Give the means in standardized units."}]}
    assert "minus the mean" in baseline.instructions(blocks)
    two = {**QUESTION, "parts": [{"series": [1.0, 2.0]}, {"text": " and "}, {"series": [3.0, 5.0]}]}
    assert "Series 1 has mean 1.5" in baseline.instructions(two) and "Series 2 has mean 4" in baseline.instructions(two)


def test_strip_thinking():
    assert baseline.strip_thinking("<think>hmm\nhmm</think>\nPoint 3.") == "Point 3."
    assert baseline.strip_thinking("<think>never finished") == ""
    assert baseline.strip_thinking(" Point 3. ") == "Point 3."


def test_plot_png_draws_one_panel_per_series():
    from PIL import Image
    import io

    one = Image.open(io.BytesIO(baseline.plot_png([np.sin(np.arange(100) / 5)])))
    two = Image.open(io.BytesIO(baseline.plot_png([np.arange(10.0), np.arange(50.0)])))
    assert one.format == "PNG" and one.size == (1200, 450) and two.size == (1200, 900)


def test_opentslm_prompt_keeps_every_text_and_standardizes_the_series():
    before, series, after = baseline.OpenTslm.split(QUESTION)
    assert before == "There is a time series of length 4:"
    (label, values), = series
    assert label == "This is the time series, it has mean 3.5000 and std 3.2016:"
    assert abs(np.mean(values)) < 1e-9 and abs(np.std(values) - 1) < 1e-9
    assert after.startswith("Where is the spike? Points are numbered from 0.")
    two = {**QUESTION, "parts": [{"series": [1.0, 2.0]}, {"text": " and "}, {"series": [3.0, 3.0]}, {"text": " differ?"}]}
    before, series, after = baseline.OpenTslm.split(two)
    assert before == "" and series[1][0].startswith("and This is the time series") and series[1][1] == [0.0, 0.0]
    assert after.startswith("differ? Points")


def test_run_appends_and_resumes(questions, tmp_path, capsys):
    path, made = questions
    out = tmp_path / "preds" / "echo.jsonl"
    echo = Echo()
    first = baseline.run(run_args(path, out, limit=10, show=2), baseline=echo)
    assert echo.calls == 10 and [record["id"] for record in first] == list(range(10))
    assert capsys.readouterr().out.count("REFERENCE") == 2
    # a second run answers only what is missing
    everything = baseline.run(run_args(path, out), baseline=echo)
    assert echo.calls == 40 and len(everything) == 40
    baseline.run(run_args(path, out, baseline="ours", hint=False), baseline=None)  # nothing left: no model is loaded
    assert "ours: 40 of 40 questions already answered" in capsys.readouterr().out


def test_score_compares_baselines(questions, tmp_path, capsys):
    path, made = questions
    baseline.run(run_args(path, tmp_path / "right.jsonl"), baseline=Echo())
    baseline.run(run_args(path, tmp_path / "wrong.jsonl"), baseline=Echo("I cannot tell."))
    tables = baseline.main(
        ["score", str(path), str(tmp_path / "right.jsonl"), str(tmp_path / "wrong.jsonl"), "--out", str(tmp_path / "s.json")]
    )
    assert json.loads((tmp_path / "s.json").read_text()) == tables
    right, wrong = tables["right"]["scores"], tables["wrong"]["scores"]
    assert tables["right"]["questions"] == 40
    assert right["block means"] == {"count right": 1.0, "mean error": 0.0}
    assert all(value in (0.0, 1.0) for metrics in right.values() for value in metrics.values())
    assert wrong["block means"] == {"count right": 0.0}
    assert all(metrics.get("answered", 0.0) == 0.0 for metrics in wrong.values())
    printed = capsys.readouterr().out
    assert "block means: count right" in printed and "100%" in printed and "0%" in printed


def test_claude_sends_the_chart_and_the_values(monkeypatch):
    sent = {}

    class Messages:
        def create(self, **request):
            sent.update(request)
            blocks = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=" at point 2. ")]
            return SimpleNamespace(stop_reason=sent.get("stop", "end_turn"), content=blocks)

    fake = SimpleNamespace(Anthropic=lambda **_: SimpleNamespace(messages=Messages()))
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    claude = baseline.Claude(SimpleNamespace(model="claude-opus-5-5", input="both"))
    assert claude.answer(QUESTION) == "at point 2."
    image, text = sent["messages"][0]["content"]
    assert sent["model"] == "claude-opus-5-5" and image["source"]["media_type"] == "image/png"
    assert "[1, 2, 9, 2] (also shown in the chart)" in text["text"]
    claude.input = "text"
    claude.answer(QUESTION)
    assert [block["type"] for block in sent["messages"][0]["content"]] == ["text"]
    sent["stop"] = "refusal"
    assert claude.answer(QUESTION) == ""


def test_ours_answers_with_a_saved_checkpoint(tmp_path):
    from test_train import tiny_model, tiny_processor

    tiny_model().save_pretrained(tmp_path)
    tiny_processor().save_pretrained(tmp_path)
    ours = baseline.Ours(SimpleNamespace(model=str(tmp_path), hint=True, max_new_tokens=4))
    assert isinstance(ours.answer(QUESTION), str)


def test_parse_args_fills_the_default_model():
    args = baseline.parse_args(["run", "chatts", "--questions", "q", "--out", "o"])
    assert args.model == "bytedance-research/ChatTS-8B" and args.max_new_tokens == 768
    ours = baseline.parse_args(["run", "ours", "--questions", "q", "--out", "o"])
    assert ours.model is None and ours.max_new_tokens == 160


def test_final_answer_reads_what_follows_the_marker():
    reasoning = "The length is 83, so there are 11 blocks. " * 10
    assert baseline.final_answer(reasoning + "\nANSWER: The level shifts upward at point 40.") == "The level shifts upward at point 40."
    assert baseline.final_answer('ANSWER: draft\nANSWER: "There are no spikes."') == "There are no spikes."
    assert baseline.final_answer("There are no spikes.") == "There are no spikes."
    assert baseline.final_answer(reasoning) == ""


def test_run_extracts_the_answer_and_discards_older_instructions(questions, tmp_path, capsys):
    path, made = questions
    out = tmp_path / "talker.jsonl"
    # answers from before the instructions changed
    out.write_text("".join(json.dumps({"id": index, "prediction": "old", "seconds": 1.0}) + "\n" for index in range(3)))

    class Talker(Echo):
        instructed = True

        def answer(self, question):
            return "Let me think. " * 40 + "\nANSWER: " + super().answer(question)

    records = baseline.run(run_args(path, out, limit=5, show=1), baseline=Talker())
    assert "discarding 3 answers" in capsys.readouterr().out
    assert [record["prediction"] for record in records] == [question["reference"] for question in made[:5]]
    assert all(record["version"] == baseline.PROMPT_VERSION and record["raw"].startswith("Let me") for record in records)
    # the trained model's answers carry no instructions and are kept as they are
    ours = tmp_path / "ours.jsonl"
    ours.write_text(json.dumps({"id": 0, "prediction": "kept", "seconds": 1.0}) + "\n")
    kept = baseline.run(run_args(path, ours, baseline="ours", hint=False, limit=1), baseline=None)
    assert kept[0]["prediction"] == "kept"


def test_spike_set_has_three_equal_groups_and_scores_by_group(tmp_path, capsys):
    path = tmp_path / "spikes.jsonl"
    made = baseline.main(["questions", "--set", "spikes", "--out", str(path), "--examples", "30"])
    assert baseline.main(["questions", "--set", "spikes", "--out", str(path), "--examples", "30"]) == made
    groups = [question["group"] for question in made]
    assert {groups.count(group) for group in baseline.SPIKE_GROUPS} == {10}
    assert {question["task"] for question in made} <= {"spike", "spike count"}
    # none of them is a reading question the training score already uses
    reading = baseline.main(["questions", "--out", str(tmp_path / "reading.jsonl"), "--examples", "100"])
    assert not {question["reference"] + str(question["parts"][1]["series"][:3]) for question in made} & {
        question["reference"] + str(question["parts"][1]["series"][:3]) for question in reading
    }
    baseline.run(run_args(path, tmp_path / "right.jsonl"), baseline=Echo())
    tables = baseline.main(["score", str(path), str(tmp_path / "right.jsonl")])
    scores = tables["right"]["scores"]
    assert set(scores) == set(baseline.SPIKE_GROUPS)
    assert scores["one spike, moving series"]["exact right"] == 1.0 and scores["several spikes"]["all points right"] == 1.0
    assert "one spike, flat series: exact right" in capsys.readouterr().out


def test_run_batches_a_baseline_that_can_answer_several_at_once(questions, tmp_path, capsys):
    path, made = questions

    class Batched(Echo):
        sizes = []

        def answer_batch(self, batch):
            self.sizes.append(len(batch))
            return [question["reference"] for question in batch]

    out = tmp_path / "batched.jsonl"
    records = baseline.run(run_args(path, out, batch_size=16, show=2), baseline=Batched())
    assert Batched.sizes == [16, 16, 8] and [record["id"] for record in records] == list(range(40))
    assert [record["prediction"] for record in records] == [question["reference"] for question in made]
    printed = capsys.readouterr().out
    assert printed.count("REFERENCE") == 2 and "40/40" in printed
    # without a batch size it still answers one at a time
    Batched.sizes.clear()
    baseline.run(run_args(path, tmp_path / "single.jsonl"), baseline=Batched())
    assert Batched.sizes == []


def test_final_answer_accepts_the_answer_is_and_score_rereads_saved_replies(questions, tmp_path):
    reply = "The curve has a downward spike between point 8 and point 10, after an upward drift. So the answer is: There is a downward spike at point 8."
    assert baseline.final_answer(reply) == "There is a downward spike at point 8."
    path, made = questions
    spike = next(question for question in made if question["task"] in ("spike", "level shift"))
    record = {"id": spike["id"], "prediction": "", "raw": "Some reasoning about upward and downward moves. " * 8 + "So the answer is: " + spike["reference"], "seconds": 1.0, "version": baseline.PROMPT_VERSION}
    (tmp_path / "wordy.jsonl").write_text(json.dumps(record) + "\n")
    scores = baseline.main(["score", str(path), str(tmp_path / "wordy.jsonl")])["wordy"]["scores"][spike["task"]]
    assert scores["exact right"] == 1.0 and scores["direction right"] == 1.0


def test_chatts_evaluation_records_become_questions_and_export_back(tmp_path):
    records = [
        {
            "timeseries": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "cols": ["a", "b"],
            "question": 'There are 2 metrics. "a" is <ts><ts/> and "b" is <ts><ts/>. 1. What is the trend of a?',
            "answer": "1. increase",
            "attributes": [{"type": "increase"}],
            "ability_types": ["trend"],
        }
    ]
    dataset, path = tmp_path / "dataset.json", tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps(records))
    (question,) = baseline.main(["questions", "--set", "chatts", "--dataset", str(dataset), "--out", str(path)])
    assert [part.get("series") for part in question["parts"] if "series" in part] == records[0]["timeseries"]
    assert baseline.render(question, lambda *_: "<ts><ts/>") == records[0]["question"]
    assert baseline.instructions(question) == ""
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"id": 0, "prediction": "1. increase", "raw": "1. increase"}) + "\n")
    exported = baseline.main(["export", str(predictions), "--out", str(tmp_path / "exp" / "generated_answer.json")])
    assert exported == [{"idx": 0, "response": "1. increase"}]
    assert json.loads((tmp_path / "exp" / "generated_answer.json").read_text()) == exported


def test_normality_questions_have_known_answers_and_score_by_group(tmp_path, capsys):
    path = tmp_path / "normality.jsonl"
    made = baseline.main(["questions", "--set", "normality", "--out", str(path), "--examples", "10"])
    assert baseline.main(["questions", "--set", "normality", "--out", str(path), "--examples", "10"]) == made
    assert [question["reference"] for question in made] == ["Yes.", "No.", "No.", "No.", "Yes."] * 2
    for question in made:
        (series,) = baseline.all_series(question)
        threshold = float(re.search(r"greater than ([\d.e+-]+) is", question["parts"][2]["text"]).group(1))
        # from the lowest point to the highest: the widest reading of "fluctuation"
        assert (series.max() - series.min() < threshold) == (question["reference"] == "Yes.")
    assert baseline.score_normality("No.", "No, there is a spike.") == {"answered": 1.0, "right": 1.0}
    assert baseline.score_normality("No.", "Yes, it is normal.") == {"answered": 1.0, "right": 0.0}
    assert baseline.score_normality("Yes.", "The series is fine.") == {"answered": 0.0, "right": 0.0}
    always_yes = tmp_path / "habit.jsonl"
    always_yes.write_text(
        "".join(json.dumps({"id": q["id"], "prediction": "Yes, normal.", "raw": "Yes, normal.", "seconds": 1.0}) + "\n" for q in made)
    )
    tables = baseline.main(["score", str(path), str(always_yes)])
    scores = tables["habit"]["scores"]
    assert scores["nothing but noise"]["right"] == 1.0 and scores["one bump"]["right"] == 0.0


def test_tsb_windows_hold_one_labelled_anomaly_or_none(tmp_path, capsys):
    rng = np.random.default_rng(0)
    folder = tmp_path / "TSB-AD-U"
    folder.mkdir()
    for index, source in enumerate(["UCR", "SMD", "TODS"]):
        values, labels = rng.normal(size=3000), np.zeros(3000, dtype=int)
        labels[1500:1540] = 1
        values[1500:1540] += 9
        rows = "\n".join(f"{value},{label}" for value, label in zip(values, labels))
        (folder / f"00{index}_{source}_id_1_Facility_tr_500_1st_1500.csv").write_text("Data,Label\n" + rows + "\n")
    path = tmp_path / "tsb.jsonl"
    made = baseline.main(["questions", "--set", "tsb", "--dataset", str(tmp_path), "--out", str(path), "--examples", "8"])
    # the generated source is left out
    assert {question["source"] for question in made} == {"UCR", "SMD"}
    assert {question["group"] for question in made} == {"with a clear anomaly", "without an anomaly"}
    assert sum(question["segment"] is None for question in made) == len(made) // 2
    for question in made:
        (series,) = baseline.all_series(question)
        assert len(series) == 512
        if question["segment"]:
            low, high = question["segment"]
            assert high - low == 39 and series[low : high + 1].mean() > 5 and 0 <= low and high < 512
        else:
            assert series.max() < 6
    yes = next(question for question in made if question["segment"])
    low = yes["segment"][0]
    assert baseline.score_anomaly(yes, f"Yes, around point {low + 3}.") == {"answered": 1.0, "right": 1.0, "place right": 1.0}
    assert baseline.score_anomaly(yes, "Yes, around point 2.")["place right"] == 0.0
    assert baseline.score_anomaly(yes, "No.") == {"answered": 1.0, "right": 0.0, "place right": 0.0}


def test_values_questions_ask_for_real_values_on_generated_and_real_series(tmp_path):
    from lfm3_ts.training import ReadingTasks, written_value

    assert [written_value(*pair) for pair in ((5210.37, 340), (0.012345, 0.0034), (-141.62, 12.0), (3.14159, 1.0))] == ["5210", "0.0123", "-142", "3.1"]
    # the training stream of these questions
    stream = ReadingTasks(profile="values", seed=3)
    answers = [example["messages"][1]["content"] for example in itertools.islice(stream, 40)]
    assert {answer.split(" is ")[0].split(" are ")[0].split(",")[0][:9] for answer in answers} == {"The value", "Between p", "The block"}
    # the test: half on the windows of a file of real questions, with the answer taken from the series
    rng = np.random.default_rng(0)
    real = [{"id": k, "parts": [{"text": "x"}, {"series": (1000 + 50 * rng.normal(size=300)).tolist()}]} for k in range(3)]
    path = tmp_path / "real.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in real))
    made = baseline.values_questions(40, str(path))
    assert sorted({question["group"] for question in made}) == ["generated series, extremes", "generated series, value", "real series, extremes", "real series, value"]
    for question in made:
        series = np.asarray(baseline.all_series(question)[0])
        # the reference answer scores as right, a value two standard deviations off does not
        scores = baseline.score_values(question, question["reference"])
        assert all(value == 1.0 for name, value in scores.items() if name.endswith("right")), (question["reference"], scores)
        if "value" in question["targets"]:
            point = question["targets"]["point"]
            assert question["targets"]["value"] == series[point]
            wrong = f"The value at point {point} is about {series[point] + 2 * series.std():.6f}."
            assert baseline.score_values(question, wrong)["value 0.25 sd right"] == 0.0
        else:
            first, last = question["targets"]["stretch"]
            assert question["targets"]["low"] == series[first : last + 1].min()
    assert baseline.score_values(made[0], "I cannot tell.")["answered"] == 0.0
    assert baseline.score_values(made[1], "The lowest value is about 3 at point 7.")["high point right"] == 0.0


def test_chatts_score_uses_the_scoring_code_of_chatts_without_its_judge(tmp_path, capsys):
    import chatts_score

    # a stand-in for `evaluation/evaluate_qa.py` of the ChatTS repository, with the imports the real one has
    (tmp_path / "ChatTS" / "evaluation").mkdir(parents=True)
    (tmp_path / "ChatTS" / "evaluation" / "evaluate_qa.py").write_text(
        "from loguru import logger\nfrom tqdm import tqdm\nfrom evaluation.ragas.score import calculate_ragas_score\n"
        "def evaluate_qa(answer, sample):\n"
        "    right = float(sample['attributes'][0] in answer)\n"
        "    judged, _ = calculate_ragas_score(question='', response=answer, label='')\n"
        "    return {'trend': ([right], [right / 2], [], []), 'local-inductive': ([right], [], [judged], [{}]), 'causal': ([], [], [1.0], [{}])}\n"
    )
    dataset = [{"attributes": ["up"], "ability_types": ["trend"]}, {"attributes": ["down"], "ability_types": ["trend"]}]
    (tmp_path / "set.json").write_text(json.dumps(dataset))
    (tmp_path / "ours.jsonl").write_text('{"id": 0, "prediction": "1. up"}\n{"id": 1, "prediction": "1. up"}\n')
    tables = chatts_score.main([str(tmp_path / "ours.jsonl"), "--chatts", str(tmp_path / "ChatTS"), "--dataset", str(tmp_path / "set.json")])
    table = tables["ours"]
    assert table["answered"] == 2 and table["categorical"]["by kind"]["trend"] == 0.5 and table["numerical"]["overall"] == 0.25
    # the questions the judge would score are named and left out; the ones scored by rule are kept
    assert "judge" in table["reason"]["by kind"]["local-inductive"] and table["reason"]["by kind"]["causal"] == 1.0
    assert "categorical: trend" in capsys.readouterr().out
    assert "loguru" not in sys.modules or hasattr(sys.modules["loguru"], "__file__")
