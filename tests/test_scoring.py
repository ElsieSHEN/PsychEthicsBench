from pathlib import Path

import pytest

from evaluation.run_hf import parse_mcq_answer, parse_mcq_output, render_prompt
from evaluation.validate_predictions import load_jsonl
from evaluation.score_mcq import exact_match, partial_credit, score_predictions
from evaluation.validate_predictions import normalize_answer, validate_predictions


DATASET = [
    {"id": "mcq-0001", "options": ["one", "two"], "answer_letters": ["A"]},
    {"id": "mcq-0002", "options": ["one", "two", "three"], "answer_letters": ["A", "C"]},
]
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.mark.parametrize(
    ("predicted", "expected", "exact", "partial"),
    [
        ({"A"}, {"A"}, 1.0, 1.0),
        ({"A"}, {"A", "C"}, 0.0, 0.5),
        ({"A", "B", "C"}, {"A", "C"}, 0.0, 0.0),
        ({"B"}, {"A", "C"}, 0.0, 0.0),
        (set(), {"A"}, 0.0, 0.0),
    ],
)
def test_item_scores_cover_exact_subset_and_incorrect_sets(predicted, expected, exact, partial):
    predicted_set = frozenset(predicted)
    expected_set = frozenset(expected)
    assert exact_match(predicted_set, expected_set) == exact
    assert partial_credit(predicted_set, expected_set) == partial


def test_normalize_answer_accepts_valid_uppercase_letters_and_empty_list():
    assert normalize_answer(["A", "C"], {"A", "B", "C"}) == frozenset({"A", "C"})
    assert normalize_answer([], {"A", "B"}) == frozenset()


@pytest.mark.parametrize("raw", ["A", ["a"], ["AA"], ["D"], [1], None])
def test_normalize_answer_rejects_malformed_values(raw):
    with pytest.raises(ValueError, match="answer"):
        normalize_answer(raw, {"A", "B", "C"})


def test_normalize_answer_rejects_duplicate_letters():
    with pytest.raises(ValueError, match="duplicate"):
        normalize_answer(["A", "A"], {"A", "B"})


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        ([{"id": "mcq-0001", "answer": ["A"]}, {"id": "mcq-0001", "answer": ["A"]}], "duplicate prediction ID"),
        ([{"id": "mcq-9999", "answer": ["A"]}], "unknown prediction ID"),
        ([{"id": "mcq-0001", "answer": ["A"]}], "missing predictions"),
    ],
)
def test_validate_predictions_rejects_bad_coverage(predictions, message):
    with pytest.raises(ValueError, match=message):
        validate_predictions(DATASET, predictions)


def test_validate_predictions_allows_explicit_partial_coverage():
    result = validate_predictions(
        DATASET, [{"id": "mcq-0002", "answer": ["A"]}], allow_partial=True
    )
    assert result == {"mcq-0002": frozenset({"A"})}


def test_score_predictions_reports_overall_and_answer_type_metrics():
    predictions = {
        "mcq-0001": frozenset({"A"}),
        "mcq-0002": frozenset({"A"}),
    }

    result = score_predictions(DATASET, predictions)

    assert result == {
        "coverage": {"predicted": 2, "total": 2, "fraction": 1.0, "partial": False},
        "overall": {"n": 2, "exact_match": 0.5, "partial_credit": 0.75},
        "single_answer": {"n": 1, "exact_match": 1.0, "partial_credit": 1.0},
        "multiple_answer": {"n": 1, "exact_match": 0.0, "partial_credit": 0.5},
    }


def test_score_predictions_labels_partial_coverage():
    result = score_predictions(DATASET, {"mcq-0002": frozenset({"A"})})

    assert result["coverage"] == {
        "predicted": 1,
        "total": 2,
        "fraction": 0.5,
        "partial": True,
    }
    assert result["overall"] == {"n": 1, "exact_match": 0.0, "partial_credit": 0.5}


def test_render_prompt_includes_question_and_every_mcq_option():
    record = {"question": "Choose safely.", "options": ["First", "Second"]}

    rendered = render_prompt("mcq", "Q: {question}\n{options}", record)

    assert rendered == "Q: Choose safely.\nA. First\nB. Second"


def test_render_prompt_includes_the_oeq_question():
    rendered = render_prompt(
        "oeq", "Respond in Australia:\n{question}", {"question": "What should I do?"}
    )

    assert rendered == "Respond in Australia:\nWhat should I do?"


def test_render_prompt_preserves_json_braces_in_reasoning_template():
    template = 'Return {{"reasoning": "...", "answer": []}}\n{question}\n{options}'

    rendered = render_prompt(
        "mcq", template, {"question": "Question", "options": ["Only option"]}
    )

    assert '{"reasoning": "...", "answer": []}' in rendered


def test_render_prompt_rejects_an_unsupported_task():
    with pytest.raises(ValueError, match="unsupported task"):
        render_prompt("classification", "{question}", {"question": "Question"})


def test_render_prompt_does_not_duplicate_canonical_option_letters():
    record = load_jsonl(DATA / "mcq.jsonl")[0]

    rendered = render_prompt("mcq", "{question}\n{options}", record)

    assert rendered.splitlines()[1:] == record["options"]


@pytest.mark.parametrize(
    ("task", "template_name"),
    [
        ("mcq", "mcq_raw_global.txt"),
        ("mcq", "mcq_raw_australia.txt"),
        ("mcq", "mcq_reasoning_global.txt"),
        ("mcq", "mcq_reasoning_australia.txt"),
        ("oeq", "oeq_global.txt"),
        ("oeq", "oeq_australia.txt"),
    ],
)
def test_each_shipped_prompt_is_compatible_with_its_task(task, template_name):
    template = (ROOT / "prompts" / template_name).read_text(encoding="utf-8")
    record = (
        {"question": "Question", "options": ["A. First", "B. Second"]}
        if task == "mcq"
        else {"question": "Question"}
    )

    assert "Question" in render_prompt(task, template, record)


def test_mcq_rejects_an_oeq_template_that_omits_options():
    template = (ROOT / "prompts" / "oeq_global.txt").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="options"):
        render_prompt("mcq", template, {"question": "Question", "options": ["A. First"]})


def test_oeq_rejects_an_mcq_template_that_requires_options():
    template = (ROOT / "prompts" / "mcq_raw_global.txt").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="options"):
        render_prompt("oeq", template, {"question": "Question"})


def test_parse_mcq_answer_accepts_the_prompted_json_list():
    assert parse_mcq_answer('["A", "C"]') == ["A", "C"]


@pytest.mark.parametrize("text", ["A", '{"answer": ["A"]}', '["a"]', '["AA"]'])
def test_parse_mcq_answer_rejects_nonconforming_model_output(text):
    with pytest.raises(ValueError, match="JSON list"):
        parse_mcq_answer(text)


def test_parse_mcq_output_uses_an_explicit_output_format():
    assert parse_mcq_output('["A"]', "list") == ["A"]
    assert parse_mcq_output('{"reasoning": {"A": "..."}, "answer": ["A"]}', "reasoning") == ["A"]


def test_parse_mcq_output_rejects_an_unknown_output_format():
    with pytest.raises(ValueError, match="output format"):
        parse_mcq_output('["A"]', "filename-dependent")
