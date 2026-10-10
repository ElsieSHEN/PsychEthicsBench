import json
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPECTED_COUNTS = {
    "mcq": 1377,
    "oeq": 2612,
    "principles": 392,
    "response_annotations": 100,
}
ANNOTATION_LABELS = {
    "conflict",
    "credentials",
    "confidentiality",
    "misinformation",
    "disrespect",
    "competence",
    "misconduct",
    "justification",
    "quality",
}


def read_jsonl(name):
    path = DATA / f"{name}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def datasets():
    return {name: read_jsonl(name) for name in EXPECTED_COUNTS}


def test_configuration_counts_are_exact(datasets):
    assert {name: len(records) for name, records in datasets.items()} == EXPECTED_COUNTS


@pytest.mark.parametrize(
    ("name", "pattern"),
    [
        ("mcq", r"mcq-\d{4}"),
        ("oeq", r"oeq-\d{4}"),
        ("response_annotations", r"response-\d{4}"),
        ("principles", r"(?:apa-coe|ahpra-coc|ranzcp-coe|ranzcp-coc|ranzcp-ps)::.+"),
    ],
)
def test_ids_are_stable_and_unique(datasets, name, pattern):
    key = "principle_id" if name == "principles" else "id"
    ids = [record[key] for record in datasets[name]]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(pattern, value) for value in ids)


def test_mcq_answers_and_principle_links_are_valid(datasets):
    principles = {record["principle_id"] for record in datasets["principles"]}
    resolved = 0
    for record in datasets["mcq"]:
        assert record["answer_indices"]
        assert len(record["answer_indices"]) == len(record["answer_letters"])
        assert len(record["answer_indices"]) == len(set(record["answer_indices"]))
        assert all(0 <= index < len(record["options"]) for index in record["answer_indices"])
        assert record["answer_letters"] == [
            chr(ord("A") + index) for index in record["answer_indices"]
        ]
        if record["principle_id"] is None:
            assert record["standard"] is None
        else:
            resolved += 1
            assert record["standard"] is not None
            assert record["principle_id"] in principles
    assert resolved == 1311
    assert len(datasets["mcq"]) - resolved == 66


def test_every_oeq_resolves_to_the_exact_principle(datasets):
    principles = {
        record["principle_id"]: record for record in datasets["principles"]
    }
    for record in datasets["oeq"]:
        principle = principles[record["principle_id"]]
        assert record["code"] == principle["code"]
        assert record["standard"] == principle["standard"]
        assert record["principle_label"] == principle["label"]
        assert record["principle_description"] == principle["description"]


def test_principles_expose_only_item_level_fields(datasets):
    expected_fields = {
        "principle_id",
        "code",
        "code_name",
        "standard",
        "label",
        "description",
    }
    assert all(set(record) == expected_fields for record in datasets["principles"])


def test_annotations_have_exact_binary_label_schema(datasets):
    for record in datasets["response_annotations"]:
        assert set(record["labels"]) == ANNOTATION_LABELS
        assert all(type(value) is int and value in (0, 1) for value in record["labels"].values())


def test_tracked_release_files_exclude_generated_caches():
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("tracked-file audit requires a Git checkout")
    tracked = result.stdout.split("\0")
    forbidden = [
        path
        for path in tracked
        if "__pycache__" in path or path.endswith(".pyc") or ".pytest_cache" in path
    ]
    assert forbidden == []


def test_documentation_declares_four_loadable_hugging_face_configs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\n")
    front_matter = readme.split("---", 2)[1]
    declared = set(re.findall(r"config_name:\s*([a-z_]+)", front_matter))
    paths = set(re.findall(r"path:\s*(data/[a-z_]+\.jsonl)", front_matter))
    assert declared == set(EXPECTED_COUNTS)
    assert paths == {f"data/{name}.jsonl" for name in EXPECTED_COUNTS}
    assert all((ROOT / path).is_file() for path in paths)


def test_documentation_exposes_required_release_metadata_and_files():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required_files = [
        "CITATION.cff",
        "LICENSE-CODE",
        "LICENSE-DATA",
        "THIRD_PARTY_NOTICES.md",
    ]
    assert all((ROOT / name).is_file() for name in required_files)
    assert "10.18653/v1/2026.findings-acl.1971" in readme
    assert "CC BY 4.0" in readme
    assert "MIT" in readme
    assert "THIRD_PARTY_NOTICES.md" in readme
    assert "non-clinical" in readme.lower()
    assert "66" in readme
    assert "load_dataset" in readme
    assert "Limitations" in readme


def test_documentation_defines_annotation_polarity_and_every_label():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("### `response_annotations`", 1)[1].split("## Prompts", 1)[0]
    assert "`0` means absent" in section
    assert "`1` means present" in section
    for label in sorted(ANNOTATION_LABELS):
        assert f"| `{label}` |" in section
    assert "legacy `Justification`" in section
    assert "annotation provenance" in section.lower()


def test_public_records_do_not_expose_legacy_ids():
    for name in EXPECTED_COUNTS:
        for record in read_jsonl(name):
            assert "legacy_id" not in record, f"legacy_id remains in {name}.jsonl"
