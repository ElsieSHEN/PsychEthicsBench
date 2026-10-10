"""Validate PsychEthicsBench MCQ prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} of {path} is not a JSON object")
            records.append(record)
    return records


def normalize_answer(raw: object, valid_letters: set[str]) -> frozenset[str]:
    if not isinstance(raw, list):
        raise ValueError("answer must be a JSON list of uppercase option letters")
    if any(not isinstance(value, str) or len(value) != 1 for value in raw):
        raise ValueError("answer entries must be single uppercase option letters")
    if any(value not in valid_letters for value in raw):
        raise ValueError(
            f"answer contains a letter outside the valid options {sorted(valid_letters)}"
        )
    if len(raw) != len(set(raw)):
        raise ValueError("answer contains a duplicate option letter")
    return frozenset(raw)


def validate_predictions(
    dataset: list[dict], predictions: list[dict], allow_partial: bool = False
) -> dict[str, frozenset[str]]:
    dataset_by_id = {}
    for item in dataset:
        item_id = item["id"]
        if item_id in dataset_by_id:
            raise ValueError(f"dataset contains duplicate ID: {item_id}")
        dataset_by_id[item_id] = item

    normalized = {}
    for position, prediction in enumerate(predictions, start=1):
        if "id" not in prediction or "answer" not in prediction:
            raise ValueError(f"prediction {position} must contain 'id' and 'answer'")
        item_id = prediction["id"]
        if item_id in normalized:
            raise ValueError(f"duplicate prediction ID: {item_id}")
        if item_id not in dataset_by_id:
            raise ValueError(f"unknown prediction ID: {item_id}")
        option_count = len(dataset_by_id[item_id]["options"])
        valid_letters = {chr(ord("A") + index) for index in range(option_count)}
        try:
            normalized[item_id] = normalize_answer(prediction["answer"], valid_letters)
        except ValueError as exc:
            raise ValueError(f"invalid answer for {item_id}: {exc}") from exc

    if not normalized:
        raise ValueError("prediction file contains no predictions")
    missing = sorted(set(dataset_by_id) - set(normalized))
    if missing and not allow_partial:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(f"missing predictions for {len(missing)} IDs: {preview}{suffix}")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_jsonl(args.data)
    predictions = load_jsonl(args.predictions)
    normalized = validate_predictions(dataset, predictions, args.allow_partial)
    print(
        json.dumps(
            {
                "valid": True,
                "predicted": len(normalized),
                "total": len(dataset),
                "partial": len(normalized) != len(dataset),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
