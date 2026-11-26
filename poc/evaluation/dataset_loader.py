
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter

from poc.datasets.schema import (
    TestCase,
    DatasetSplit
)


DATASET_DIR = (
    Path(__file__).resolve().parents[1]
    / "datasets"
)


ADAPTER = TypeAdapter(list[TestCase])

DatasetEdition = Literal["original", "expanded"]


def load_cases(
    split: DatasetSplit = "development",
    *,
    dataset_edition: DatasetEdition = "original"
) -> list[TestCase]:

    if split not in {"development", "held_out"}:
        raise ValueError("Invalid dataset split")

    if dataset_edition not in {"original", "expanded"}:
        raise ValueError("Invalid dataset edition")

    dataset_dir = DATASET_DIR / "expanded" if dataset_edition == "expanded" else DATASET_DIR
    dataset_path = dataset_dir / f"{split}.json"

    cases = ADAPTER.validate_json(
        dataset_path.read_bytes()
    )

    if not cases:
        raise ValueError("Dataset is empty")

    if any(case.split != split for case in cases):
        raise ValueError("Dataset split mismatch")

    case_ids = [
        case.case_id
        for case in cases
    ]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case IDs")

    return cases


def load_case(
    case_id: str,
    split: DatasetSplit = "development",
    *,
    dataset_edition: DatasetEdition = "original"
) -> TestCase:

    cases = load_cases(split, dataset_edition=dataset_edition)

    for case in cases:

        if case.case_id == case_id:
            return case

    raise KeyError(
        f"Case not found in {split}: {case_id}"
    )
