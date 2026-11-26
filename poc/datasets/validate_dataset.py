
from collections import Counter
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from poc.datasets.schema import TestCase


DATASET_DIR = Path(__file__).resolve().parent

ADAPTER = TypeAdapter(list[TestCase])

EXPECTED_COUNTS = {
    "development": {
        "normal": 4,
        "injection": 4,
        "benign_suspicious": 2,
        "complex": 2
    },
    "held_out": {
        "normal": 2,
        "injection": 2,
        "benign_suspicious": 2,
        "complex": 2
    }
}


def require(condition, message):

    if not condition:
        raise ValueError(message)


def load_dataset(split):

    path = DATASET_DIR / f"{split}.json"

    return ADAPTER.validate_json(
        path.read_bytes()
    )


def validate_split(cases, split):

    expected_counts = EXPECTED_COUNTS[split]

    require(
        len(cases) == sum(expected_counts.values()),
        f"Incorrect case count in {split}"
    )

    require(
        all(case.split == split for case in cases),
        f"Incorrect split assignment in {split}"
    )

    counts = Counter(
        case.category for case in cases
    )

    require(
        counts == Counter(expected_counts),
        f"Incorrect category distribution in {split}"
    )

    print(
        f"[PASS] {split}: "
        f"{len(cases)} valid cases"
    )

    for category, count in expected_counts.items():
        print(f"  {category}: {count}")


def validate_negative_cases(case):

    invalid_changes = [
        {"case_id": "INVALID"},
        {"category": "unknown"},
        {"split": "unknown"},
        {"contains_injection": True},
        {"document": ""},
        {"unexpected_field": "invalid"},
        {"allowed_actions": ["delete_file"]}
    ]

    for changes in invalid_changes:

        data = case.model_dump()
        data.update(changes)

        try:
            TestCase.model_validate(data)

        except ValidationError:
            continue

        raise ValueError(
            f"Invalid case was accepted: {changes}"
        )

    print(
        f"[PASS] Negative schema tests: "
        f"{len(invalid_changes)}/{len(invalid_changes)}"
    )


def main():

    development = load_dataset("development")
    held_out = load_dataset("held_out")

    validate_split(development, "development")
    validate_split(held_out, "held_out")

    all_cases = development + held_out

    require(
        len(all_cases) == 20,
        "Incorrect total case count"
    )

    print("[PASS] Total case count: 20")

    case_ids = [
        case.case_id for case in all_cases
    ]

    require(
        len(case_ids) == len(set(case_ids)),
        "Duplicate case IDs detected"
    )

    print("[PASS] All case IDs are unique")

    documents = [
        case.document for case in all_cases
    ]

    require(
        len(documents) == len(set(documents)),
        "Duplicate documents detected"
    )

    print("[PASS] All documents are unique")

    development_ids = {
        case.case_id for case in development
    }

    held_out_ids = {
        case.case_id for case in held_out
    }

    require(
        development_ids.isdisjoint(held_out_ids),
        "Dataset split overlap detected"
    )

    print("[PASS] No case ID overlap between splits")

    for case in all_cases:

        require(
            set(case.allowed_actions) == {
                "read_document",
                "save_report"
            },
            f"Invalid allowed actions: {case.case_id}"
        )

        require(
            len(case.allowed_actions) == 2,
            f"Duplicate allowed actions: {case.case_id}"
        )

    print("[PASS] Allowed actions validated")

    validate_negative_cases(development[0])

    print("\n=== DATASET VALIDATION PASSED ===")


if __name__ == "__main__":
    main()
