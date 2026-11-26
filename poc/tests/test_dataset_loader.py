
import pytest
from pydantic import ValidationError

from poc.evaluation import dataset_loader


@pytest.mark.parametrize("split,count", [("development", 24), ("held_out", 16)])
def test_loads_each_expanded_split_without_changing_original(split, count):
    original = dataset_loader.load_cases(split)
    expanded = dataset_loader.load_cases(split, dataset_edition="expanded")

    assert len(expanded) == count
    assert expanded[:len(original)] == original
    additional = expanded[len(original)]
    assert dataset_loader.load_case(
        additional.case_id, split, dataset_edition="expanded"
    ) == additional
    with pytest.raises(KeyError):
        dataset_loader.load_case(additional.case_id, split)


def test_rejects_unknown_dataset_edition():
    with pytest.raises(ValueError, match="Invalid dataset edition"):
        dataset_loader.load_cases(dataset_edition="unknown")

@pytest.mark.parametrize(
    "split,expected_count",
    [
        ("development", 12),
        ("held_out", 8)
    ]
)
def test_load_dataset(
    split,
    expected_count
):

    cases = dataset_loader.load_cases(split)

    assert len(cases) == expected_count

    assert all(
        case.split == split
        for case in cases
    )


def test_default_split():

    cases = dataset_loader.load_cases()

    assert len(cases) == 12

    assert all(
        case.split == "development"
        for case in cases
    )


def test_load_single_case():

    case = dataset_loader.load_case("N01")

    assert case.case_id == "N01"

    assert case.split == "development"


def test_missing_case():

    with pytest.raises(KeyError):

        dataset_loader.load_case("N99")


def test_held_out_case_not_in_development():

    with pytest.raises(KeyError):

        dataset_loader.load_case(
            "N05",
            "development"
        )


@pytest.mark.parametrize(
    "split",
    ["unknown", "../outside", ""]
)
def test_invalid_split(split):

    with pytest.raises(ValueError):

        dataset_loader.load_cases(split)


def test_missing_dataset(
    tmp_path,
    monkeypatch
):

    monkeypatch.setattr(
        dataset_loader,
        "DATASET_DIR",
        tmp_path
    )

    with pytest.raises(FileNotFoundError):

        dataset_loader.load_cases()


def test_invalid_dataset_json(
    tmp_path,
    monkeypatch
):

    monkeypatch.setattr(
        dataset_loader,
        "DATASET_DIR",
        tmp_path
    )

    path = tmp_path / "development.json"

    path.write_text(
        "Invalid JSON",
        encoding="utf-8"
    )

    with pytest.raises(ValidationError):

        dataset_loader.load_cases()


def test_empty_dataset(
    tmp_path,
    monkeypatch
):

    monkeypatch.setattr(
        dataset_loader,
        "DATASET_DIR",
        tmp_path
    )

    path = tmp_path / "development.json"

    path.write_text(
        "[]",
        encoding="utf-8"
    )

    with pytest.raises(
        ValueError,
        match="Dataset is empty"
    ):

        dataset_loader.load_cases()


def test_duplicate_case_ids(
    tmp_path,
    monkeypatch
):

    case = dataset_loader.load_case("N01")

    monkeypatch.setattr(
        dataset_loader,
        "DATASET_DIR",
        tmp_path
    )

    path = tmp_path / "development.json"

    path.write_bytes(
        dataset_loader.ADAPTER.dump_json(
            [case, case]
        )
    )

    with pytest.raises(
        ValueError,
        match="Duplicate case IDs"
    ):

        dataset_loader.load_cases()


def test_dataset_split_mismatch(
    tmp_path,
    monkeypatch
):

    case = dataset_loader.load_case(
        "N05",
        "held_out"
    )

    monkeypatch.setattr(
        dataset_loader,
        "DATASET_DIR",
        tmp_path
    )

    path = tmp_path / "development.json"

    path.write_bytes(
        dataset_loader.ADAPTER.dump_json(
            [case]
        )
    )

    with pytest.raises(
        ValueError,
        match="Dataset split mismatch"
    ):

        dataset_loader.load_cases()
