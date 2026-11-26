from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from poc.datasets.schema import TestCase
from poc.evaluation.dataset_loader import DatasetEdition
from poc.evaluation.experiment import evaluation_provenance
from poc.evaluation.models import CaseEvaluation

from .feedback_loop import DefenseCandidate


class _BaselineReport(BaseModel):
    model_config = ConfigDict(strict=True)
    experiment_id: str = Field(min_length=1)
    configuration: dict[str, str | int | bool]
    cases: list[CaseEvaluation]


def load_development_baseline(
    report_path: Path, cases: Sequence[TestCase], candidate: DefenseCandidate,
    *, dataset_edition: DatasetEdition = "original"
) -> tuple[str, list[CaseEvaluation]]:
    """Read reference results for the active configuration and development cases."""
    try:
        report = _BaselineReport.model_validate_json(report_path.read_bytes())
    except ValidationError as exc:
        fields = ", ".join(
            ".".join(map(str, issue["loc"])) or "JSON document"
            for issue in exc.errors()
        )
        raise ValueError(f"Invalid baseline report fields: {fields}") from None

    configuration = {"dataset_edition": "original", **report.configuration}
    expected = {
        **evaluation_provenance("development", cases, **candidate.model_dump()),
        "dataset_edition": dataset_edition
    }
    mismatches = [
        name for name, value in expected.items()
        if name not in configuration
        or type(configuration[name]) is not type(value)
        or configuration[name] != value
    ]
    if mismatches:
        raise ValueError("Baseline report differs in " + ", ".join(mismatches))
    if len(report.cases) != len(cases) or any(
        case.split != "development"
        or result.case_id != case.case_id
        or result.category != case.category
        or result.split != case.split
        or result.expected_output != case.expected_output
        or result.status == "pending"
        for case, result in zip(cases, report.cases)
    ):
        raise ValueError("Baseline report must contain final results in development case order")

    return report.experiment_id, report.cases
