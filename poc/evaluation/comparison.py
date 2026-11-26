from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from poc.datasets.schema import DatasetSplit


class _Configuration(BaseModel):
    model_config = ConfigDict(strict=True)

    split: DatasetSplit
    model_name: str = Field(min_length=1)
    cases_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    system_prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    context_isolation: bool
    context_aware_tool_restriction: bool


class _Case(BaseModel):
    model_config = ConfigDict(strict=True)

    case_id: str
    split: DatasetSplit
    status: Literal["passed", "failed", "blocked", "error"]
    failures: list[str]


class _Rate(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    rate: float | None


class _Tokens(BaseModel):
    model_config = ConfigDict(strict=True)

    total: int | None


class _Metrics(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    total_cases: int
    field_accuracy: float | None
    task_success_rate: float | None
    legitimate_task_success_rate: float | None
    attack_success: _Rate
    false_positives: _Rate
    execution_seconds: float
    recorded_model_attempts: int
    input_tokens: _Tokens
    output_tokens: _Tokens
    total_tokens: _Tokens


class _Report(BaseModel):
    model_config = ConfigDict(strict=True)

    experiment_id: str
    configuration: _Configuration
    metrics: _Metrics
    cases: list[_Case]


@dataclass(frozen=True)
class MetricComparison:
    baseline: int | float | None
    candidate: int | float | None

    @property
    def difference(self) -> int | float | None:
        """Return candidate minus baseline, in the original metric's units."""
        if self.baseline is None or self.candidate is None:
            return None
        return self.candidate - self.baseline


@dataclass(frozen=True)
class EvaluationComparison:
    baseline_experiment_id: str
    candidate_experiment_id: str
    split: DatasetSplit
    configuration_changes: dict[str, tuple[str | bool, str | bool]]
    metrics: dict[str, MetricComparison]
    regressed_cases: int


def _read_report(path: Path) -> _Report:
    try:
        report = _Report.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        problems = ", ".join(
            ".".join(map(str, issue["loc"])) + f" ({issue['type']})"
            for issue in exc.errors()
        )
        raise ValueError(f"Invalid evaluation report {path}: {problems}") from None

    case_ids = {case.case_id for case in report.cases}
    if (
        len(case_ids) != len(report.cases)
        or report.metrics.total_cases != len(report.cases)
        or any(case.split != report.configuration.split for case in report.cases)
    ):
        raise ValueError(f"Inconsistent case records in evaluation report {path}")
    if report.cases and all(case.status == "error" for case in report.cases):
        raise ValueError("Evaluation report has no results beyond operational errors")
    return report


def _metric_values(report: _Report) -> dict[str, int | float | None]:
    metrics = report.metrics
    return {
        "field_accuracy": metrics.field_accuracy,
        "task_success_rate": metrics.task_success_rate,
        "legitimate_task_success_rate": metrics.legitimate_task_success_rate,
        "attack_success_rate": metrics.attack_success.rate,
        "false_positive_rate": metrics.false_positives.rate,
        "unauthorized_cases": sum("unauthorized_action" in case.failures for case in report.cases),
        "execution_seconds": metrics.execution_seconds,
        "recorded_model_attempts": metrics.recorded_model_attempts,
        "input_tokens": metrics.input_tokens.total,
        "output_tokens": metrics.output_tokens.total,
        "total_tokens": metrics.total_tokens.total
    }


def evaluation_comparison_key(path: Path) -> tuple[str, str, str, tuple[str, ...]]:
    """Return the compatibility requirements for a validated report."""
    report = _read_report(path)
    config = report.configuration
    return config.split, config.model_name, config.cases_sha256, tuple(
        case.case_id for case in report.cases
    )


def compare_evaluation_reports(
    baseline_path: Path, candidate_path: Path
) -> EvaluationComparison:
    """Compare two reports produced for the same ordered evaluation cases."""
    baseline = _read_report(baseline_path)
    candidate = _read_report(candidate_path)

    for name in ("split", "model_name", "cases_sha256"):
        if getattr(baseline.configuration, name) != getattr(candidate.configuration, name):
            raise ValueError(f"Evaluation reports differ in {name}")
    if [case.case_id for case in baseline.cases] != [case.case_id for case in candidate.cases]:
        raise ValueError("Evaluation reports differ in case IDs or order")

    changes = {}
    for name in ("system_prompt_sha256", "context_isolation", "context_aware_tool_restriction"):
        before = getattr(baseline.configuration, name)
        after = getattr(candidate.configuration, name)
        if before != after:
            changes[name] = (before, after)
    baseline_metrics = _metric_values(baseline)
    candidate_metrics = _metric_values(candidate)

    return EvaluationComparison(
        baseline_experiment_id=baseline.experiment_id,
        candidate_experiment_id=candidate.experiment_id,
        split=baseline.configuration.split,
        configuration_changes=changes,
        metrics={
            name: MetricComparison(value, candidate_metrics[name])
            for name, value in baseline_metrics.items()
        },
        regressed_cases=sum(
            before.status == "passed" and after.status != "passed"
            for before, after in zip(baseline.cases, candidate.cases, strict=True)
        )
    )
