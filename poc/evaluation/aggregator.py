from collections.abc import Iterable
from dataclasses import dataclass

from .models import CaseEvaluation


def _percentage(numerator: int, denominator: int) -> float | None:
    return numerator / denominator * 100 if denominator else None


@dataclass(frozen=True)
class SecurityMetric:
    true_count: int
    false_count: int
    unknown_count: int

    @property
    def rate(self) -> float | None:
        if self.unknown_count:
            return None
        return _percentage(self.true_count, self.true_count + self.false_count)


@dataclass(frozen=True)
class TokenUsage:
    known_cases: int
    total_cases: int
    known_sum: int | None

    @property
    def coverage(self) -> float | None:
        return _percentage(self.known_cases, self.total_cases)

    @property
    def total(self) -> int | None:
        if self.known_cases == self.total_cases:
            return self.known_sum
        return None


@dataclass(frozen=True)
class MetricsSummary:
    total_cases: int
    field_accuracy: float | None
    task_success_rate: float | None
    legitimate_task_success_rate: float | None
    attack_success: SecurityMetric
    false_positives: SecurityMetric
    execution_seconds: float
    recorded_model_attempts: int
    input_tokens: TokenUsage
    output_tokens: TokenUsage
    total_tokens: TokenUsage


def _security_metric(values: Iterable[bool | None]) -> SecurityMetric:
    outcomes = list(values)
    return SecurityMetric(outcomes.count(True), outcomes.count(False), outcomes.count(None))


def _token_usage(values: Iterable[int | None], total_cases: int) -> TokenUsage:
    known_values = [value for value in values if value is not None]
    return TokenUsage(
        known_cases=len(known_values),
        total_cases=total_cases,
        known_sum=sum(known_values) if known_values else None
    )


def aggregate_metrics(evaluations: Iterable[CaseEvaluation]) -> MetricsSummary:
    """Summarize final evaluations. Rates and coverage are percentages."""
    cases = list(evaluations)
    if any(case.status == "pending" for case in cases):
        raise ValueError("Cannot aggregate pending evaluations")

    legitimate = [case for case in cases if case.category != "injection"]
    metrics = [case.metrics for case in cases if case.metrics is not None]
    total_cases = len(cases)

    return MetricsSummary(
        total_cases=total_cases,
        field_accuracy=_percentage(
            sum(case.correct_fields for case in cases),
            sum(case.total_fields for case in cases)
        ),
        task_success_rate=_percentage(sum(case.task_success for case in cases), total_cases),
        legitimate_task_success_rate=_percentage(
            sum(case.task_success for case in legitimate), len(legitimate)
        ),
        attack_success=_security_metric(
            case.attack_succeeded for case in cases if case.category == "injection"
        ),
        false_positives=_security_metric(case.false_positive for case in legitimate),
        execution_seconds=sum((case.execution_seconds for case in cases), 0.0),
        recorded_model_attempts=sum(metric.model_calls for metric in metrics),
        input_tokens=_token_usage((metric.input_tokens for metric in metrics), total_cases),
        output_tokens=_token_usage((metric.output_tokens for metric in metrics), total_cases),
        total_tokens=_token_usage((metric.total_tokens for metric in metrics), total_cases)
    )
