
from dataclasses import dataclass, field

from typing import Literal

from poc.datasets.schema import (
    Category,
    DatasetSplit
)

from poc.target.models import PurchaseOrder

from poc.target.metrics import ExtractionMetrics

from poc.target.action_log import ActionEvent


EvaluationStatus = Literal[
    "pending",
    "passed",
    "failed",
    "blocked",
    "error"
]


FailureType = Literal[
    "field_mismatch",
    "invalid_output",
    "unauthorized_action",
    "missing_action",
    "report_mismatch",
    "policy_violation",
    "model_error",
    "storage_error"
]


@dataclass(frozen=True)
class FieldEvaluation:

    field_name: str

    expected: str

    actual: str | None

    correct: bool


@dataclass
class CaseEvaluation:

    case_id: str

    category: Category

    split: DatasetSplit

    expected_output: PurchaseOrder

    actual_output: PurchaseOrder | None = None

    status: EvaluationStatus = "error"

    fields: list[FieldEvaluation] = field(
        default_factory=list
    )

    actions: list[ActionEvent] = field(
        default_factory=list
    )

    failures: list[FailureType] = field(
        default_factory=list
    )

    metrics: ExtractionMetrics | None = None

    error_message: str | None = None

    execution_seconds: float = 0.0

    report_verified: bool = False

    attack_succeeded: bool | None = None

    false_positive: bool | None = None

    @property
    def task_success(self) -> bool:

        return self.status == "passed"

    @property
    def correct_fields(self) -> int:

        return sum(
            item.correct
            for item in self.fields
        )

    @property
    def total_fields(self) -> int:

        return len(
            PurchaseOrder.model_fields
        )

    @property
    def field_accuracy(self) -> float:

        return (
            self.correct_fields
            / self.total_fields
        ) * 100

    @property
    def blocked_actions(self) -> int:

        return sum(
            event.status == "blocked"
            for event in self.actions
        )
