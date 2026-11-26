
from dataclasses import dataclass

from .models import PurchaseOrder


@dataclass(frozen=True)
class ExtractionMetrics:

    model_name: str

    invocation_seconds: float

    input_tokens: int | None

    output_tokens: int | None

    total_tokens: int | None

    model_calls: int


@dataclass(frozen=True)
class ExtractionResult:

    order: PurchaseOrder

    metrics: ExtractionMetrics

class ExtractionFailure(RuntimeError):

    def __init__(
        self,
        message: str,
        metrics: ExtractionMetrics,
        failure_type: str
    ):

        super().__init__(message)

        self.metrics = metrics

        self.failure_type = failure_type
