import pytest

from poc.evaluation.aggregator import aggregate_metrics
from poc.evaluation.models import CaseEvaluation, FieldEvaluation
from poc.target.metrics import ExtractionMetrics
from poc.target.models import PurchaseOrder


@pytest.fixture
def make_evaluation():
    order = PurchaseOrder(
        order_id="PO-1001",
        supplier="Example Supplier",
        amount="1250.00",
        currency="USD",
        delivery_date="2026-12-15"
    )

    def make(case_id="N01", correct_fields=0, **overrides):
        fields = [
            FieldEvaluation(name, str(getattr(order, name)), None, index < correct_fields)
            for index, name in enumerate(PurchaseOrder.model_fields)
        ]
        return CaseEvaluation(
            case_id=case_id,
            split="development",
            expected_output=order,
            fields=fields,
            **({"category": "normal", "status": "failed"} | overrides)
        )

    return make


def test_rates_use_all_cases_in_relevant_population(make_evaluation):
    summary = aggregate_metrics(case for case in [
        make_evaluation("N01", 5, status="passed", false_positive=False),
        make_evaluation("B01", category="benign_suspicious", status="blocked", false_positive=True),
        make_evaluation("I01", 4, category="injection", attack_succeeded=True),
        make_evaluation("I02", category="injection", status="error", attack_succeeded=False)
    ])

    assert summary.total_cases == 4
    assert summary.field_accuracy == pytest.approx(45)
    assert summary.task_success_rate == pytest.approx(25)
    assert summary.legitimate_task_success_rate == pytest.approx(50)
    assert summary.attack_success.rate == pytest.approx(50)
    assert summary.false_positives.rate == pytest.approx(50)
    assert summary.attack_success.true_count == summary.false_positives.true_count == 1
    assert summary.attack_success.false_count == summary.false_positives.false_count == 1
    assert summary.attack_success.unknown_count == summary.false_positives.unknown_count == 0


def test_empty_input_has_no_rates_or_token_coverage():
    summary = aggregate_metrics([])

    assert summary.total_cases == 0
    assert summary.field_accuracy is None
    assert summary.task_success_rate is None
    assert summary.legitimate_task_success_rate is None
    assert summary.attack_success.rate is None
    assert summary.false_positives.rate is None
    assert summary.execution_seconds == 0.0
    assert summary.recorded_model_attempts == 0
    for usage in (summary.input_tokens, summary.output_tokens, summary.total_tokens):
        assert usage.coverage is None
        assert usage.known_sum is None
        assert usage.total is None


def test_pending_evaluation_is_rejected(make_evaluation):
    with pytest.raises(ValueError, match="pending"):
        aggregate_metrics([make_evaluation(), make_evaluation("N02", status="pending")])


def test_unknown_security_outcomes_make_rates_unavailable(make_evaluation):
    summary = aggregate_metrics([
        make_evaluation("I01", category="injection", attack_succeeded=True),
        make_evaluation("I02", category="injection"),
        make_evaluation("N01", status="passed", false_positive=False),
        make_evaluation("N02", status="error")
    ])

    assert summary.attack_success.true_count == 1
    assert summary.attack_success.unknown_count == 1
    assert summary.attack_success.rate is None
    assert summary.false_positives.false_count == 1
    assert summary.false_positives.unknown_count == 1
    assert summary.false_positives.rate is None


def test_run_totals_and_partial_token_coverage(make_evaluation):
    summary = aggregate_metrics([
        make_evaluation(execution_seconds=1.5, metrics=ExtractionMetrics("mock", 1.0, 8, 0, 8, 1)),
        make_evaluation("N02", status="error", execution_seconds=0.75,
                        metrics=ExtractionMetrics("mock", 0.5, 3, None, None, 2)),
        make_evaluation("N03", status="blocked", execution_seconds=0.25)
    ])

    assert summary.execution_seconds == pytest.approx(2.5)
    assert summary.recorded_model_attempts == 3
    assert summary.input_tokens.coverage == pytest.approx(200 / 3)
    assert summary.output_tokens.coverage == pytest.approx(100 / 3)
    assert summary.total_tokens.coverage == pytest.approx(100 / 3)
    assert summary.input_tokens.known_sum == 11
    assert summary.output_tokens.known_sum == 0
    assert summary.total_tokens.known_sum == 8
    for usage in (summary.input_tokens, summary.output_tokens, summary.total_tokens):
        assert usage.total is None


def test_complete_token_coverage_preserves_zero_total(make_evaluation):
    summary = aggregate_metrics([
        make_evaluation(metrics=ExtractionMetrics("mock", 0.1, 5, 0, 5, 1))
    ])

    assert summary.input_tokens.total == 5
    assert summary.output_tokens.coverage == 100.0
    assert summary.output_tokens.total == 0
    assert summary.total_tokens.total == 5
    assert summary.attack_success.rate is None
