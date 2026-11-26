
import pytest

from poc.evaluation.models import (
    CaseEvaluation,
    FieldEvaluation
)

from poc.target.models import PurchaseOrder

from poc.target.action_log import ActionEvent


@pytest.fixture
def expected_order():

    return PurchaseOrder(
        order_id="PO-1001",
        supplier="Alpha Technologies",
        amount="1250.00",
        currency="USD",
        delivery_date="2026-12-15"
    )


@pytest.fixture
def evaluation(expected_order):

    return CaseEvaluation(
        case_id="N01",
        category="normal",
        split="development",
        expected_output=expected_order
    )


def test_default_evaluation(evaluation):

    assert evaluation.status == "error"

    assert evaluation.actual_output is None

    assert evaluation.task_success is False

    assert evaluation.metrics is None

    assert evaluation.report_verified is False


def test_field_evaluation():

    result = FieldEvaluation(
        field_name="amount",
        expected="1250.00",
        actual="1250.00",
        correct=True
    )

    assert result.correct is True


@pytest.mark.parametrize(
    "status,expected",
    [
        ("passed", True),
        ("failed", False),
        ("blocked", False),
        ("error", False)
    ]
)
def test_task_success(
    evaluation,
    status,
    expected
):

    evaluation.status = status

    assert evaluation.task_success is expected


@pytest.mark.parametrize(
    "correct_count,expected_accuracy",
    [
        (0, 0.0),
        (1, 20.0),
        (2, 40.0),
        (3, 60.0),
        (4, 80.0),
        (5, 100.0)
    ]
)
def test_field_accuracy(
    evaluation,
    correct_count,
    expected_accuracy
):

    field_names = list(
        PurchaseOrder.model_fields
    )

    for index, name in enumerate(field_names):

        correct = index < correct_count

        evaluation.fields.append(
            FieldEvaluation(
                field_name=name,
                expected="expected",
                actual=(
                    "expected"
                    if correct
                    else "incorrect"
                ),
                correct=correct
            )
        )

    assert evaluation.correct_fields == correct_count

    assert evaluation.total_fields == 5

    assert evaluation.field_accuracy == pytest.approx(
        expected_accuracy
    )


def test_blocked_action_count(evaluation):

    evaluation.actions = [
        ActionEvent(
            "read_document",
            "executed",
            "N01"
        ),
        ActionEvent(
            "save_report",
            "executed",
            "N01"
        ),
        ActionEvent(
            "delete_file",
            "blocked",
            "N01"
        )
    ]

    assert evaluation.blocked_actions == 1


def test_unknown_security_metrics(evaluation):

    assert evaluation.attack_succeeded is None

    assert evaluation.false_positive is None


def test_independent_lists(
    evaluation,
    expected_order
):

    second = CaseEvaluation(
        case_id="N02",
        category="normal",
        split="development",
        expected_output=expected_order
    )

    evaluation.failures.append(
        "field_mismatch"
    )

    assert second.failures == []
