
import json
from types import SimpleNamespace

from unittest.mock import Mock

import pytest

from poc.evaluation.dataset_loader import load_case

from poc.evaluation.workspace import create_workspace

from poc.evaluation import runner

from poc.target import application, extractor

from poc.target.models import PurchaseOrder

from poc.target.policy import PolicyViolation


@pytest.fixture
def case():

    return load_case("N01")


@pytest.fixture
def workspace(tmp_path):

    return create_workspace(tmp_path)


@pytest.fixture
def mock_model(monkeypatch, case):

    model = Mock()

    model.invoke.return_value = SimpleNamespace(
        text=case.expected_output.model_dump_json(),
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150
        }
    )

    monkeypatch.setattr(
        extractor,
        "get_model",
        lambda: model
    )

    return model


def test_workspace_creation(workspace):

    assert workspace.root_dir.is_dir()

    assert workspace.documents_dir.is_dir()

    assert workspace.reports_dir.is_dir()


def test_unique_workspaces(tmp_path):

    first = create_workspace(tmp_path)

    second = create_workspace(tmp_path)

    assert first.experiment_id != second.experiment_id

    assert first.root_dir != second.root_dir


def test_document_preparation(
    case,
    workspace
):

    runner.prepare_document(
        case,
        workspace
    )

    path = (
        workspace.documents_dir
        / "N01.txt"
    )

    assert path.read_text(
        encoding="utf-8"
    ) == case.document


def test_successful_execution(
    case,
    workspace,
    mock_model
):

    result = runner.run_case(
        case,
        workspace
    )

    assert result.status == "pending"

    assert result.actual_output == (
        case.expected_output
    )

    assert result.report_verified is True

    assert result.failures == []

    assert result.execution_seconds >= 0

    mock_model.invoke.assert_called_once()


def test_report_saved(
    case,
    workspace,
    mock_model
):

    runner.run_case(
        case,
        workspace
    )

    path = (
        workspace.reports_dir
        / "N01.json"
    )

    assert path.exists()

    saved_order = PurchaseOrder.model_validate_json(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert saved_order == case.expected_output


def test_action_log(
    case,
    workspace,
    mock_model
):

    result = runner.run_case(
        case,
        workspace
    )

    events = [
        (event.action, event.status)
        for event in result.actions
    ]

    assert events == [
        ("read_document", "requested"),
        ("read_document", "executed"),
        ("save_report", "requested"),
        ("save_report", "executed")
    ]

    assert result.blocked_actions == 0


def test_metrics_collected(
    case,
    workspace,
    mock_model
):

    result = runner.run_case(
        case,
        workspace
    )

    assert result.metrics is not None

    assert result.metrics.input_tokens == 100

    assert result.metrics.output_tokens == 50

    assert result.metrics.total_tokens == 150

    assert result.metrics.model_calls == 1


def test_custom_prompt(
    case,
    workspace,
    mock_model
):

    prompt = "Extract the purchase order."

    runner.run_case(
        case,
        workspace,
        system_prompt=prompt
    )

    messages = mock_model.invoke.call_args.args[0]

    assert messages[0].content == prompt

    assert messages[1].content == case.document


def test_run_case_enables_context_isolation(case, workspace, mock_model):
    result = runner.run_case(case, workspace, context_isolation=True)

    instruction, data = mock_model.invoke.call_args.args[0]
    assert instruction.content.endswith(extractor.CONTEXT_ISOLATION_INSTRUCTION)
    assert json.loads(data.content) == {"document": case.document}
    assert result.status == "pending"
    assert result.report_verified is True


def test_context_restriction_allows_required_actions(case, workspace, mock_model):
    result = runner.run_case(case, workspace, context_aware_tool_restriction=True)

    assert result.status == "pending"
    assert result.report_verified is True
    assert [event.action for event in result.actions if event.status == "executed"] == [
        "read_document", "save_report"
    ]


def test_context_restriction_blocks_save_without_permission(case, workspace, mock_model):
    case = case.model_copy(update={"allowed_actions": ["read_document"]})

    result = runner.run_case(case, workspace, context_aware_tool_restriction=True)

    assert result.status == "blocked"
    assert result.failures == ["policy_violation"]
    assert [(event.action, event.status) for event in result.actions] == [
        ("read_document", "requested"),
        ("read_document", "executed"),
        ("save_report", "requested"),
        ("save_report", "blocked")
    ]
    assert list(workspace.reports_dir.iterdir()) == []


def test_invalid_model_output(
    case,
    workspace,
    mock_model
):

    mock_model.invoke.return_value.text = "{}"

    result = runner.run_case(
        case,
        workspace
    )

    assert result.status == "failed"

    assert "invalid_output" in result.failures

    assert result.metrics.total_tokens == 150

    assert result.actual_output is None

    assert not list(
        workspace.reports_dir.iterdir()
    )


def test_model_timeout(
    case,
    workspace,
    mock_model
):

    mock_model.invoke.side_effect = TimeoutError(
        "Synthetic timeout"
    )

    result = runner.run_case(
        case,
        workspace
    )

    assert result.status == "error"

    assert "model_error" in result.failures

    assert result.metrics.model_calls == 1

    assert result.metrics.total_tokens is None

    assert not list(
        workspace.reports_dir.iterdir()
    )


def test_policy_violation(
    case,
    workspace,
    mock_model,
    monkeypatch
):

    def blocked_read(*args, **kwargs):

        raise PolicyViolation(
            "Synthetic policy violation"
        )

    monkeypatch.setattr(
        application,
        "read_document",
        blocked_read
    )

    result = runner.run_case(
        case,
        workspace
    )

    assert result.status == "blocked"

    assert "policy_violation" in result.failures

    mock_model.invoke.assert_not_called()


def test_storage_failure(
    case,
    workspace,
    mock_model,
    monkeypatch
):

    def failed_save(*args, **kwargs):

        raise OSError(
            "Synthetic storage failure"
        )

    monkeypatch.setattr(
        application,
        "save_report",
        failed_save
    )

    result = runner.run_case(
        case,
        workspace
    )

    assert result.status == "error"

    assert "storage_error" in result.failures

    assert result.metrics.total_tokens == 150
    assert result.actual_output == case.expected_output
    assert result.report_verified is False


def test_report_mismatch(
    case,
    workspace,
    mock_model,
    monkeypatch
):

    def incorrect_save(*args, **kwargs):

        wrong_order = case.expected_output.model_copy(
            update={"supplier": "Wrong Supplier"}
        )

        path = (
            workspace.reports_dir
            / "N01.json"
        )

        path.write_text(
            wrong_order.model_dump_json(),
            encoding="utf-8"
        )

        return path

    monkeypatch.setattr(
        application,
        "save_report",
        incorrect_save
    )

    result = runner.run_case(
        case,
        workspace
    )

    assert result.status == "failed"

    assert "report_mismatch" in result.failures

    assert result.report_verified is False


def test_no_document_overwrite(
    case,
    workspace,
    mock_model
):

    first = runner.run_case(
        case,
        workspace
    )

    second = runner.run_case(
        case,
        workspace
    )

    assert first.status == "pending"

    assert second.status == "error"

    assert "storage_error" in second.failures

    mock_model.invoke.assert_called_once()


def test_independent_experiments(
    case,
    tmp_path,
    mock_model
):

    first = create_workspace(tmp_path)

    second = create_workspace(tmp_path)

    first_result = runner.run_case(
        case,
        first
    )

    second_result = runner.run_case(
        case,
        second
    )

    assert first_result.status == "pending"

    assert second_result.status == "pending"

    assert (
        first.reports_dir / "N01.json"
    ).exists()

    assert (
        second.reports_dir / "N01.json"
    ).exists()

    assert mock_model.invoke.call_count == 2


def test_wrong_extraction_remains_pending(
    case,
    workspace,
    mock_model
):

    wrong_order = case.expected_output.model_copy(
        update={"supplier": "Wrong Supplier"}
    )

    mock_model.invoke.return_value.text = (
        wrong_order.model_dump_json()
    )

    result = runner.run_case(
        case,
        workspace
    )

    assert result.status == "pending"

    assert result.task_success is False

    assert result.report_verified is True

    assert result.actual_output != (
        case.expected_output
    )
