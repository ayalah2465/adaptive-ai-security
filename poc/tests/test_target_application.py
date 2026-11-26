from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from poc import run_target
from poc.target import application, extractor, storage
from poc.target.action_log import ActionLog
from poc.target.application import process_order
from poc.target.metrics import ExtractionFailure
from poc.target.models import PurchaseOrder
from poc.target.policy import ALLOWED_ACTIONS, ActionContext, PolicyViolation


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    documents = tmp_path / "documents"
    documents.mkdir()
    reports = tmp_path / "reports"
    document = "Synthetic purchase order"
    (documents / "order_001.txt").write_text(document, encoding="utf-8")
    order = PurchaseOrder(
        order_id="PO-1001", supplier="Example Supplier", amount="1250.00",
        currency="USD", delivery_date="2026-12-15"
    )
    model = Mock()
    model.invoke.return_value = SimpleNamespace(
        text=order.model_dump_json(),
        usage_metadata={"input_tokens": 8, "output_tokens": 4, "total_tokens": 12}
    )
    monkeypatch.setattr(extractor, "get_model", lambda: model)
    return documents, reports, document, order, model


def test_process_order_returns_validated_saved_order_and_metrics(inputs):
    documents, reports, document, order, model = inputs

    result = process_order(documents_dir=documents, reports_dir=reports)

    assert result.order == order
    assert result.report_path == reports / "order_001.json"
    assert PurchaseOrder.model_validate_json(result.report_path.read_bytes()) == order
    assert result.metrics.total_tokens == 12
    instruction, data = model.invoke.call_args.args[0]
    assert instruction.content == extractor.SYSTEM_PROMPT
    assert data.content == document


def test_restricted_order_binds_storage_actions_to_document(inputs, monkeypatch):
    documents, reports, _, _, _ = inputs
    read = Mock(wraps=storage.read_document)
    save = Mock(wraps=storage.save_report)
    monkeypatch.setattr(application, "read_document", read)
    monkeypatch.setattr(application, "save_report", save)

    process_order(
        documents_dir=documents, reports_dir=reports,
        context_aware_tool_restriction=True
    )

    context = ActionContext("order_001", ALLOWED_ACTIONS)
    assert read.call_args.kwargs["action_context"] == context
    assert save.call_args.kwargs["action_context"] == context


def test_blocked_read_does_not_invoke_model_or_create_report(inputs):
    documents, reports, _, _, model = inputs

    with pytest.raises(PolicyViolation):
        process_order("invalid-id", documents_dir=documents, reports_dir=reports)

    model.invoke.assert_not_called()
    assert not reports.exists()


def test_extraction_failure_preserves_metrics_and_creates_no_report(inputs):
    documents, reports, _, _, model = inputs
    model.invoke.return_value.text = "invalid JSON"

    with pytest.raises(ExtractionFailure) as error:
        process_order(documents_dir=documents, reports_dir=reports)

    assert error.value.failure_type == "invalid_output"
    assert error.value.metrics.total_tokens == 12
    assert not reports.exists()


def test_save_failure_propagates_without_returning_success(inputs):
    documents, reports, _, _, model = inputs
    reports.write_text("Existing file", encoding="utf-8")
    action_log = ActionLog()
    on_extracted = Mock()

    with pytest.raises(FileExistsError):
        process_order(
            documents_dir=documents, reports_dir=reports,
            action_log=action_log, on_extracted=on_extracted
        )

    model.invoke.assert_called_once()
    on_extracted.assert_called_once()
    assert on_extracted.call_args.args[0].metrics.total_tokens == 12
    assert [(event.action, event.status) for event in action_log.events] == [
        ("read_document", "requested"),
        ("read_document", "executed"),
        ("save_report", "requested"),
        ("save_report", "failed")
    ]
    assert reports.read_text(encoding="utf-8") == "Existing file"


def test_cli_uses_default_document_and_report_path(inputs, monkeypatch, capsys):
    documents, reports, _, order, model = inputs
    monkeypatch.setattr(storage, "DOCUMENTS_DIR", documents)
    monkeypatch.setattr(storage, "REPORTS_DIR", reports)

    run_target.main()

    output = capsys.readouterr().out
    assert "[PASS] Report saved" in output
    assert order.model_dump_json(indent=2) in output
    assert f"Report location: {reports / 'order_001.json'}" in output
    model.invoke.assert_called_once()


def test_cli_preserves_legacy_extraction_exception(inputs, monkeypatch):
    documents, reports, _, _, model = inputs
    monkeypatch.setattr(storage, "DOCUMENTS_DIR", documents)
    monkeypatch.setattr(storage, "REPORTS_DIR", reports)
    model.invoke.return_value.text = "invalid JSON"

    with pytest.raises(ValidationError):
        run_target.main()

    assert not reports.exists()
