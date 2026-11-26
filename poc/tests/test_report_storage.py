import json

import pytest

from poc.evaluation import report_storage
from poc.evaluation.models import CaseEvaluation
from poc.evaluation.workspace import create_workspace
from poc.target.action_log import ActionEvent
from poc.target.metrics import ExtractionMetrics
from poc.target.models import PurchaseOrder


@pytest.fixture
def evaluation():
    order = PurchaseOrder(
        order_id="PO-1001",
        supplier="ספק לדוגמה",
        amount="1250.10",
        currency="ILS",
        delivery_date="2026-12-15"
    )
    return CaseEvaluation(
        case_id="I01",
        category="injection",
        split="development",
        expected_output=order,
        actual_output=order,
        status="passed",
        actions=[ActionEvent("save_report", "executed", "I01")],
        metrics=ExtractionMetrics("mock", 0.1, 0, None, None, 1),
        report_verified=True
    )


def test_saves_complete_utf8_report_in_experiment_root(tmp_path, evaluation):
    workspace = create_workspace(tmp_path)

    path = report_storage.save_experiment_report(workspace, iter([evaluation]))

    assert path == workspace.root_dir / "evaluation_report.json"
    content = path.read_text(encoding="utf-8")
    assert "ספק לדוגמה" in content
    payload = json.loads(content)
    assert payload["experiment_id"] == workspace.experiment_id
    assert payload["metrics"]["total_cases"] == 1
    assert payload["metrics"]["task_success_rate"] == 100
    assert payload["metrics"]["attack_success"] == {
        "true_count": 0, "false_count": 0, "unknown_count": 1, "rate": None
    }
    assert payload["metrics"]["input_tokens"]["total"] == 0
    assert payload["metrics"]["output_tokens"]["total"] is None
    case, = payload["cases"]
    assert case["case_id"] == "I01"
    assert case["expected_output"]["amount"] == "1250.10"
    assert case["actual_output"]["delivery_date"] == "2026-12-15"
    assert case["attack_succeeded"] is None
    assert case["metrics"]["output_tokens"] is None
    assert case["actions"] == [
        {"action": "save_report", "status": "executed", "resource": "I01"}
    ]
    assert list(workspace.reports_dir.iterdir()) == []
    assert {item.name for item in workspace.root_dir.iterdir()} == {
        "documents", "reports", "evaluation_report.json"
    }


def test_pending_case_leaves_no_report(tmp_path, evaluation):
    workspace = create_workspace(tmp_path)
    evaluation.status = "pending"

    with pytest.raises(ValueError, match="pending"):
        report_storage.save_experiment_report(workspace, [evaluation])

    assert {item.name for item in workspace.root_dir.iterdir()} == {"documents", "reports"}


def test_existing_report_is_not_replaced(tmp_path, evaluation):
    workspace = create_workspace(tmp_path)
    path = report_storage.save_experiment_report(workspace, [evaluation])
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        report_storage.save_experiment_report(workspace, [])

    assert path.read_bytes() == original
    assert not list(workspace.root_dir.glob(".evaluation_report-*.tmp"))


@pytest.mark.parametrize("failure_point", ["write", "publish"])
def test_failure_cleans_temporary_file(tmp_path, evaluation, monkeypatch, failure_point):
    workspace = create_workspace(tmp_path)

    def fail_write(payload, stream, **options):
        stream.write("{")
        raise OSError("write failed")

    def fail_publish(source, destination):
        assert json.loads(source.read_text(encoding="utf-8"))["cases"][0]["case_id"] == "I01"
        raise OSError("publish failed")

    if failure_point == "write":
        monkeypatch.setattr(report_storage.json, "dump", fail_write)
    else:
        monkeypatch.setattr(report_storage.os, "link", fail_publish)

    with pytest.raises(OSError, match=f"{failure_point} failed"):
        report_storage.save_experiment_report(workspace, [evaluation])

    assert {item.name for item in workspace.root_dir.iterdir()} == {"documents", "reports"}
