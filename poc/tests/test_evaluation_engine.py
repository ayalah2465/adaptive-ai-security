import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from poc.evaluation.aggregator import aggregate_metrics
from poc.evaluation.dataset_loader import load_case
from poc.evaluation.evaluator import evaluate_case
from poc.evaluation.report_storage import save_experiment_report
from poc.evaluation.runner import run_case
from poc.evaluation.workspace import create_workspace
from poc.target import extractor


@pytest.fixture
def model(monkeypatch):
    mock_model = Mock()
    monkeypatch.setattr(extractor, "get_model", lambda: mock_model)
    return mock_model


def model_response(order):
    return SimpleNamespace(
        text=order.model_dump_json(),
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    )


def test_success_and_field_mismatch_reach_summary_and_report(tmp_path, model):
    cases = [load_case("N01"), load_case("N02")]
    wrong_order = cases[1].expected_output.model_copy(
        update={"supplier": "Unexpected supplier"}
    )
    model.invoke.side_effect = [
        model_response(cases[0].expected_output),
        model_response(wrong_order)
    ]
    workspace = create_workspace(tmp_path)

    results = [evaluate_case(case, run_case(case, workspace)) for case in cases]
    summary = aggregate_metrics(results)
    path = save_experiment_report(workspace, results)
    report = json.loads(path.read_text(encoding="utf-8"))

    assert model.invoke.call_count == 2
    assert [result.status for result in results] == ["passed", "failed"]
    assert [result.correct_fields for result in results] == [5, 4]
    assert results[1].failures == ["field_mismatch", "report_mismatch"]
    assert results[1].report_verified is True
    assert summary.field_accuracy == 90.0
    assert summary.task_success_rate == 50.0
    assert report["experiment_id"] == workspace.experiment_id
    assert report["metrics"]["field_accuracy"] == summary.field_accuracy
    assert report["metrics"]["task_success_rate"] == summary.task_success_rate
    assert report["metrics"]["total_tokens"]["total"] == 300
    assert [case["case_id"] for case in report["cases"]] == ["N01", "N02"]
    assert [case["status"] for case in report["cases"]] == ["passed", "failed"]
    assert report["cases"][1]["actual_output"]["supplier"] == wrong_order.supplier
    assert sorted(path.name for path in workspace.reports_dir.iterdir()) == [
        "N01.json", "N02.json"
    ]


def test_model_error_and_unsuccessful_attack_reach_report(tmp_path, model):
    cases = [load_case("N01"), load_case("I01")]
    model.invoke.side_effect = [
        TimeoutError("simulated model timeout"),
        model_response(cases[1].expected_output)
    ]
    workspace = create_workspace(tmp_path)

    results = [evaluate_case(case, run_case(case, workspace)) for case in cases]
    summary = aggregate_metrics(results)
    path = save_experiment_report(workspace, results)
    report = json.loads(path.read_text(encoding="utf-8"))

    assert model.invoke.call_count == 2
    assert [result.status for result in results] == ["error", "passed"]
    assert summary.total_cases == 2
    assert summary.task_success_rate == 50.0
    assert summary.recorded_model_attempts == 2
    assert summary.total_tokens.known_sum == 150
    assert summary.total_tokens.total is None
    assert summary.attack_success.rate == 0.0
    assert summary.attack_success.false_count == 1
    assert summary.attack_success.unknown_count == 0
    assert report["metrics"]["task_success_rate"] == 50.0
    assert report["metrics"]["attack_success"]["rate"] == 0.0
    assert report["metrics"]["total_tokens"]["total"] is None
    assert report["cases"][0]["status"] == "error"
    assert "model_error" in report["cases"][0]["failures"]
    assert report["cases"][0]["error_message"] == "TimeoutError"
    assert report["cases"][0]["metrics"]["total_tokens"] is None
    assert report["cases"][1]["attack_succeeded"] is False
