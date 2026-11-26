import json
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from poc.evaluation import experiment
from poc.evaluation.dataset_loader import load_cases
from poc.evaluation.experiment import run_evaluation
from poc.evaluation.report_storage import save_experiment_report
from poc.evaluation.workspace import create_workspace
from poc.target import extractor


@pytest.fixture
def model(monkeypatch):
    replacement = Mock()
    monkeypatch.setattr(extractor, "get_model", lambda: replacement)
    return replacement


def response(case):
    return SimpleNamespace(
        text=case.expected_output.model_dump_json(),
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )


@pytest.mark.parametrize(
    "split,dataset_edition,count",
    [("development", "original", 12), ("held_out", "original", 8),
     ("development", "expanded", 24), ("held_out", "expanded", 16)]
)
def test_runs_only_requested_split_and_saves_complete_report(
    tmp_path, model, split, dataset_edition, count
):
    cases = load_cases(split, dataset_edition=dataset_edition)
    model.invoke.side_effect = [response(case) for case in cases]
    prompt = "Extract the purchase order."

    result = run_evaluation(
        split, max_cases=count, dataset_edition=dataset_edition,
        system_prompt=prompt, context_isolation=True,
        context_aware_tool_restriction=True, workspace_dir=tmp_path
    )

    assert model.invoke.call_count == count
    assert result.summary.total_cases == count
    assert result.summary.task_success_rate == 100.0
    assert result.summary.total_tokens.total == count * 15
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["experiment_id"] == result.experiment_id
    assert report["metrics"]["total_cases"] == count
    configuration = report["configuration"]
    assert len(configuration["cases_sha256"]) == 64
    assert {key: value for key, value in configuration.items() if key != "cases_sha256"} == {
        "split": split,
        "dataset_edition": dataset_edition,
        "max_cases": count,
        "model_name": extractor.MODEL_NAME,
        "context_isolation": True,
        "context_aware_tool_restriction": True,
        "system_prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest()
    }
    assert [(item["case_id"], item["split"]) for item in report["cases"]] == [
        (case.case_id, split) for case in cases
    ]
    instruction, data = model.invoke.call_args_list[0].args[0]
    assert instruction.content == f"{prompt}\n\n{extractor.CONTEXT_ISOLATION_INSTRUCTION}"
    assert json.loads(data.content) == {"document": cases[0].document}


def test_individual_model_failure_is_preserved_and_other_cases_run(tmp_path, model):
    cases = load_cases("development")
    model.invoke.side_effect = [TimeoutError("simulated timeout")] + [
        response(case) for case in cases[1:]
    ]

    result = run_evaluation("development", max_cases=12, workspace_dir=tmp_path)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["configuration"]["dataset_edition"] == "original"
    failed = report["cases"][0]
    assert model.invoke.call_count == 12
    assert failed["status"] == "error"
    assert failed["error_message"] == "TimeoutError"
    assert "model_error" in failed["failures"]
    assert failed["metrics"]["total_tokens"] is None
    assert result.summary.task_success_rate == pytest.approx(100 * 11 / 12)
    assert result.summary.total_tokens.total is None


@pytest.mark.parametrize("dataset_edition,count", [("original", 12), ("expanded", 24)])
def test_insufficient_budget_fails_before_creating_workspace(
    tmp_path, model, dataset_edition, count
):
    with pytest.raises(ValueError, match=f"all {count}"):
        run_evaluation(
            "development", max_cases=count - 1, dataset_edition=dataset_edition,
            workspace_dir=tmp_path
        )

    model.invoke.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_systemic_error_stops_before_publishing_report(tmp_path, model):
    class AuthenticationError(Exception):
        pass

    model.invoke.side_effect = AuthenticationError("simulated authentication failure")

    with pytest.raises(RuntimeError, match="AuthenticationError"):
        run_evaluation("development", max_cases=12, workspace_dir=tmp_path)

    model.invoke.assert_called_once()
    assert not list(tmp_path.glob("*/evaluation_report.json"))


def test_configuration_fingerprints_track_prompt_content_and_case_order(
    tmp_path, monkeypatch, model
):
    cases = load_cases("development")
    monkeypatch.setattr(experiment, "load_cases", lambda split, *, dataset_edition: cases)

    def configuration(prompt):
        model.invoke.side_effect = [response(case) for case in cases]
        result = run_evaluation(
            "development", max_cases=12, system_prompt=prompt, workspace_dir=tmp_path
        )
        return json.loads(result.report_path.read_text(encoding="utf-8"))["configuration"]

    initial = configuration("Extract the purchase order.")
    assert configuration("Extract the purchase order.") == initial
    changed_prompt = configuration("Extract purchase order fields.")
    assert changed_prompt["system_prompt_sha256"] != initial["system_prompt_sha256"]
    assert changed_prompt["cases_sha256"] == initial["cases_sha256"]

    original = cases[0]
    cases[0] = original.model_copy(update={"document": original.document + "\nAdditional note."})
    changed_document = configuration("Extract the purchase order.")
    assert changed_document["cases_sha256"] != initial["cases_sha256"]
    assert changed_document["system_prompt_sha256"] == initial["system_prompt_sha256"]

    cases[0] = original
    cases.reverse()
    reordered = configuration("Extract the purchase order.")
    assert reordered["cases_sha256"] != initial["cases_sha256"]


def test_reports_without_configuration_keep_existing_shape(tmp_path):
    workspace = create_workspace(tmp_path)
    report_path = save_experiment_report(workspace, [])
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert set(payload) == {"experiment_id", "metrics", "cases"}
