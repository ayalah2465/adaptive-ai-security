import json
from unittest.mock import Mock

import pytest
from langchain_core.runnables import RunnableLambda

from poc.adaptive.configuration import ConfigurationStore
from poc.adaptive.experiment import AdaptiveExperiment
from poc.adaptive.feedback_loop import InvestigationProposal
from poc.evaluation.dataset_loader import load_cases
from poc.evaluation.evaluator import evaluate_case
from poc.evaluation.experiment import run_evaluation
from poc.target import extractor
from poc.tests.test_adaptive_approval_workflow import approval_inputs, model_response


@pytest.mark.parametrize(
    "decision,status,version",
    [("approve", "activated", 1), ("reject", "rejected", 0),
     ("revise", "revision_requested", 0)]
)
def test_experiment_reports_decision_and_active_version(
    tmp_path, approval_inputs, decision, status, version
):
    cases, baseline, investigation, model = approval_inputs
    proposal = investigation["proposal"]
    model.invoke.side_effect = [model_response(cases[0])] + [
        model_response(case) for case in cases
    ]
    advisor = Mock(return_value=proposal)
    store = ConfigurationStore()
    experiment = AdaptiveExperiment(store, workspace_dir=tmp_path)

    pending = experiment.start(
        baseline, RunnableLambda(advisor), base_version=0, max_rounds=2, diagnostic_budget=2
    )
    assert pending.status == "awaiting_decision"
    assert pending.report_path is None
    assert store.active.version == 0

    final = experiment.resume(pending.approval["candidate_id"], decision)

    assert final.status == status
    assert store.active.version == version
    assert model.invoke.call_count == 13
    advisor.assert_called_once()
    report = json.loads(final.report_path.read_text(encoding="utf-8"))
    assert report["split"] == "development"
    assert report["status"] == status
    assert report["active_version"] == version
    assert report["candidate_id"] == pending.approval["candidate_id"]
    assert report["approval_decision"] == {
        "approve": "approved", "reject": "rejected", "revise": "revision_requested"
    }[decision]
    assert report["advisor_rounds_used"] == report["diagnostic_cases_used"] == 1
    assert report["diagnostics"][0]["cases"][0]["case_id"] == cases[0].case_id
    assert report["regression"]["total_cases"] == 12
    assert not list(final.report_path.parent.glob("*.tmp"))


def test_experiment_reports_no_evidence_without_model_initialization(
    tmp_path, approval_inputs, monkeypatch
):
    cases, baseline, _, model = approval_inputs
    first = baseline[0]
    first.status = "pending"
    first.failures = []
    first.actual_output = cases[0].expected_output
    baseline[0] = evaluate_case(cases[0], first)
    get_model = Mock(side_effect=AssertionError("The model must remain uninitialized"))
    monkeypatch.setattr(extractor, "get_model", get_model)
    store = ConfigurationStore()
    experiment = AdaptiveExperiment(store, workspace_dir=tmp_path)

    final = experiment.start(
        baseline, base_version=0, max_rounds=1, diagnostic_budget=1
    )

    assert final.status == "no_evidence"
    report = json.loads(final.report_path.read_text(encoding="utf-8"))
    assert report["advisor_rounds_used"] == report["diagnostic_cases_used"] == 0
    assert report["diagnostics"] == []
    assert report["approval_decision"] is None
    assert report["active_version"] == 0
    model.invoke.assert_not_called()
    get_model.assert_not_called()


def test_default_advisor_requests_structured_development_proposal(tmp_path, approval_inputs):
    cases, baseline, investigation, model = approval_inputs
    model.with_structured_output.return_value.invoke.return_value = investigation["proposal"]
    model.invoke.side_effect = [model_response(cases[0])] + [
        model_response(case) for case in cases
    ]
    experiment = AdaptiveExperiment(ConfigurationStore(), workspace_dir=tmp_path)

    result = experiment.start(
        baseline, base_version=0, max_rounds=1, diagnostic_budget=1
    )

    assert result.status == "awaiting_decision"
    model.with_structured_output.assert_called_once_with(
        InvestigationProposal, method="json_schema", strict=True
    )
    structured_invoke = model.with_structured_output.return_value.invoke
    structured_invoke.assert_called_once()
    evidence = json.loads(structured_invoke.call_args.args[0][1].content)
    assert [item["case_id"] for item in evidence["baseline"]] == [
        case.case_id for case in cases
    ]
    assert all("document" not in item for item in evidence["baseline"])
    assert model.invoke.call_count == 13


def test_invalid_default_advisor_response_stops_before_diagnostics(tmp_path, approval_inputs):
    _, baseline, _, model = approval_inputs
    model.with_structured_output.return_value.invoke.return_value = {"candidate": {}}
    experiment = AdaptiveExperiment(ConfigurationStore(), workspace_dir=tmp_path)

    result = experiment.start(
        baseline, base_version=0, max_rounds=1, diagnostic_budget=1
    )

    assert result.status == "invalid_proposal"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["advisor_rounds_used"] == 1
    assert report["diagnostic_cases_used"] == 0
    model.invoke.assert_not_called()


def test_stale_baseline_is_rejected_before_investigation(tmp_path, approval_inputs):
    _, baseline, _, model = approval_inputs
    advisor = Mock()
    store = ConfigurationStore()
    store.rollback(0)
    experiment = AdaptiveExperiment(store, workspace_dir=tmp_path)

    with pytest.raises(ValueError, match="Baseline version is no longer active"):
        experiment.start(
            baseline, RunnableLambda(advisor), base_version=0,
            max_rounds=1, diagnostic_budget=1
        )

    advisor.assert_not_called()
    model.invoke.assert_not_called()


def test_start_from_matching_development_report_records_source(tmp_path, approval_inputs):
    cases, _, _, model = approval_inputs
    model.invoke.side_effect = [model_response(case) for case in cases]
    baseline = run_evaluation("development", max_cases=12, workspace_dir=tmp_path)
    advisor = Mock()
    experiment = AdaptiveExperiment(ConfigurationStore(), workspace_dir=tmp_path)

    result = experiment.start_from_report(
        baseline.report_path, RunnableLambda(advisor), max_rounds=1, diagnostic_budget=1
    )

    assert result.status == "no_evidence"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["baseline_report_id"] == baseline.experiment_id
    assert report["base_version"] == 0
    advisor.assert_not_called()
    assert model.invoke.call_count == 12


@pytest.mark.parametrize("dataset_edition,count", [("original", 12), ("expanded", 24)])
def test_saved_baseline_leads_to_human_approved_activation(
    tmp_path, approval_inputs, dataset_edition, count
):
    _, _, investigation, model = approval_inputs
    cases = load_cases("development", dataset_edition=dataset_edition)
    model.invoke.side_effect = [
        model_response(case, wrong=index == 0) for index, case in enumerate(cases)
    ]
    baseline = run_evaluation(
        "development", max_cases=count, workspace_dir=tmp_path, dataset_edition=dataset_edition
    )
    assert baseline.summary.task_success_rate == pytest.approx(100 * (count - 1) / count)

    model.with_structured_output.return_value.invoke.return_value = investigation["proposal"]
    model.invoke.side_effect = [model_response(cases[0])] + [
        model_response(case) for case in cases
    ]
    store = ConfigurationStore()
    experiment = AdaptiveExperiment(store, workspace_dir=tmp_path)

    if dataset_edition == "expanded":
        with pytest.raises(ValueError, match="dataset_edition"):
            experiment.start_from_report(baseline.report_path, max_rounds=1, diagnostic_budget=1)
        assert model.invoke.call_count == count
        model.with_structured_output.assert_not_called()

    pending = experiment.start_from_report(
        baseline.report_path, max_rounds=1, diagnostic_budget=1,
        dataset_edition=dataset_edition, max_regression_cases=count
    )

    assert pending.status == "awaiting_decision"
    assert pending.report_path is None
    assert store.active.version == 0
    assert pending.approval["regression"]["total_cases"] == count
    assert pending.approval["regression"]["dataset_edition"] == dataset_edition
    assert pending.approval["regression"]["task_success_rate"] == 100.0
    candidate_id = pending.approval["candidate_id"]

    with pytest.raises(ValueError, match="does not match the pending approval"):
        experiment.resume("wrong-candidate", "approve")
    assert store.active.version == 0

    final = experiment.resume(candidate_id, "approve")

    assert final.status == "activated"
    assert store.active.version == 1
    assert store.active.candidate_id == candidate_id
    model.with_structured_output.return_value.invoke.assert_called_once()
    assert model.invoke.call_count == 2 * count + 1
    report = json.loads(final.report_path.read_text(encoding="utf-8"))
    assert report["baseline_report_id"] == baseline.experiment_id
    assert report["dataset_edition"] == dataset_edition
    assert report["candidate_id"] == candidate_id
    assert report["approval_decision"] == "approved"
    assert report["active_version"] == 1
    assert report["diagnostics"][0]["cases"] == [
        {"case_id": cases[0].case_id, "status": "passed", "failures": []}
    ]
    assert report["regression"]["recorded_model_attempts"] == count


def test_compatible_legacy_baseline_defaults_to_original(tmp_path, approval_inputs):
    cases, _, _, model = approval_inputs
    model.invoke.side_effect = [model_response(case) for case in cases]
    baseline = run_evaluation("development", max_cases=12, workspace_dir=tmp_path)
    payload = json.loads(baseline.report_path.read_text(encoding="utf-8"))
    del payload["configuration"]["dataset_edition"]
    legacy_report = tmp_path / "legacy.json"
    legacy_report.write_text(json.dumps(payload), encoding="utf-8")
    experiment = AdaptiveExperiment(ConfigurationStore(), workspace_dir=tmp_path)

    result = experiment.start_from_report(legacy_report, max_rounds=1, diagnostic_budget=1)

    assert result.status == "no_evidence"
    assert json.loads(result.report_path.read_text(encoding="utf-8"))["dataset_edition"] == "original"
    model.with_structured_output.assert_not_called()


def test_held_out_report_is_rejected_before_model_call(tmp_path, approval_inputs):
    _, _, _, model = approval_inputs
    model.invoke.side_effect = [model_response(case) for case in load_cases("held_out")]
    baseline = run_evaluation("held_out", max_cases=8, workspace_dir=tmp_path)
    model.reset_mock()
    experiment = AdaptiveExperiment(ConfigurationStore(), workspace_dir=tmp_path)

    with pytest.raises(ValueError, match="differs in split"):
        experiment.start_from_report(
            baseline.report_path, max_rounds=1, diagnostic_budget=1
        )

    model.invoke.assert_not_called()
    model.with_structured_output.assert_not_called()


@pytest.mark.parametrize(
    "change,message",
    [
        ("legacy", "configuration"),
        ("prompt", "system_prompt_sha256"),
        ("defense", "context_isolation"),
        ("dataset", "cases_sha256"),
        ("model", "model_name"),
        ("order", "development case order"),
        ("pending", "development case order")
    ]
)
def test_incompatible_baseline_reports_are_rejected(
    tmp_path, approval_inputs, change, message
):
    cases, _, _, model = approval_inputs
    model.invoke.side_effect = [model_response(case) for case in cases]
    baseline = run_evaluation("development", max_cases=12, workspace_dir=tmp_path)
    payload = json.loads(baseline.report_path.read_text(encoding="utf-8"))
    if change == "legacy":
        del payload["configuration"]
    elif change == "order":
        payload["cases"].reverse()
    elif change == "pending":
        payload["cases"][0]["status"] = "pending"
    else:
        field, value = {
            "prompt": ("system_prompt_sha256", "0" * 64),
            "defense": ("context_isolation", True),
            "dataset": ("cases_sha256", "0" * 64),
            "model": ("model_name", "another-model")
        }[change]
        payload["configuration"][field] = value
    incompatible = tmp_path / "incompatible.json"
    incompatible.write_text(json.dumps(payload), encoding="utf-8")
    model.reset_mock()
    experiment = AdaptiveExperiment(ConfigurationStore(), workspace_dir=tmp_path)

    with pytest.raises(ValueError, match=message):
        experiment.start_from_report(incompatible, max_rounds=1, diagnostic_budget=1)

    model.invoke.assert_not_called()
    model.with_structured_output.assert_not_called()
