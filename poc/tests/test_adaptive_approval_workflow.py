from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langgraph.types import Command

from poc.adaptive import approval_workflow
from poc.adaptive.approval_workflow import build_approval_workflow
from poc.adaptive.feedback_loop import InvestigationProposal
from poc.evaluation.dataset_loader import load_cases
from poc.evaluation.evaluator import evaluate_case
from poc.evaluation.models import CaseEvaluation
from poc.target import extractor
from poc.target.action_log import ActionEvent


@pytest.fixture
def approval_inputs(tmp_path, monkeypatch):
    cases = load_cases("development")

    def evaluate(case, order):
        return evaluate_case(case, CaseEvaluation(
            case_id=case.case_id,
            category=case.category,
            split=case.split,
            expected_output=case.expected_output,
            actual_output=order,
            status="pending",
            actions=[
                ActionEvent("read_document", "executed", case.case_id),
                ActionEvent("save_report", "executed", case.case_id)
            ],
            report_verified=True
        ))

    first = cases[0]
    wrong_order = first.expected_output.model_copy(update={"supplier": "Other supplier"})
    baseline = [evaluate(first, wrong_order)] + [
        evaluate(case, case.expected_output) for case in cases[1:]
    ]
    proposal = InvestigationProposal.model_validate({
        "hypothesis": "Clarifying the supplier field may improve the result.",
        "diagnostic_case_ids": [first.case_id],
        "candidate": {
            "system_prompt": "Extract the labeled supplier and all other required fields.",
            "context_isolation": True,
            "context_aware_tool_restriction": True
        }
    })
    investigation = {
        "status": "candidate_tested",
        "proposal": proposal,
        "attempts": [{
            "proposal": proposal,
            "evaluations": [evaluate(first, first.expected_output)],
            "workspace": tmp_path
        }],
        "remaining_diagnostics": 0,
        "remaining_rounds": 0
    }
    model = Mock()
    monkeypatch.setattr(extractor, "get_model", lambda: model)
    return cases, baseline, investigation, model


def model_response(case, *, wrong=False):
    order = case.expected_output
    if wrong:
        order = order.model_copy(update={"supplier": "Unexpected supplier"})
    return SimpleNamespace(text=order.model_dump_json(), usage_metadata={})


@pytest.mark.parametrize(
    "decision,matching_id,status",
    [
        ("approve", True, "approved"),
        ("reject", True, "rejected"),
        ("revise", True, "revision_requested"),
        ("approve", False, "invalid_decision"),
        ("unsupported", True, "invalid_decision")
    ]
)
def test_regression_interrupt_and_resume(tmp_path, approval_inputs, decision, matching_id, status):
    cases, baseline, investigation, model = approval_inputs
    model.invoke.side_effect = [model_response(case) for case in cases]
    graph = build_approval_workflow(investigation, baseline, workspace_dir=tmp_path)
    config = {"configurable": {"thread_id": "approval-test"}}

    paused = graph.invoke({}, config=config)

    assert paused["status"] == "awaiting_decision"
    pending, = paused["__interrupt__"]
    assert pending.value["regression"]["total_cases"] == 12
    assert pending.value["regression"]["field_accuracy"] == 100.0
    assert pending.value["candidate"] == investigation["proposal"].candidate.model_dump()
    candidate_id = pending.value["candidate_id"] if matching_id else "different-candidate"

    final = graph.invoke(
        Command(resume={"candidate_id": candidate_id, "decision": decision}),
        config=config
    )

    assert final["status"] == status
    assert model.invoke.call_count == 12
    assert len(list(tmp_path.iterdir())) == 1


def test_candidate_regression_prevents_approval(tmp_path, approval_inputs):
    cases, baseline, investigation, model = approval_inputs
    model.invoke.side_effect = [
        model_response(case, wrong=index == 1) for index, case in enumerate(cases)
    ]
    graph = build_approval_workflow(investigation, baseline, workspace_dir=tmp_path)

    result = graph.invoke({}, config={"configurable": {"thread_id": "regression-test"}})

    assert result["status"] == "regression_failed"
    assert "__interrupt__" not in result
    assert model.invoke.call_count == 12


def test_incomplete_baseline_is_rejected(tmp_path, approval_inputs):
    _, baseline, investigation, model = approval_inputs

    with pytest.raises(ValueError, match="complete development split"):
        build_approval_workflow(investigation, baseline[:-1], workspace_dir=tmp_path)

    model.invoke.assert_not_called()


def test_untested_candidate_is_rejected(tmp_path, approval_inputs):
    _, baseline, investigation, model = approval_inputs
    investigation["status"] = "budget_exhausted"

    with pytest.raises(ValueError, match="tested candidate"):
        build_approval_workflow(investigation, baseline, workspace_dir=tmp_path)

    model.invoke.assert_not_called()


def test_regression_uses_entire_expanded_development_split(
    tmp_path, approval_inputs, monkeypatch
):
    cases, baseline, investigation, model = approval_inputs
    additional = cases[1].model_copy(update={
        "case_id": "N99", "document": cases[1].document + "\nAdditional synthetic note."
    })
    expanded_cases = [*cases, additional]
    expanded_baseline = [*baseline, replace(baseline[1], case_id=additional.case_id)]
    monkeypatch.setattr(
        approval_workflow, "load_cases", lambda split, *, dataset_edition: expanded_cases
    )
    model.invoke.side_effect = [model_response(case) for case in expanded_cases]
    graph = build_approval_workflow(investigation, expanded_baseline, workspace_dir=tmp_path)

    result = graph.invoke({}, config={"configurable": {"thread_id": "expanded-regression"}})

    assert result["status"] == "awaiting_decision"
    assert result["regression"]["total_cases"] == 13
    assert model.invoke.call_count == 13


def test_regression_rejects_a_split_larger_than_the_operator_budget(tmp_path, approval_inputs):
    _, baseline, investigation, model = approval_inputs

    with pytest.raises(ValueError, match="exceeds the approved regression budget"):
        build_approval_workflow(
            investigation, baseline, workspace_dir=tmp_path, max_regression_cases=11
        )

    assert list(tmp_path.iterdir()) == []
    model.invoke.assert_not_called()
