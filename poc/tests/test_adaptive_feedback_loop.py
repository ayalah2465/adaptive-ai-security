import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.runnables import RunnableLambda

from poc.adaptive.feedback_loop import InvestigationProposal, run_feedback_loop
from poc.evaluation.dataset_loader import load_case
from poc.evaluation.evaluator import evaluate_case
from poc.evaluation.models import CaseEvaluation
from poc.target import extractor


@pytest.fixture
def investigation(monkeypatch):
    case = load_case("N01")
    incorrect = case.expected_output.model_copy(update={"supplier": "Other supplier"})
    baseline = evaluate_case(
        case,
        CaseEvaluation(
            case_id=case.case_id,
            category=case.category,
            split=case.split,
            expected_output=case.expected_output,
            actual_output=incorrect,
            status="failed"
        )
    )
    model = Mock()
    monkeypatch.setattr(extractor, "get_model", lambda: model)
    return case, baseline, model


def proposal(case_ids):
    return InvestigationProposal.model_validate({
        "hypothesis": "Clarifying which label identifies the supplier may improve extraction.",
        "diagnostic_case_ids": case_ids,
        "candidate": {
            "system_prompt": "Extract the purchase order fields using the labeled supplier.",
            "context_isolation": True,
            "context_aware_tool_restriction": True
        }
    })


def response(order):
    return SimpleNamespace(text=order.model_dump_json(), usage_metadata={})


def test_candidate_is_tested_in_isolated_workspace(tmp_path, investigation):
    case, baseline, model = investigation
    model.invoke.return_value = response(case.expected_output)
    advisor = Mock(return_value=proposal([case.case_id]))

    final = run_feedback_loop(
        [case], [baseline], RunnableLambda(advisor), max_rounds=2,
        diagnostic_budget=2, workspace_dir=tmp_path
    )

    assert final["status"] == "candidate_tested"
    assert final["remaining_diagnostics"] == 1
    attempt, = final["attempts"]
    assert attempt["evaluations"][0].status == "passed"
    assert (attempt["workspace"] / "reports" / f"{case.case_id}.json").is_file()
    advisor.assert_called_once()
    evidence_message = advisor.call_args.args[0][1].content
    assert case.document not in evidence_message
    instruction, data = model.invoke.call_args.args[0]
    assert instruction.content.startswith(attempt["proposal"].candidate.system_prompt)
    assert json.loads(data.content) == {"document": case.document}


@pytest.mark.parametrize("ids,budget", [(["I05"], 1), (["N01", "N01"], 2)])
def test_invalid_selection_does_not_run_diagnostics(tmp_path, investigation, ids, budget):
    case, baseline, model = investigation

    final = run_feedback_loop(
        [case], [baseline], RunnableLambda(lambda _: proposal(ids)),
        max_rounds=1, diagnostic_budget=budget, workspace_dir=tmp_path
    )

    assert final["status"] == "invalid_proposal"
    assert final["remaining_diagnostics"] == budget
    model.invoke.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_failed_candidate_returns_evidence_to_advisor(tmp_path, investigation):
    case, baseline, model = investigation
    model.invoke.side_effect = [
        response(baseline.actual_output), response(case.expected_output)
    ]
    original = proposal([case.case_id])
    revised_candidate = original.candidate.model_copy(
        update={"system_prompt": "Extract all fields and use the explicitly labeled supplier."}
    )
    revised = original.model_copy(update={"candidate": revised_candidate})
    advisor = Mock(side_effect=[original, revised])

    final = run_feedback_loop(
        [case], [baseline], RunnableLambda(advisor), max_rounds=2,
        diagnostic_budget=2, workspace_dir=tmp_path
    )

    assert final["status"] == "candidate_tested"
    first, second = final["attempts"]
    assert first["workspace"] != second["workspace"]
    assert first["evaluations"][0].status == "failed"
    assert second["evaluations"][0].status == "passed"
    assert advisor.call_count == 2
    second_input = json.loads(advisor.call_args.args[0][1].content)
    previous_attempt = second_input["attempts"][0]
    assert previous_attempt["proposal"] == original.model_dump(mode="json")
    assert previous_attempt["evaluations"][0]["status"] == "failed"
    assert "expected_output" not in previous_attempt["evaluations"][0]


def test_repeat_configuration_and_cases_stops_even_in_different_order(
    tmp_path, investigation
):
    first, first_baseline, model = investigation
    second = load_case("N02")
    second_baseline = evaluate_case(second, replace(
        first_baseline,
        case_id=second.case_id,
        expected_output=second.expected_output,
        actual_output=second.expected_output.model_copy(update={"supplier": "Other supplier"})
    ))
    model.invoke.side_effect = [
        response(first_baseline.actual_output), response(second_baseline.actual_output)
    ]
    first_proposal = proposal([first.case_id, second.case_id])
    repeated = first_proposal.model_copy(update={
        "diagnostic_case_ids": [second.case_id, first.case_id],
        "hypothesis": "Restating the same diagnostic."
    })
    advisor = Mock(side_effect=[first_proposal, repeated])

    final = run_feedback_loop(
        [first, second], [first_baseline, second_baseline], RunnableLambda(advisor),
        max_rounds=3, diagnostic_budget=4, workspace_dir=tmp_path
    )

    assert final["status"] == "no_progress"
    assert final["remaining_rounds"] == 1
    assert final["remaining_diagnostics"] == 2
    assert len(final["attempts"]) == 1
    assert len(list(tmp_path.iterdir())) == 1
    assert advisor.call_count == model.invoke.call_count == 2


@pytest.mark.parametrize("rounds,budget", [(1, 3), (3, 1)])
def test_failed_diagnostics_stop_at_budget(tmp_path, investigation, rounds, budget):
    case, baseline, model = investigation
    model.invoke.return_value = response(baseline.actual_output)
    advisor = Mock(return_value=proposal([case.case_id]))

    final = run_feedback_loop(
        [case], [baseline], RunnableLambda(advisor), max_rounds=rounds,
        diagnostic_budget=budget, workspace_dir=tmp_path
    )

    assert final["status"] == "budget_exhausted"
    assert len(final["attempts"]) == 1
    advisor.assert_called_once()
    model.invoke.assert_called_once()


def test_no_failures_stop_before_model_calls(tmp_path, investigation):
    case, baseline, model = investigation
    baseline.status = "passed"
    baseline.failures = []
    baseline.fields = [type(item)(item.field_name, item.expected, item.expected, True)
                       for item in baseline.fields]
    advisor = Mock()

    final = run_feedback_loop(
        [case], [baseline], RunnableLambda(advisor), max_rounds=1,
        diagnostic_budget=1, workspace_dir=tmp_path
    )

    assert final["status"] == "no_evidence"
    advisor.assert_not_called()
    model.invoke.assert_not_called()
