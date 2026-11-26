import json
from collections import Counter
from unittest.mock import Mock

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from poc.evaluation import agent_benchmark
from poc.evaluation.agent_benchmark import (
    BenchmarkScenario,
    load_agent_benchmark,
    run_agent_benchmark,
    score_agent_decision
)
from poc.adaptive.feedback_loop import InvestigationProposal


def _scenario(scenario_id):
    return next(item for item in load_agent_benchmark() if item.inputs.scenario_id == scenario_id)


def _proposal(case_ids):
    return InvestigationProposal.model_validate({
        "hypothesis": "The recorded mismatch warrants targeted diagnosis.",
        "diagnostic_case_ids": case_ids,
        "candidate": {
            "system_prompt": "Extract the labeled purchase order fields.",
            "context_isolation": True,
            "context_aware_tool_restriction": False
        }
    })


def test_benchmark_is_partitioned_and_has_no_shared_scenario_ids():
    scenarios = load_agent_benchmark()

    assert Counter(scenario.split for scenario in scenarios) == {"calibration": 4, "held_out": 4}
    assert len({item.inputs.scenario_id for item in scenarios}) == 8


def test_reference_cannot_repeat_the_prior_selection():
    raw = _scenario("calibration-03").model_dump()
    raw["reference"]["acceptable_case_sets"] = [raw["inputs"]["prior_case_ids"]]

    with pytest.raises(ValidationError, match="reference is inconsistent"):
        BenchmarkScenario.model_validate(raw)


@pytest.mark.parametrize("case_id", ["B06", "C06"])
def test_rubric_accepts_each_alternative(case_id):
    scenario = _scenario("held-02")
    outcome = {
        "status": "budget_exhausted", "model_attempted": True,
        "proposal": _proposal([case_id]).model_dump()
    }

    assert all(score_agent_decision(scenario.inputs, scenario.reference, outcome).values())


@pytest.mark.parametrize("ids", [["B07", "C07"], ["C07"], ["N07"], ["B07", "B07"], ["I06"]])
def test_rubric_rejects_over_budget_repetition_and_irrelevant_cases(ids):
    scenario = _scenario("held-03")
    outcome = {
        "status": "budget_exhausted", "model_attempted": True,
        "proposal": _proposal(ids).model_dump()
    }

    assert not score_agent_decision(scenario.inputs, scenario.reference, outcome)["overall"]


def test_graph_passes_prior_evidence_without_rubric(monkeypatch):
    scenario = _scenario("calibration-03")
    request = Mock(return_value=_proposal(["C01"]))
    monkeypatch.setattr(
        agent_benchmark, "create_advisor", lambda **kwargs: RunnableLambda(request)
    )

    outcome = run_agent_benchmark(scenario.inputs, "basic")

    request.assert_called_once()
    evidence = json.loads(request.call_args.args[0][1].content)
    assert evidence["attempts"][0]["proposal"]["diagnostic_case_ids"] == ["B01"]
    assert evidence["remaining_diagnostics"] == 1
    assert "acceptable_case_sets" not in evidence
    assert score_agent_decision(scenario.inputs, scenario.reference, outcome)["overall"]


def test_no_evidence_stops_without_invoking_advisor(monkeypatch):
    scenario = _scenario("calibration-04")
    request = Mock()
    monkeypatch.setattr(
        agent_benchmark, "create_advisor", lambda **kwargs: RunnableLambda(request)
    )

    outcome = run_agent_benchmark(scenario.inputs, "evidence_driven")

    assert outcome["status"] == "no_evidence"
    request.assert_not_called()
    assert score_agent_decision(scenario.inputs, scenario.reference, outcome)["overall"]
    incorrect = {**outcome, "model_attempted": True}
    assert not score_agent_decision(scenario.inputs, scenario.reference, incorrect)["overall"]
