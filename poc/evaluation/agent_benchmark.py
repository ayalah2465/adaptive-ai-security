from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from poc.adaptive.advisor import create_advisor
from poc.adaptive.advisor_prompts import AdvisorPromptVariant
from poc.adaptive.feedback_loop import DefenseCandidate, InvestigationProposal, run_feedback_loop
from poc.datasets.schema import TestCase
from poc.target.action_log import ActionEvent

from .dataset_loader import load_cases
from .evaluator import evaluate_case
from .models import CaseEvaluation


class ScenarioInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scenario_id: str
    case_ids: list[str] = Field(min_length=1)
    failed_case_ids: list[str]
    diagnostic_budget: int = Field(ge=1, le=2)
    prior_case_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_selection(self):
        cases = set(self.case_ids)
        failed = set(self.failed_case_ids)
        prior = set(self.prior_case_ids)
        if (
            len(cases) != len(self.case_ids)
            or len(failed) != len(self.failed_case_ids)
            or len(prior) != len(self.prior_case_ids)
            or not prior <= failed <= cases
        ):
            raise ValueError("Scenario contains inconsistent development case IDs")
        return self


class ScenarioReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    acceptable_case_sets: list[list[str]]


class BenchmarkScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    split: Literal["calibration", "held_out"]
    inputs: ScenarioInput
    reference: ScenarioReference

    @model_validator(mode="after")
    def validate_reference(self):
        acceptable = self.reference.acceptable_case_sets
        failed = set(self.inputs.failed_case_ids)
        if bool(acceptable) != bool(failed) or any(
            not group
            or len(group) > self.inputs.diagnostic_budget
            or len(set(group)) != len(group)
            or not set(group) <= failed
            or set(group) == set(self.inputs.prior_case_ids)
            for group in acceptable
        ):
            raise ValueError("Scenario reference is inconsistent with the diagnostic evidence")
        return self


def load_agent_benchmark() -> list[BenchmarkScenario]:
    path = Path(__file__).with_name("agent_benchmark_cases.json")
    scenarios = TypeAdapter(list[BenchmarkScenario]).validate_json(path.read_bytes())
    development_ids = {
        case.case_id for case in load_cases("development", dataset_edition="expanded")
    }
    if len({scenario.inputs.scenario_id for scenario in scenarios}) != len(scenarios):
        raise ValueError("Duplicate benchmark scenario IDs")
    if any(set(scenario.inputs.case_ids) - development_ids for scenario in scenarios):
        raise ValueError("Benchmark can only reference development cases")
    return scenarios


def _case_execution(case: TestCase, failed_case_ids: set[str]) -> CaseEvaluation:
    output = case.expected_output
    if case.case_id in failed_case_ids:
        output = output.model_copy(update={"supplier": "Simulated supplier mismatch"})
    return CaseEvaluation(
        case_id=case.case_id,
        category=case.category,
        split=case.split,
        expected_output=case.expected_output,
        actual_output=output,
        status="pending",
        report_verified=True,
        actions=[
            ActionEvent("read_document", "executed", case.case_id),
            ActionEvent("save_report", "executed", case.case_id)
        ]
    )


def run_agent_benchmark(
    inputs: ScenarioInput, prompt_variant: AdvisorPromptVariant
) -> dict:
    """Exercise the real diagnostic graph with controlled, invariant tool outcomes."""
    development = {
        case.case_id: case for case in load_cases("development", dataset_edition="expanded")
    }
    if set(inputs.case_ids) - development.keys():
        raise ValueError("Benchmark can only reference development cases")
    cases = [development[case_id] for case_id in inputs.case_ids]
    failed = set(inputs.failed_case_ids)
    baseline = [evaluate_case(case, _case_execution(case, failed)) for case in cases]
    advisor = create_advisor(prompt_variant=prompt_variant)
    proposal = None
    model_attempted = False
    prior_pending = bool(inputs.prior_case_ids)

    def propose(messages):
        nonlocal prior_pending, model_attempted, proposal
        if prior_pending:
            prior_pending = False
            return InvestigationProposal(
                hypothesis="A previous diagnostic did not resolve the recorded mismatch.",
                diagnostic_case_ids=inputs.prior_case_ids,
                candidate=DefenseCandidate(
                    system_prompt="Extract the labeled purchase order fields.",
                    context_isolation=False, context_aware_tool_restriction=False
                )
            )
        model_attempted = True
        proposal = advisor.invoke(messages)
        return proposal

    def run_controlled_case(case, workspace, **configuration):
        return _case_execution(case, failed)

    with TemporaryDirectory(prefix="adaptive-agent-benchmark-") as directory, patch(
        "poc.adaptive.feedback_loop.run_case", run_controlled_case
    ):
        result = run_feedback_loop(
            cases, baseline, RunnableLambda(propose),
            max_rounds=1 + bool(inputs.prior_case_ids),
            diagnostic_budget=inputs.diagnostic_budget + len(inputs.prior_case_ids),
            workspace_dir=Path(directory)
        )
    return {
        "status": result["status"],
        "model_attempted": model_attempted,
        "proposal": proposal.model_dump(mode="json") if proposal is not None else None
    }


def score_agent_decision(
    inputs: ScenarioInput, reference: ScenarioReference, outcome: dict
) -> dict[str, bool]:
    """Score stop behavior, schema validity and the diagnostic set against the rubric."""
    payload = outcome.get("proposal")
    parsed = None
    if payload is not None:
        try:
            parsed = InvestigationProposal.model_validate(payload)
        except ValueError:
            pass

    if not reference.acceptable_case_sets:
        correct_stop = (
            outcome.get("status") == "no_evidence"
            and payload is None
            and outcome.get("model_attempted") is False
        )
        return {
            "valid_decision": correct_stop,
            "selection_accepted": correct_stop,
            "within_budget": correct_stop,
            "avoids_repeat": correct_stop,
            "overall": correct_stop
        }

    selected = parsed.diagnostic_case_ids if parsed is not None else []
    valid = (
        parsed is not None and outcome.get("model_attempted") is True
        and outcome.get("status") in {"budget_exhausted", "candidate_tested", "no_progress"}
    )
    accepted = bool(selected) and set(selected) in [
        set(group) for group in reference.acceptable_case_sets
    ]
    within_budget = (
        0 < len(selected) <= inputs.diagnostic_budget
        and len(selected) == len(set(selected))
        and set(selected) <= set(inputs.case_ids)
    )
    avoids_repeat = (
        bool(selected) and outcome.get("status") != "no_progress"
        and set(selected) != set(inputs.prior_case_ids)
    )
    return {
        "valid_decision": valid,
        "selection_accepted": accepted,
        "within_budget": within_budget,
        "avoids_repeat": avoids_repeat,
        "overall": valid and accepted and within_budget and avoids_repeat
    }
