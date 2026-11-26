from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, ValidationError

from poc.evaluation.aggregator import aggregate_metrics
from poc.evaluation.dataset_loader import DatasetEdition, load_cases
from poc.evaluation.evaluator import evaluate_case
from poc.evaluation.models import CaseEvaluation
from poc.evaluation.runner import run_case
from poc.evaluation.workspace import create_workspace

from .feedback_loop import DefenseCandidate, FeedbackState, _candidate_improves


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    candidate_id: str
    decision: Literal["approve", "reject", "revise"]


class ApprovalState(TypedDict, total=False):
    status: Literal[
        "regression_failed", "awaiting_decision", "approved", "rejected",
        "revision_requested", "invalid_decision"
    ]
    candidate_id: str
    base_version: int
    candidate: dict
    regression: dict


def build_approval_workflow(
    investigation: FeedbackState,
    baseline_results: Sequence[CaseEvaluation],
    *,
    workspace_dir: Path | None = None,
    base_version: int = 0,
    checkpointer: BaseCheckpointSaver | None = None,
    dataset_edition: DatasetEdition = "original",
    max_regression_cases: int | None = None
):
    """Build an in-memory regression and approval graph for one tested candidate."""
    proposal = investigation["proposal"]
    attempts = investigation["attempts"]
    if (
        investigation["status"] != "candidate_tested"
        or proposal is None
        or not attempts
        or attempts[-1]["proposal"] != proposal
    ):
        raise ValueError("Investigation must end with a tested candidate")

    cases = {
        case.case_id: case
        for case in load_cases("development", dataset_edition=dataset_edition)
    }
    if max_regression_cases is not None and len(cases) > max_regression_cases:
        raise ValueError("The development split exceeds the approved regression budget")
    baseline = {result.case_id: result for result in baseline_results}
    if (
        not cases
        or len(baseline) != len(baseline_results)
        or baseline.keys() != cases.keys()
        or any(
            result.split != "development"
            or result.status == "pending"
            or result.category != cases[case_id].category
            or result.expected_output != cases[case_id].expected_output
            for case_id, result in baseline.items()
        )
    ):
        raise ValueError("Baseline must cover the complete development split")

    diagnostics = attempts[-1]["evaluations"]
    if (
        not diagnostics
        or [result.case_id for result in diagnostics] != proposal.diagnostic_case_ids
        or any(
            result.case_id not in baseline
            or result.split != "development"
            or result.status == "pending"
            or result.category != baseline[result.case_id].category
            or result.expected_output != baseline[result.case_id].expected_output
            for result in diagnostics
        )
        or not _candidate_improves(diagnostics, baseline)
    ):
        raise ValueError("Investigation diagnostics must support the tested candidate")

    candidate = DefenseCandidate.model_validate(
        proposal.candidate.model_dump()
    ).model_dump()

    def regress(_: ApprovalState) -> dict:
        workspace = create_workspace(workspace_dir)
        results = [
            evaluate_case(case, run_case(case, workspace, **candidate))
            for case in cases.values()
        ]
        summary = aggregate_metrics(results)
        return {
            "status": "awaiting_decision" if _candidate_improves(results, baseline)
            else "regression_failed",
            "candidate_id": uuid4().hex,
            "base_version": base_version,
            "candidate": candidate,
            "regression": {
                "workspace": str(workspace.root_dir),
                "dataset_edition": dataset_edition,
                "total_cases": summary.total_cases,
                "field_accuracy": summary.field_accuracy,
                "task_success_rate": summary.task_success_rate,
                "attack_success_rate": summary.attack_success.rate,
                "false_positive_rate": summary.false_positives.rate,
                "recorded_model_attempts": summary.recorded_model_attempts
            }
        }

    return _compile_approval_workflow(
        regress, checkpointer if checkpointer is not None else InMemorySaver()
    )


def restore_approval_workflow(checkpointer: BaseCheckpointSaver):
    """Restore an existing thread without allowing regression to run again."""
    def regress(_: ApprovalState) -> dict:
        raise ValueError("Restored approval cannot start a new regression")

    return _compile_approval_workflow(regress, checkpointer)


def _compile_approval_workflow(
    regress: Callable[[ApprovalState], dict], checkpointer: BaseCheckpointSaver
):
    def decide(state: ApprovalState) -> dict:
        response = interrupt({
            "candidate_id": state["candidate_id"],
            "base_version": state["base_version"],
            "candidate": state["candidate"],
            "regression": state["regression"]
        })
        try:
            decision = ApprovalDecision.model_validate(response)
        except ValidationError:
            return {"status": "invalid_decision"}
        if decision.candidate_id != state["candidate_id"]:
            return {"status": "invalid_decision"}
        return {"status": {
            "approve": "approved", "reject": "rejected", "revise": "revision_requested"
        }[decision.decision]}

    graph = StateGraph(ApprovalState)
    graph.add_node("regress", regress)
    graph.add_node("decide", decide)
    graph.add_edge(START, "regress")
    graph.add_conditional_edges(
        "regress", lambda state: state["status"] == "awaiting_decision",
        {True: "decide", False: END}
    )
    graph.add_edge("decide", END)
    return graph.compile(checkpointer=checkpointer)
