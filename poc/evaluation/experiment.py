import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from poc.datasets.schema import DatasetSplit, TestCase
from poc.target.extractor import MODEL_NAME, SYSTEM_PROMPT

from .aggregator import MetricsSummary, aggregate_metrics
from .dataset_loader import DatasetEdition, load_cases
from .evaluator import evaluate_case
from .report_storage import save_experiment_report
from .runner import run_case
from .workspace import create_workspace


SYSTEMIC_MODEL_ERRORS = frozenset({
    "AuthenticationError", "PermissionDeniedError", "NotFoundError",
    "BadRequestError", "APIConnectionError", "RateLimitError",
    "OpenAIConnectionError", "OpenAIAuthenticationError", "OpenAIPermissionDeniedError",
    "OpenAINotFoundError", "OpenAIBadRequestError", "OpenAIRateLimitError"
})


class EvaluationUnavailableError(RuntimeError):
    def __init__(self, reason: str | None) -> None:
        self.reason = reason
        super().__init__(f"Evaluation stopped after an operational error: {reason or 'unknown'}")


@dataclass(frozen=True)
class EvaluationExperiment:
    experiment_id: str
    report_path: Path
    summary: MetricsSummary


def evaluation_provenance(
    split: DatasetSplit,
    cases: Sequence[TestCase],
    *,
    system_prompt: str,
    context_isolation: bool,
    context_aware_tool_restriction: bool
) -> dict[str, str | bool]:
    serialized_cases = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return {
        "split": split,
        "model_name": MODEL_NAME,
        "context_isolation": context_isolation,
        "context_aware_tool_restriction": context_aware_tool_restriction,
        "system_prompt_sha256": sha256(system_prompt.encode("utf-8")).hexdigest(),
        "cases_sha256": sha256(serialized_cases.encode("utf-8")).hexdigest()
    }


def run_evaluation(
    split: DatasetSplit,
    *,
    max_cases: int,
    dataset_edition: DatasetEdition = "original",
    system_prompt: str = SYSTEM_PROMPT,
    context_isolation: bool = False,
    context_aware_tool_restriction: bool = False,
    workspace_dir: Path | None = None
) -> EvaluationExperiment:
    """Evaluate a complete split once and publish its report after completion."""
    cases = load_cases(split, dataset_edition=dataset_edition)
    if max_cases < len(cases):
        raise ValueError(f"Case budget must cover all {len(cases)} cases in {split}")
    if not system_prompt.strip():
        raise ValueError("System prompt must not be empty")

    configuration = evaluation_provenance(
        split, cases, system_prompt=system_prompt, context_isolation=context_isolation,
        context_aware_tool_restriction=context_aware_tool_restriction
    )

    workspace = create_workspace(workspace_dir)
    results = []
    for case in cases:
        execution = run_case(
            case, workspace, system_prompt=system_prompt,
            context_isolation=context_isolation,
            context_aware_tool_restriction=context_aware_tool_restriction
        )
        if (
            "model_error" in execution.failures
            and execution.error_message in SYSTEMIC_MODEL_ERRORS
        ):
            raise EvaluationUnavailableError(execution.error_message)
        results.append(evaluate_case(case, execution))

    if all(result.status == "error" for result in results):
        raise EvaluationUnavailableError(results[-1].error_message)

    report_path = save_experiment_report(
        workspace, results, configuration={
            **configuration, "max_cases": max_cases, "dataset_edition": dataset_edition
        }
    )
    return EvaluationExperiment(workspace.experiment_id, report_path, aggregate_metrics(results))
