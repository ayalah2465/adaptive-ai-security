
import time

from pydantic import ValidationError

from poc.datasets.schema import TestCase

from poc.target.action_log import ActionLog

from poc.target.application import process_order

from poc.target.extractor import SYSTEM_PROMPT

from poc.target.metrics import ExtractionFailure, ExtractionResult

from poc.target.models import PurchaseOrder

from poc.target.policy import (
    PolicyViolation,
    resolve_safe_path
)

from .models import CaseEvaluation

from .workspace import ExperimentWorkspace


def prepare_document(
    case: TestCase,
    workspace: ExperimentWorkspace
) -> None:

    document_path = resolve_safe_path(
        workspace.documents_dir,
        case.case_id,
        ".txt"
    )

    with document_path.open(
        "x",
        encoding="utf-8"
    ) as file:

        file.write(case.document)


def run_case(
    case: TestCase,
    workspace: ExperimentWorkspace,
    system_prompt: str = SYSTEM_PROMPT,
    *,
    context_isolation: bool = False,
    context_aware_tool_restriction: bool = False
) -> CaseEvaluation:

    evaluation = CaseEvaluation(
        case_id=case.case_id,
        category=case.category,
        split=case.split,
        expected_output=case.expected_output
    )

    action_log = ActionLog()

    def record_extraction(extraction: ExtractionResult) -> None:
        evaluation.actual_output = extraction.order
        evaluation.metrics = extraction.metrics

    start_time = time.perf_counter()

    try:

        prepare_document(case, workspace)

        result = process_order(
            case.case_id,
            documents_dir=workspace.documents_dir,
            reports_dir=workspace.reports_dir,
            system_prompt=system_prompt,
            context_isolation=context_isolation,
            context_aware_tool_restriction=context_aware_tool_restriction,
            allowed_actions=frozenset(case.allowed_actions),
            action_log=action_log,
            on_extracted=record_extraction
        )

        saved_order = PurchaseOrder.model_validate_json(
            result.report_path.read_text(
                encoding="utf-8"
            )
        )

        evaluation.report_verified = (
            saved_order == result.order
        )

        if not evaluation.report_verified:

            evaluation.status = "failed"

            evaluation.failures.append(
                "report_mismatch"
            )

        else:

            # The deterministic evaluator assigns the final task status.
            evaluation.status = "pending"

    except ExtractionFailure as exc:

        evaluation.metrics = exc.metrics

        if exc.failure_type == "invalid_output":

            evaluation.status = "failed"

            evaluation.failures.append(
                "invalid_output"
            )

        else:

            evaluation.status = "error"

            evaluation.failures.append(
                "model_error"
            )

        evaluation.error_message = str(exc)

    except PolicyViolation as exc:

        evaluation.status = "blocked"

        evaluation.failures.append(
            "policy_violation"
        )

        evaluation.error_message = str(exc)

    except ValidationError:

        evaluation.status = "failed"

        evaluation.failures.append(
            "report_mismatch"
        )

        evaluation.error_message = (
            "Saved report is invalid"
        )

    except (OSError, UnicodeError) as exc:

        evaluation.status = "error"

        evaluation.failures.append(
            "storage_error"
        )

        evaluation.error_message = (
            type(exc).__name__
        )

    finally:

        evaluation.actions = list(
            action_log.events
        )

        evaluation.execution_seconds = (
            time.perf_counter() - start_time
        )

    return evaluation
