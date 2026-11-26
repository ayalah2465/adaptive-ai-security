
from pathlib import Path

from .models import PurchaseOrder

from .action_log import ActionLog

from .policy import (
    ActionContext,
    PolicyViolation,
    authorize_action,
    resolve_safe_path
)


DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
)

DOCUMENTS_DIR = DATA_DIR / "documents"

REPORTS_DIR = DATA_DIR / "reports"


def read_document(
    document_id: str = "order_001",
    action_log: ActionLog | None = None,
    *,
    documents_dir: Path | None = None,
    action_context: ActionContext | None = None
) -> str:

    action = "read_document"

    base_dir = (
        documents_dir
        if documents_dir is not None
        else DOCUMENTS_DIR
    )

    if action_log is not None:

        action_log.record(
            action, "requested", document_id
        )

    try:

        authorize_action(action, resource=document_id, context=action_context)

        document_path = resolve_safe_path(
            base_dir,
            document_id,
            ".txt"
        )

        document = document_path.read_text(
            encoding="utf-8"
        )

    except PolicyViolation:

        if action_log is not None:

            action_log.record(
                action, "blocked", document_id
            )

        raise

    except OSError:

        if action_log is not None:

            action_log.record(
                action, "failed", document_id
            )

        raise

    if action_log is not None:

        action_log.record(
            action, "executed", document_id
        )

    return document


def save_report(
    order: PurchaseOrder,
    document_id: str = "order_001",
    action_log: ActionLog | None = None,
    *,
    reports_dir: Path | None = None,
    action_context: ActionContext | None = None
) -> Path:

    action = "save_report"

    base_dir = (
        reports_dir
        if reports_dir is not None
        else REPORTS_DIR
    )

    if action_log is not None:

        action_log.record(
            action, "requested", document_id
        )

    try:

        authorize_action(action, resource=document_id, context=action_context)

        report_path = resolve_safe_path(
            base_dir,
            document_id,
            ".json"
        )

        base_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        report_path.write_text(
            order.model_dump_json(indent=2),
            encoding="utf-8"
        )

    except PolicyViolation:

        if action_log is not None:

            action_log.record(
                action, "blocked", document_id
            )

        raise

    except OSError:

        if action_log is not None:

            action_log.record(
                action, "failed", document_id
            )

        raise

    if action_log is not None:

        action_log.record(
            action, "executed", document_id
        )

    return report_path
