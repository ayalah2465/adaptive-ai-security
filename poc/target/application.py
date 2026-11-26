from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .action_log import ActionLog
from .extractor import SYSTEM_PROMPT, extract_order_with_metrics
from .metrics import ExtractionMetrics, ExtractionResult
from .models import PurchaseOrder
from .policy import ALLOWED_ACTIONS, ActionContext
from .storage import read_document, save_report


@dataclass(frozen=True)
class ProcessedOrder:
    order: PurchaseOrder
    report_path: Path
    metrics: ExtractionMetrics


def process_order(
    document_id: str = "order_001",
    *,
    documents_dir: Path | None = None,
    reports_dir: Path | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    context_isolation: bool = False,
    context_aware_tool_restriction: bool = False,
    allowed_actions: frozenset[str] = ALLOWED_ACTIONS,
    action_log: ActionLog | None = None,
    on_extracted: Callable[[ExtractionResult], None] | None = None
) -> ProcessedOrder:
    """Process an order, optionally notifying the caller before saving the report."""
    action_context = (
        ActionContext(document_id, allowed_actions)
        if context_aware_tool_restriction
        else None
    )
    document = read_document(
        document_id, action_log, documents_dir=documents_dir, action_context=action_context
    )
    extraction = extract_order_with_metrics(
        document, system_prompt=system_prompt, context_isolation=context_isolation
    )
    if on_extracted is not None:
        on_extracted(extraction)
    report_path = save_report(
        extraction.order, document_id, action_log, reports_dir=reports_dir,
        action_context=action_context
    )
    return ProcessedOrder(extraction.order, report_path, extraction.metrics)
