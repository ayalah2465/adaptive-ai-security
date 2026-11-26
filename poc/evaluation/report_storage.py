import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from poc.target.models import PurchaseOrder

from .aggregator import aggregate_metrics
from .models import CaseEvaluation
from .workspace import ExperimentWorkspace


def _encode_purchase_order(value: object) -> dict:
    if isinstance(value, PurchaseOrder):
        return value.model_dump(mode="json")
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_experiment_report(
    workspace: ExperimentWorkspace,
    evaluations: Iterable[CaseEvaluation],
    *,
    configuration: Mapping[str, str | int | bool] | None = None
) -> Path:
    """Publish a complete experiment report without replacing an existing file."""
    cases = list(evaluations)
    summary = aggregate_metrics(cases)
    metrics = asdict(summary)
    for name in ("attack_success", "false_positives"):
        metrics[name]["rate"] = getattr(summary, name).rate
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        usage = getattr(summary, name)
        metrics[name].update(coverage=usage.coverage, total=usage.total)

    payload = {
        "experiment_id": workspace.experiment_id,
        "metrics": metrics,
        "cases": [asdict(case) for case in cases]
    }
    if configuration is not None:
        payload["configuration"] = dict(configuration)
    destination = workspace.root_dir / "evaluation_report.json"

    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=workspace.root_dir,
            prefix=".evaluation_report-",
            suffix=".tmp",
            delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                payload, temporary, default=_encode_purchase_order,
                ensure_ascii=False, allow_nan=False, indent=2
            )
            temporary.write("\n")

        # A hard link publishes the complete file and fails if the name exists.
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink()

    return destination
