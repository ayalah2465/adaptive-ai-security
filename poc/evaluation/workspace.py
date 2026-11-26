
from dataclasses import dataclass

from datetime import datetime, timezone

from pathlib import Path

from uuid import uuid4


EXPERIMENTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "experiments"
)


@dataclass(frozen=True)
class ExperimentWorkspace:

    experiment_id: str

    root_dir: Path

    documents_dir: Path

    reports_dir: Path


def create_workspace(
    base_dir: Path | None = None
) -> ExperimentWorkspace:

    base = (
        base_dir
        if base_dir is not None
        else EXPERIMENTS_DIR
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    experiment_id = (
        f"{timestamp}_{uuid4().hex[:12]}"
    )

    root_dir = base / experiment_id

    documents_dir = root_dir / "documents"

    reports_dir = root_dir / "reports"

    root_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    documents_dir.mkdir()

    reports_dir.mkdir()

    return ExperimentWorkspace(
        experiment_id=experiment_id,
        root_dir=root_dir,
        documents_dir=documents_dir,
        reports_dir=reports_dir
    )
