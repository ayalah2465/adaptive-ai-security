import json
import os
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from langchain_core.runnables import Runnable

from poc.datasets.schema import TestCase
from poc.evaluation.aggregator import aggregate_metrics
from poc.evaluation.dataset_loader import DatasetEdition, load_cases
from poc.evaluation.models import CaseEvaluation
from poc.evaluation.workspace import create_workspace

from .advisor import create_advisor
from .baseline import load_development_baseline
from .configuration import ConfigurationStore
from .feedback_loop import InvestigationProposal, run_feedback_loop


@dataclass(frozen=True)
class ExperimentResult:
    status: str
    approval: dict | None = None
    report_path: Path | None = None
    experiment_id: str | None = None


class AdaptiveExperiment:
    """Coordinate a single development experiment and its human decision."""

    def __init__(
        self, configurations: ConfigurationStore, *, workspace_dir: Path | None = None
    ) -> None:
        self._configurations = configurations
        self._workspace_dir = workspace_dir
        self._report_path: Path | None = None
        self._report: dict | None = None
        self._thread_id: str | None = None
        self._recoverable = False

    def start(
        self,
        baseline_results: Sequence[CaseEvaluation],
        advisor: Runnable[list, InvestigationProposal] | None = None,
        *,
        base_version: int,
        max_rounds: int,
        diagnostic_budget: int,
        dataset_edition: DatasetEdition = "original",
        max_regression_cases: int | None = None
    ) -> ExperimentResult:
        return self._start(
            load_cases("development", dataset_edition=dataset_edition), baseline_results, advisor,
            base_version=base_version, max_rounds=max_rounds,
            diagnostic_budget=diagnostic_budget, dataset_edition=dataset_edition,
            max_regression_cases=max_regression_cases
        )

    def start_from_report(
        self,
        baseline_path: Path,
        advisor: Runnable[list, InvestigationProposal] | None = None,
        *,
        max_rounds: int,
        diagnostic_budget: int,
        dataset_edition: DatasetEdition = "original",
        max_regression_cases: int | None = None
    ) -> ExperimentResult:
        active = self._configurations.active
        cases = load_cases("development", dataset_edition=dataset_edition)
        report_id, results = load_development_baseline(
            baseline_path, cases, active.candidate, dataset_edition=dataset_edition
        )
        return self._start(
            cases, results, advisor, base_version=active.version,
            max_rounds=max_rounds, diagnostic_budget=diagnostic_budget,
            baseline_report_id=report_id, dataset_edition=dataset_edition,
            max_regression_cases=max_regression_cases
        )

    def _start(
        self,
        cases: Sequence[TestCase],
        baseline_results: Sequence[CaseEvaluation],
        advisor: Runnable[list, InvestigationProposal] | None,
        *,
        base_version: int,
        max_rounds: int,
        diagnostic_budget: int,
        dataset_edition: DatasetEdition,
        max_regression_cases: int | None,
        baseline_report_id: str | None = None
    ) -> ExperimentResult:
        if self._report is not None:
            raise ValueError("Experiment has already started")

        if base_version != self._configurations.active.version:
            raise ValueError("Baseline version is no longer active")
        if max_regression_cases is not None and len(cases) > max_regression_cases:
            raise ValueError("The development split exceeds the approved regression budget")
        if advisor is None:
            advisor = create_advisor()
        state = run_feedback_loop(
            cases, baseline_results, advisor,
            max_rounds=max_rounds, diagnostic_budget=diagnostic_budget,
            workspace_dir=self._workspace_dir
        )
        workspace = create_workspace(self._workspace_dir)
        self._report_path = workspace.root_dir / "adaptive_experiment.json"

        diagnostics = []
        for attempt in state["attempts"]:
            evaluations = attempt["evaluations"]
            summary = aggregate_metrics(evaluations)
            diagnostics.append({
                "proposal": attempt["proposal"].model_dump(mode="json"),
                "workspace": str(attempt["workspace"]),
                "task_success_rate": summary.task_success_rate,
                "field_accuracy": summary.field_accuracy,
                "attack_success_rate": summary.attack_success.rate,
                "false_positive_rate": summary.false_positives.rate,
                "cases": [
                    {"case_id": case.case_id, "status": case.status, "failures": case.failures}
                    for case in evaluations
                ]
            })
        self._report = {
            "experiment_id": workspace.experiment_id,
            "split": "development",
            "dataset_edition": dataset_edition,
            "base_version": base_version,
            "investigation_status": state["status"],
            "advisor_rounds_used": max_rounds - state["remaining_rounds"],
            "diagnostic_cases_used": diagnostic_budget - state["remaining_diagnostics"],
            "diagnostics": diagnostics,
            "regression": None,
            "candidate_id": None,
            "approval_decision": None
        }
        if baseline_report_id is not None:
            self._report["baseline_report_id"] = baseline_report_id
        if state["status"] != "candidate_tested":
            return self._finish(state["status"])

        try:
            thread_id, approval = self._configurations.start_approval(
                state, baseline_results, base_version=base_version,
                workspace_dir=self._workspace_dir, dataset_edition=dataset_edition,
                max_regression_cases=max_regression_cases
            )
        except ValueError:
            if self._configurations.active.version != base_version:
                return self._finish("stale_configuration")
            raise

        self._report["regression"] = approval["regression"]
        self._report["candidate_id"] = approval["candidate_id"]
        if approval["status"] != "awaiting_decision":
            return self._finish(approval["status"])

        self._thread_id = thread_id
        self._recoverable = self._configurations.save_experiment(
            thread_id, self._report_path, self._report
        )
        return ExperimentResult(
            "awaiting_decision", approval=deepcopy(approval["__interrupt__"][0].value),
            experiment_id=workspace.experiment_id
        )

    def restore(self, experiment_id: str) -> ExperimentResult:
        """Restore a pending experiment or finish publishing its terminal report."""
        if self._report is not None:
            raise ValueError("Experiment has already started")
        thread_id, self._report_path, self._report = self._configurations.load_experiment(
            experiment_id
        )
        self._recoverable = True
        if "status" in self._report:
            return self._publish_report()

        approval = self._configurations.approval_state(thread_id)
        if approval.get("candidate_id") != self._report["candidate_id"]:
            raise ValueError("Restored approval does not match the experiment")
        if approval["status"] == "awaiting_decision":
            self._thread_id = thread_id
            return ExperimentResult(
                "awaiting_decision", approval=self._configurations.pending_approval(thread_id),
                experiment_id=experiment_id
            )
        return self._complete_approval(thread_id, approval)

    def resume(
        self, candidate_id: str, decision: Literal["approve", "reject", "revise"]
    ) -> ExperimentResult:
        if self._thread_id is None or self._report is None:
            raise ValueError("Experiment is not awaiting a decision")

        thread_id = self._thread_id
        approval = self._configurations.record_decision(thread_id, candidate_id, decision)
        self._thread_id = None
        return self._complete_approval(thread_id, approval)

    def _complete_approval(self, thread_id: str, approval: dict) -> ExperimentResult:
        assert self._report is not None
        status = approval["status"]
        self._report["approval_decision"] = status
        if status == "approved":
            activated_version = self._configurations.activated_version(thread_id)
            if activated_version is None:
                try:
                    activated_version = self._configurations.activate(
                        thread_id, approval["candidate_id"]
                    ).version
                except ValueError:
                    activated_version = self._configurations.activated_version(thread_id)
                    if activated_version is None:
                        if self._configurations.active.version != approval["base_version"]:
                            return self._finish("stale_configuration")
                        raise
            if self._recoverable:
                self._report["activated_version"] = activated_version
            status = "activated"
        return self._finish(status)

    def _finish(self, status: str) -> ExperimentResult:
        assert self._report is not None
        assert self._report_path is not None
        self._report["status"] = status
        self._report["active_version"] = self._configurations.active.version
        if self._recoverable:
            self._report = self._configurations.finalize_experiment(self._report)
        return self._publish_report()

    def _publish_report(self) -> ExperimentResult:
        assert self._report is not None
        assert self._report_path is not None
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._report_path.parent,
                suffix=".tmp", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(self._report, temporary, ensure_ascii=False, allow_nan=False, indent=2)
                temporary.write("\n")
            try:
                os.link(temporary_path, self._report_path)
            except FileExistsError:
                if not self._recoverable or self._report_path.read_bytes() != temporary_path.read_bytes():
                    raise
        finally:
            if temporary_path is not None:
                temporary_path.unlink()
        return ExperimentResult(
            self._report["status"], report_path=self._report_path,
            experiment_id=self._report["experiment_id"]
        )
