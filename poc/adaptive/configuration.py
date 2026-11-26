from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import ValidationError

from poc.evaluation.dataset_loader import DatasetEdition
from poc.evaluation.models import CaseEvaluation
from poc.target.application import ProcessedOrder, process_order
from poc.target.extractor import SYSTEM_PROMPT

from .approval_workflow import (
    ApprovalDecision,
    build_approval_workflow,
    restore_approval_workflow
)
from .feedback_loop import DefenseCandidate, FeedbackState
from .persistence import ConfigurationPersistence, ConfigurationVersion


@dataclass(frozen=True)
class ConfiguredOrder:
    version: int
    result: ProcessedOrder


class ConfigurationStore:
    """Configuration history for trusted application and operator calls."""

    def __init__(self, *, database_path: Path | None = None) -> None:
        baseline = DefenseCandidate(
            system_prompt=SYSTEM_PROMPT,
            context_isolation=False,
            context_aware_tool_restriction=False
        )
        self._history = [ConfigurationVersion(0, baseline)]
        self._workflows: dict[str, CompiledStateGraph] = {}
        self._used_approvals: set[str] = set()
        self._lock = Lock()
        self._persistence = (
            ConfigurationPersistence(database_path, baseline)
            if database_path is not None else None
        )
        self._restored_workflow = (
            restore_approval_workflow(self._persistence.checkpointer)
            if self._persistence is not None else None
        )

    def close(self) -> None:
        with self._lock:
            if self._persistence is not None:
                self._persistence.close()

    def _versions(self) -> Sequence[ConfigurationVersion]:
        if self._persistence is not None:
            return self._persistence.history()
        return self._history

    @property
    def active(self) -> ConfigurationVersion:
        with self._lock:
            return self._copy_version(self._versions()[-1])

    @property
    def history(self) -> tuple[ConfigurationVersion, ...]:
        with self._lock:
            return tuple(self._copy_version(version) for version in self._versions())

    def process_with_active_configuration(
        self,
        document_id: str = "order_001",
        *,
        documents_dir: Path | None = None,
        reports_dir: Path | None = None
    ) -> ConfiguredOrder:
        configuration = self.active
        result = process_order(
            document_id, documents_dir=documents_dir, reports_dir=reports_dir,
            system_prompt=configuration.candidate.system_prompt,
            context_isolation=configuration.candidate.context_isolation,
            context_aware_tool_restriction=configuration.candidate.context_aware_tool_restriction
        )
        return ConfiguredOrder(configuration.version, result)

    @staticmethod
    def _copy_version(version: ConfigurationVersion) -> ConfigurationVersion:
        return ConfigurationVersion(
            version.version, version.candidate.model_copy(deep=True),
            version.candidate_id, version.restored_from
        )

    @staticmethod
    def _thread(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _workflow(self, thread_id: str) -> CompiledStateGraph:
        if self._restored_workflow is not None:
            return self._restored_workflow
        return self._workflows[thread_id]

    def approval_state(self, thread_id: str) -> dict:
        with self._lock:
            return self._workflow(thread_id).get_state(self._thread(thread_id)).values

    def save_experiment(self, thread_id: str, report_path: Path, report: dict) -> bool:
        if self._persistence is None:
            return False
        self._persistence.save_experiment(thread_id, report_path, report)
        return True

    def load_experiment(self, experiment_id: str) -> tuple[str, Path, dict]:
        if self._persistence is None:
            raise ValueError("Experiment recovery requires SQLite storage")
        return self._persistence.load_experiment(experiment_id)

    def finalize_experiment(self, report: dict) -> dict:
        if self._persistence is None:
            return report
        return self._persistence.finalize_experiment(report)

    def activated_version(self, thread_id: str) -> int | None:
        if self._persistence is None:
            return None
        return self._persistence.activated_version(thread_id)

    def pending_approval(self, thread_id: str) -> dict:
        """Retrieve the pending interrupt payload without resuming the graph."""
        with self._lock:
            snapshot = self._workflow(thread_id).get_state(self._thread(thread_id))
            if snapshot.values.get("status") != "awaiting_decision":
                raise ValueError("Workflow is not awaiting a decision")
            for task in snapshot.tasks:
                for request in task.interrupts:
                    return request.value
            raise ValueError("Workflow has no pending approval interrupt")

    def start_approval(
        self,
        investigation: FeedbackState,
        baseline_results: Sequence[CaseEvaluation],
        *,
        base_version: int,
        workspace_dir: Path | None = None,
        dataset_edition: DatasetEdition = "original",
        max_regression_cases: int | None = None
    ) -> tuple[str, dict]:
        with self._lock:
            if base_version != self._versions()[-1].version:
                raise ValueError("Regression base version is no longer active")

        graph = build_approval_workflow(
            investigation, baseline_results, workspace_dir=workspace_dir,
            base_version=base_version,
            checkpointer=self._persistence.checkpointer if self._persistence is not None else None,
            dataset_edition=dataset_edition, max_regression_cases=max_regression_cases
        )
        thread_id = uuid4().hex
        state = graph.invoke({}, config=self._thread(thread_id))
        with self._lock:
            if self._persistence is None:
                self._workflows[thread_id] = graph
        return thread_id, state

    def record_decision(
        self,
        thread_id: str,
        candidate_id: str,
        decision: Literal["approve", "reject", "revise"]
    ) -> dict:
        with self._lock:
            graph = self._workflow(thread_id)
            config = self._thread(thread_id)
            state = graph.get_state(config).values
            if state.get("status") != "awaiting_decision":
                raise ValueError("Workflow is not awaiting a decision")
            if state["candidate_id"] != candidate_id:
                raise ValueError("Candidate does not match the pending approval")
            try:
                validated = ApprovalDecision(candidate_id=candidate_id, decision=decision)
            except ValidationError:
                raise ValueError("Unsupported approval decision") from None
            return graph.invoke(
                Command(resume=validated.model_dump()),
                config=config
            )

    def activate(self, thread_id: str, candidate_id: str) -> ConfigurationVersion:
        with self._lock:
            graph = self._workflow(thread_id)
            snapshot = graph.get_state(self._thread(thread_id))
            state = snapshot.values
            if snapshot.next or state.get("status") != "approved":
                raise ValueError("Workflow has no final approval")
            if state["candidate_id"] != candidate_id:
                raise ValueError("Candidate does not match the approval")
            candidate = DefenseCandidate.model_validate(state["candidate"])
            base_version = state["base_version"]
            version = ConfigurationVersion(base_version + 1, candidate, candidate_id)
            self._append_version(version, base_version, approval_thread=thread_id)
            return self._copy_version(version)

    def rollback(self, version: int) -> ConfigurationVersion:
        """Restore a known version through an explicit trusted operator call."""
        with self._lock:
            history = self._versions()
            if not 0 <= version < len(history):
                raise ValueError("Unknown configuration version")
            target = history[version]
            active_version = history[-1].version
            restored = ConfigurationVersion(
                active_version + 1, target.candidate.model_copy(deep=True),
                target.candidate_id, restored_from=version
            )
            self._append_version(restored, active_version)
            return self._copy_version(restored)

    def _append_version(
        self, version: ConfigurationVersion, expected_version: int,
        *, approval_thread: str | None = None
    ) -> None:
        if self._persistence is not None:
            self._persistence.append(
                version, expected_version=expected_version, approval_thread=approval_thread
            )
            return
        if approval_thread in self._used_approvals:
            raise ValueError("Approval has already been used")
        if expected_version != self._history[-1].version:
            raise ValueError("Approval is stale for the active version")
        self._history.append(version)
        if approval_thread is not None:
            self._used_approvals.add(approval_thread)
