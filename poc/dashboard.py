import json
import os
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Error as DatabaseError
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Literal

import streamlit as st

from poc.adaptive.configuration import ConfigurationStore
from poc.adaptive.experiment import AdaptiveExperiment, ExperimentResult
from poc.dashboard_translations import TRANSLATIONS
from poc.evaluation import workspace
from poc.evaluation.comparison import compare_evaluation_reports
from poc.evaluation.dataset_loader import DatasetEdition, load_cases
from poc.evaluation.experiment import evaluation_provenance, run_evaluation


@dataclass(frozen=True)
class LocalEvaluationReport:
    path: Path
    split: str
    metrics: dict
    configuration: dict

    @property
    def dataset_edition(self) -> str:
        return self.configuration.get("dataset_edition", "original")


@st.cache_resource
def _approval_lock() -> Lock:
    return Lock()


def _load_experiment(database: Path) -> None:
    st.session_state.pop("dashboard_result", None)
    st.session_state.pop("dashboard_error", None)
    experiment_id = st.session_state.dashboard_experiment_id.strip()
    try:
        with _approval_lock(), closing(ConfigurationStore(database_path=database)) as store:
            st.session_state.dashboard_result = AdaptiveExperiment(store).restore(experiment_id)
    except (ValueError, OSError, DatabaseError):
        st.session_state.dashboard_error = "load_error"


def _decide(database: Path, decision: Literal["approve", "reject", "revise"]) -> None:
    previous = st.session_state.get("dashboard_result")
    if previous is None or previous.approval is None:
        return

    st.session_state.pop("dashboard_error", None)
    try:
        with _approval_lock(), closing(ConfigurationStore(database_path=database)) as store:
            experiment = AdaptiveExperiment(store)
            restored = experiment.restore(previous.experiment_id)
            if restored.approval is not None:
                restored = experiment.resume(previous.approval["candidate_id"], decision)
            st.session_state.dashboard_result = restored
            investigation = st.session_state.get("dashboard_investigation")
            if (
                investigation is not None
                and investigation[0].experiment_id == restored.experiment_id
                and restored.report_path is not None
            ):
                report = json.loads(restored.report_path.read_text(encoding="utf-8"))
                st.session_state.dashboard_investigation = (restored, report)
    except (ValueError, OSError, DatabaseError):
        st.session_state.dashboard_error = "decision_error"


def _show_defenses(candidate: dict, labels: dict[str, str]) -> None:
    for setting, label in (
        ("context_isolation", "isolation"),
        ("context_aware_tool_restriction", "restriction")
    ):
        status = labels["enabled"] if candidate[setting] else labels["disabled"]
        st.write(f"{labels[label]} — {status}")


def _show_pending(result: ExperimentResult, database: Path, labels: dict[str, str]) -> None:
    request = result.approval
    st.write(labels["candidate_id"], request["candidate_id"])
    st.write(labels["base_version"], request["base_version"])
    task, fields, cases = st.columns(3)
    regression = request["regression"]
    task.metric(labels["task_success"], f'{regression["task_success_rate"]:.1f}%')
    fields.metric(labels["field_accuracy"], f'{regression["field_accuracy"]:.1f}%')
    cases.metric(labels["cases"], regression["total_cases"])

    candidate = request["candidate"]
    _show_defenses(candidate, labels)
    with st.expander(labels["prompt"]):
        st.code(candidate["system_prompt"], language=None, wrap_lines=True)

    for column, decision in zip(st.columns(3), ("approve", "reject", "revise"), strict=True):
        column.button(
            labels[decision], key=f"dashboard_{decision}",
            on_click=_decide, args=(database, decision)
        )


def _show_approval(database: Path, labels: dict[str, str]) -> None:
    with st.form("dashboard_load"):
        st.text_input(labels["experiment_id"], key="dashboard_experiment_id")
        st.form_submit_button(labels["load_experiment"], on_click=_load_experiment, args=(database,))

    error = st.session_state.get("dashboard_error")
    if error:
        st.error(labels[error])
    result = st.session_state.get("dashboard_result")
    if result is None:
        return
    st.subheader(labels.get(result.status, result.status))
    if result.approval is not None:
        _show_pending(result, database, labels)
    elif result.report_path is not None:
        try:
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st.error(labels["report_error"])
            return
        st.metric(labels["active_version"], report["active_version"])
        with st.expander(labels["report"]):
            st.json(report)


def _process_order(database: Path) -> None:
    st.session_state.pop("dashboard_order", None)
    st.session_state.pop("dashboard_processing_error", None)
    document_id = st.session_state.dashboard_document_id.strip()
    try:
        with closing(ConfigurationStore(database_path=database)) as store, TemporaryDirectory(
            prefix="adaptive-playground-"
        ) as directory:
            order = store.process_with_active_configuration(
                document_id, reports_dir=Path(directory)
            )
        st.session_state.dashboard_order = order
    except (OSError, ValueError, RuntimeError, DatabaseError):
        st.session_state.dashboard_processing_error = "processing_error"


def _show_playground(database: Path, labels: dict[str, str]) -> None:
    with st.form("dashboard_process"):
        st.text_input(
            labels["document_id"], value="order_001", key="dashboard_document_id"
        )
        st.form_submit_button(labels["process_order"], on_click=_process_order, args=(database,))

    error = st.session_state.get("dashboard_processing_error")
    if error:
        st.error(labels[error])
    configured_order = st.session_state.get("dashboard_order")
    if configured_order is None:
        return

    st.subheader(labels["order"])
    result = configured_order.result
    version, timing = st.columns(2)
    version.metric(labels["version_used"], configured_order.version)
    timing.metric(labels["invocation_seconds"], f"{result.metrics.invocation_seconds:.2f}")
    st.table([
        {labels["field"]: labels[field], labels["value"]: value}
        for field, value in result.order.model_dump(mode="json").items()
    ])
    for column, name in zip(st.columns(2), ("input_tokens", "output_tokens"), strict=True):
        value = getattr(result.metrics, name)
        column.metric(labels[name], labels["unavailable"] if value is None else value)


def _rollback(database: Path) -> None:
    st.session_state.pop("dashboard_restored_version", None)
    st.session_state.pop("dashboard_rollback_error", None)
    if not st.session_state.dashboard_confirm_rollback:
        st.session_state.dashboard_rollback_error = "rollback_confirmation_required"
        return
    st.session_state.dashboard_confirm_rollback = False
    version = st.session_state.dashboard_restore_version
    try:
        with _approval_lock(), closing(ConfigurationStore(database_path=database)) as store:
            if version >= store.active.version:
                raise ValueError("Select an earlier version")
            st.session_state.dashboard_restored_version = store.rollback(version).version
    except (OSError, ValueError, DatabaseError):
        st.session_state.dashboard_rollback_error = "rollback_error"


def _show_history(database: Path, labels: dict[str, str]) -> None:
    try:
        with closing(ConfigurationStore(database_path=database)) as store:
            history = store.history
    except (ValueError, OSError, DatabaseError):
        st.error(labels["database_error"])
        return

    error = st.session_state.get("dashboard_rollback_error")
    if error:
        st.error(labels[error])
    restored = st.session_state.get("dashboard_restored_version")
    if restored is not None:
        st.success(labels["rollback_done"].format(version=restored))
    st.metric(labels["active_version"], history[-1].version)
    st.table([
        {
            labels["version"]: version.version,
            labels["candidate_id"]: version.candidate_id or "—",
            labels["restored_from"]: version.restored_from if version.restored_from is not None else "—"
        }
        for version in reversed(history)
    ])
    earlier_versions = [version.version for version in reversed(history[:-1])]
    if not earlier_versions:
        st.info(labels["no_history"])
        return
    with st.form("dashboard_rollback"):
        st.selectbox(labels["restore_version"], earlier_versions, key="dashboard_restore_version")
        st.checkbox(labels["confirm_rollback"], key="dashboard_confirm_rollback")
        st.form_submit_button(labels["rollback"], on_click=_rollback, args=(database,))


def _evaluation_reports() -> dict[str, LocalEvaluationReport]:
    reports = {}
    for path in sorted(workspace.EXPERIMENTS_DIR.glob("*/evaluation_report.json"), reverse=True):
        if path.is_symlink() or path.parent.is_symlink():
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            experiment_id = report["experiment_id"]
            split = report["cases"][0]["split"]
            metrics = report["metrics"]
            configuration = report.get("configuration", {})
        except (ValueError, OSError, AttributeError, KeyError, IndexError, TypeError):
            continue
        if (
            experiment_id == path.parent.name
            and split in ("development", "held_out")
            and isinstance(metrics, dict)
            and isinstance(configuration, dict)
        ):
            reports[experiment_id] = LocalEvaluationReport(path, split, metrics, configuration)
    return reports


def _generate_baseline(database: Path, max_cases: int, dataset_edition: DatasetEdition) -> None:
    st.session_state.pop("dashboard_baseline_error", None)
    try:
        with closing(ConfigurationStore(database_path=database)) as store:
            result = run_evaluation(
                "development", max_cases=max_cases, workspace_dir=workspace.EXPERIMENTS_DIR,
                dataset_edition=dataset_edition,
                **store.active.candidate.model_dump()
            )
        st.session_state.dashboard_report_id = result.experiment_id
    except (OSError, ValueError, RuntimeError, DatabaseError):
        st.session_state.dashboard_baseline_error = "baseline_error"


def _show_experiments(
    database: Path, labels: dict[str, str], dataset_edition: DatasetEdition
) -> None:
    try:
        development_count = len(load_cases("development", dataset_edition=dataset_edition))
    except (OSError, ValueError):
        st.error(labels["baseline_error"])
    else:
        st.caption(labels["baseline_budget"].format(cases=development_count))
        st.button(
            labels["generate_baseline"], on_click=_generate_baseline,
            args=(database, development_count, dataset_edition)
        )
    error = st.session_state.get("dashboard_baseline_error")
    if error:
        st.error(labels[error])
    reports = {
        report_id: report for report_id, report in _evaluation_reports().items()
        if report.dataset_edition == dataset_edition
    }
    if not reports:
        st.info(labels["no_reports"])
        return

    def label(report_id: str) -> str:
        return f"{report_id} ({reports[report_id].split})"

    report_id = st.selectbox(
        labels["report_id"], list(reports), format_func=label, key="dashboard_report_id"
    )
    st.json(reports[report_id].metrics)
    if len(reports) < 2:
        return

    with st.form("dashboard_compare"):
        before = st.selectbox(
            labels["baseline"], list(reports), format_func=label, key="dashboard_compare_before"
        )
        after = st.selectbox(
            labels["candidate_report"], list(reports), index=1, format_func=label,
            key="dashboard_compare_after"
        )
        if st.form_submit_button(labels["compare"]):
            try:
                comparison = compare_evaluation_reports(reports[before].path, reports[after].path)
            except (ValueError, OSError):
                st.session_state.pop("dashboard_comparison", None)
                st.session_state.dashboard_comparison_error = True
            else:
                st.session_state.pop("dashboard_comparison_error", None)
                st.session_state.dashboard_comparison = comparison

    if st.session_state.get("dashboard_comparison_error"):
        st.error(labels["comparison_error"])
    comparison = st.session_state.get("dashboard_comparison")
    if comparison is not None:
        st.caption(f"{comparison.baseline_experiment_id} → {comparison.candidate_experiment_id}")
        st.metric(labels["regressions"], comparison.regressed_cases)
        st.json({
            "configuration_changes": comparison.configuration_changes,
            "metrics": {
                name: {
                    "baseline": metric.baseline, "candidate": metric.candidate,
                    "difference": metric.difference
                }
                for name, metric in comparison.metrics.items()
            }
        })


def _start_investigation(
    database: Path, report_path: Path, max_regression_cases: int, dataset_edition: DatasetEdition
) -> None:
    st.session_state.pop("dashboard_investigation_error", None)
    try:
        with closing(ConfigurationStore(database_path=database)) as store:
            result = AdaptiveExperiment(store, workspace_dir=workspace.EXPERIMENTS_DIR).start_from_report(
                report_path, max_rounds=1, diagnostic_budget=2, dataset_edition=dataset_edition,
                max_regression_cases=max_regression_cases
            )
            if result.approval is not None:
                _, _, report = store.load_experiment(result.experiment_id)
            else:
                report = json.loads(result.report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError, DatabaseError):
        st.session_state.dashboard_investigation_error = True
    else:
        st.session_state.dashboard_investigation = (result, report)


def _show_investigation(
    database: Path, labels: dict[str, str], dataset_edition: DatasetEdition
) -> None:
    existing = st.session_state.get("dashboard_investigation")
    if existing is not None and existing[1].get("dataset_edition", "original") == dataset_edition:
        result, report = existing
        st.subheader(labels.get(result.status, result.status))
        st.metric(labels["proposal_rounds"], report["advisor_rounds_used"])
        if result.approval is not None:
            st.info(labels["investigation_waiting"])
            st.code(result.experiment_id, language=None)
        for attempt in report["diagnostics"]:
            st.write(labels["hypothesis"])
            st.text(attempt["proposal"]["hypothesis"])
            with st.expander(labels["diagnostics"]):
                st.json({
                    "cases": attempt["cases"], "field_accuracy": attempt["field_accuracy"],
                    "task_success_rate": attempt["task_success_rate"]
                })
    if st.session_state.get("dashboard_investigation_error"):
        st.error(labels["investigation_error"])

    try:
        with closing(ConfigurationStore(database_path=database)) as store:
            active = store.active.candidate
        cases = load_cases("development", dataset_edition=dataset_edition)
        expected = evaluation_provenance("development", cases, **active.model_dump())
    except (OSError, ValueError, DatabaseError):
        st.error(labels["investigation_error"])
        return
    eligible = {
        report_id: report.path
        for report_id, report in _evaluation_reports().items()
        if report.split == "development"
        and report.dataset_edition == dataset_edition
        and all(report.configuration.get(key) == value for key, value in expected.items())
    }
    if not eligible:
        st.info(labels["no_baseline"])
        return
    baseline = st.selectbox(labels["baseline"], list(eligible), key="dashboard_investigation_baseline")
    st.caption(labels["investigation_budget"].format(cases=len(cases)))
    st.button(
        labels["investigate"], on_click=_start_investigation,
        args=(database, eligible[baseline], len(cases), dataset_edition)
    )


def _remember_dataset_edition() -> None:
    st.session_state.dashboard_selected_edition = st.session_state.dashboard_dataset_edition
    for key in (
        "dashboard_report_id", "dashboard_compare_before", "dashboard_compare_after",
        "dashboard_comparison", "dashboard_comparison_error",
        "dashboard_investigation_baseline", "dashboard_investigation_error"
    ):
        st.session_state.pop(key, None)


def main() -> None:
    st.set_page_config(page_title="Adaptive defense", layout="centered")
    language = st.sidebar.selectbox("Language / שפה", list(TRANSLATIONS), key="dashboard_language")
    labels = TRANSLATIONS[language]
    if language == "עברית":
        st.html("""
            <style>
            [data-testid="stMain"], [data-testid="stSidebarContent"] {
                direction: rtl;
                text-align: right;
            }
            [data-testid="stCode"] { direction: ltr; text-align: left; }
            </style>
        """)
    screen = st.sidebar.radio(
        labels["navigation"],
        ["overview", "playground", "investigation", "experiments", "approval", "history"],
        format_func=labels.__getitem__,
        key="dashboard_screen"
    )
    st.title(labels["title"])
    st.header(labels[screen])

    database_setting = os.getenv("ADAPTIVE_AI_DATABASE_PATH")
    if not database_setting:
        st.info(labels["configure_database"])
        return
    database = Path(database_setting).expanduser()

    if screen in ("investigation", "experiments"):
        editions = ["original", "expanded"]
        selected = st.session_state.get("dashboard_selected_edition", "original")
        dataset_edition = st.sidebar.selectbox(
            labels["dataset_edition"], editions, index=editions.index(selected),
            format_func=labels.__getitem__, key="dashboard_dataset_edition",
            on_change=_remember_dataset_edition
        )
        if screen == "investigation":
            _show_investigation(database, labels, dataset_edition)
        else:
            _show_experiments(database, labels, dataset_edition)
        return

    screens = {
        "playground": _show_playground, "approval": _show_approval, "history": _show_history
    }
    if screen in screens:
        screens[screen](database, labels)
        return
    try:
        with closing(ConfigurationStore(database_path=database)) as store:
            configuration = store.active
    except (ValueError, OSError, DatabaseError):
        st.error(labels["database_error"])
        return
    st.metric(labels["active_version"], configuration.version)
    _show_defenses(configuration.candidate.model_dump(), labels)


if __name__ == "__main__":
    main()
