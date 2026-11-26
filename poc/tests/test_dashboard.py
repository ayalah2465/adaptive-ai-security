import json
from contextlib import closing
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from poc.adaptive.configuration import ConfigurationStore
from poc.dashboard_translations import TRANSLATIONS
from poc.evaluation import dataset_loader, workspace
from poc.evaluation.experiment import run_evaluation
from poc.tests.test_adaptive_approval_workflow import approval_inputs, model_response
from poc.tests.test_adaptive_persistence import start_experiment


APP = Path(__file__).resolve().parents[1] / "dashboard.py"


def _navigate(app, screen):
    language = app.sidebar.selectbox(key="dashboard_language").value
    key = "dashboard_screen" if language == "English" else "dashboard_screen_he"
    app.sidebar.radio(key=key).set_value(screen).run()


def _click(app, label):
    next(button for button in app.button if button.label == label).click().run()


@pytest.fixture
def saved_experiment(tmp_path, monkeypatch, start_experiment, approval_inputs):
    _, _, _, model = approval_inputs
    database = tmp_path / "dashboard.db"
    with closing(ConfigurationStore(database_path=database)) as store:
        _, pending = start_experiment(store)
        _, report_path, _ = store.load_experiment(pending.experiment_id)
    model.invoke.reset_mock()
    monkeypatch.setenv("ADAPTIVE_AI_DATABASE_PATH", str(database))
    return pending, report_path, database, model


def _load_experiment(app, experiment_id):
    _navigate(app, "approval")
    app.text_input(key="dashboard_experiment_id").set_value(experiment_id)
    language = app.sidebar.selectbox(key="dashboard_language").value
    _click(app, TRANSLATIONS[language]["load_experiment"])


def test_dashboard_restores_approval_switches_language_and_activates(saved_experiment):
    pending, report_path, database, model = saved_experiment
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert app.header[0].value == "Overview"
    assert app.metric[0].value == "0"

    _load_experiment(app, pending.experiment_id)
    assert not app.exception
    assert app.subheader[0].value == "Awaiting a human decision"
    assert app.session_state.dashboard_result == pending

    app.sidebar.selectbox(key="dashboard_language").set_value("עברית").run()
    assert not app.exception
    assert app.header[0].value == "אישור"
    assert app.session_state.dashboard_result == pending
    assert app.button(key="dashboard_approve").label == "אישור"
    assert not report_path.exists()

    app.button(key="dashboard_approve").click().run()
    assert not app.exception
    assert app.subheader[0].value == "המועמד אושר והופעל"
    assert app.metric[0].value == "1"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["experiment_id"] == pending.experiment_id
    assert report["candidate_id"] == pending.approval["candidate_id"]
    assert report["approval_decision"] == "approved"

    app.run()
    assert not app.exception
    assert all(button.key != "dashboard_approve" for button in app.button)
    with closing(ConfigurationStore(database_path=database)) as store:
        assert [version.version for version in store.history] == [0, 1]
    model.invoke.assert_not_called()


def test_new_dashboard_session_restores_and_rejects_without_model_calls(saved_experiment):
    pending, report_path, database, model = saved_experiment
    first = AppTest.from_file(str(APP), default_timeout=20).run()
    _load_experiment(first, pending.experiment_id)
    second = AppTest.from_file(str(APP), default_timeout=20).run()
    _load_experiment(second, pending.experiment_id)
    assert first.session_state.dashboard_result == second.session_state.dashboard_result

    second.button(key="dashboard_reject").click().run()

    assert not second.exception
    assert second.subheader[0].value == "Candidate rejected"
    assert second.metric[0].value == "0"
    assert json.loads(report_path.read_text(encoding="utf-8"))["approval_decision"] == "rejected"
    first.button(key="dashboard_approve").click().run()
    assert not first.exception
    assert first.subheader[0].value == "Candidate rejected"
    with closing(ConfigurationStore(database_path=database)) as store:
        assert store.active.version == 0
    model.invoke.assert_not_called()


def test_playground_processes_once_and_keeps_result_on_language_change(
    saved_experiment, approval_inputs
):
    _, _, _, model = saved_experiment
    cases, _, _, _ = approval_inputs
    model.invoke.side_effect = None
    model.invoke.return_value = model_response(cases[0])
    app = AppTest.from_file(str(APP), default_timeout=20).run()

    app.sidebar.radio(key="dashboard_screen").set_value("playground").run()
    model.invoke.assert_not_called()
    next(button for button in app.button if button.label == "Process order").click().run()

    assert not app.exception
    assert app.subheader[0].value == "Extracted order"
    original = app.session_state.dashboard_order
    assert original.version == 0
    assert original.result.order == cases[0].expected_output
    assert not original.result.report_path.exists()
    assert app.metric[2].value == "Unavailable"

    app.sidebar.selectbox(key="dashboard_language").set_value("עברית").run()
    assert not app.exception
    assert app.header[0].value == "עיבוד הזמנות"
    assert app.subheader[0].value == "הזמנה שחולצה"
    assert app.session_state.dashboard_order == original
    model.invoke.assert_called_once()


@pytest.mark.parametrize("language", TRANSLATIONS)
def test_confirmed_rollback_changes_next_playground_order(
    saved_experiment, approval_inputs, language
):
    pending, _, database, model = saved_experiment
    cases, _, _, _ = approval_inputs
    candidate = pending.approval["candidate_id"]
    with closing(ConfigurationStore(database_path=database)) as store:
        thread, _, _ = store.load_experiment(pending.experiment_id)
        store.record_decision(thread, candidate, "approve")
        store.activate(thread, candidate)
    model.invoke.side_effect = None
    model.invoke.return_value = model_response(cases[0])

    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.sidebar.selectbox(key="dashboard_language").set_value(language).run()
    labels = TRANSLATIONS[language]
    _navigate(app, "playground")
    _click(app, labels["process_order"])
    assert app.session_state.dashboard_order.version == 1
    assert json.loads(model.invoke.call_args.args[0][1].content)["document"]

    _navigate(app, "history")
    assert app.metric[0].value == "1"
    _click(app, labels["rollback"])
    assert app.error
    assert app.metric[0].value == "1"
    app.checkbox(key="dashboard_confirm_rollback").check()
    _click(app, labels["rollback"])
    assert not app.exception
    assert app.metric[0].value == "2"
    assert app.success[0].value == labels["rollback_done"].format(version=2)
    assert not app.checkbox(key="dashboard_confirm_rollback").value

    _navigate(app, "playground")
    _click(app, labels["process_order"])
    assert not app.exception
    assert app.session_state.dashboard_order.version == 2
    instruction, _ = model.invoke.call_args.args[0]
    with closing(ConfigurationStore(database_path=database)) as store:
        assert store.active.restored_from == 0
        assert instruction.content == store.history[0].candidate.system_prompt
    assert model.invoke.call_count == 2


@pytest.mark.parametrize("language", TRANSLATIONS)
def test_experiments_generate_and_compare_reports_with_mocked_model(
    saved_experiment, approval_inputs, tmp_path, monkeypatch, language
):
    _, _, _, model = saved_experiment
    cases, _, _, _ = approval_inputs
    reports_dir = tmp_path / "evaluation"
    monkeypatch.setattr(workspace, "EXPERIMENTS_DIR", reports_dir)
    model.invoke.side_effect = [model_response(case) for case in cases]
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.sidebar.selectbox(key="dashboard_language").set_value(language).run()
    labels = TRANSLATIONS[language]
    _navigate(app, "experiments")
    assert any(labels["no_reports"] == info.value for info in app.info)
    _click(app, labels["generate_baseline"])
    assert not app.exception
    assert not app.error
    baseline_id = app.selectbox(key="dashboard_report_id").value
    assert (reports_dir / baseline_id / "evaluation_report.json").is_file()
    assert json.loads(app.json[0].value)["total_cases"] == 12

    original = json.loads((reports_dir / baseline_id / "evaluation_report.json").read_bytes())
    for name, changes in (("legacy", None), ("other-model", {"model_name": "other"}),
                          ("operational-failure", {}),
                          ("other-split", {"split": "held_out"})):
        report = json.loads(json.dumps(original))
        report["experiment_id"] = name
        if changes is None:
            report.pop("configuration")
        else:
            report["configuration"].update(changes)
            if "split" in changes:
                for case in report["cases"]:
                    case["split"] = "held_out"
            if name == "operational-failure":
                for case in report["cases"]:
                    case["status"] = "error"
        directory = reports_dir / name
        directory.mkdir()
        (directory / "evaluation_report.json").write_text(json.dumps(report), encoding="utf-8")
    app.run()
    assert any(info.value == labels["no_comparable_reports"] for info in app.info)
    assert not any(button.label == labels["compare"] for button in app.button)
    app.selectbox(key="dashboard_report_id").set_value("legacy").run()
    assert json.loads(app.json[0].value)["total_cases"] == 12
    app.selectbox(key="dashboard_report_id").set_value("operational-failure").run()
    assert app.warning[0].value == labels["failed_report"]

    model.invoke.side_effect = [model_response(case) for case in cases]
    candidate = run_evaluation(
        "development", max_cases=12, context_isolation=True, workspace_dir=reports_dir
    )
    app.run()
    assert set(app.selectbox(key="dashboard_compare_before").options) == {
        baseline_id, candidate.experiment_id
    }
    app.selectbox(key="dashboard_compare_before").set_value(baseline_id).run()
    assert app.selectbox(key=f"dashboard_compare_after_{baseline_id}").options == [
        candidate.experiment_id
    ]
    _click(app, labels["compare"])

    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "0"
    comparison = app.session_state.dashboard_comparison
    assert comparison.configuration_changes == {"context_isolation": (False, True)}
    assert comparison.metrics["task_success_rate"].difference == 0
    other_language = "עברית" if language == "English" else "English"
    app.sidebar.selectbox(key="dashboard_language").set_value(other_language).run()
    assert not app.exception
    assert app.metric[0].label == TRANSLATIONS[other_language]["regressions"]
    assert app.session_state.dashboard_comparison == comparison

    app.selectbox(key="dashboard_compare_before").set_value(candidate.experiment_id).run()
    assert app.selectbox(key=f"dashboard_compare_after_{candidate.experiment_id}").value == baseline_id
    _click(app, TRANSLATIONS[other_language]["compare"])
    reversed_comparison = app.session_state.dashboard_comparison
    assert reversed_comparison.baseline_experiment_id == candidate.experiment_id
    assert reversed_comparison.candidate_experiment_id == baseline_id
    assert model.invoke.call_count == 24


@pytest.mark.parametrize("language", TRANSLATIONS)
def test_investigation_displays_real_diagnostic_evidence_and_preserves_pending_state(
    saved_experiment, approval_inputs, tmp_path, monkeypatch, language
):
    _, _, _, model = saved_experiment
    cases, _, investigation, _ = approval_inputs
    reports_dir = tmp_path / "evaluation"
    monkeypatch.setattr(workspace, "EXPERIMENTS_DIR", reports_dir)
    model.invoke.side_effect = [
        model_response(case, wrong=index == 0) for index, case in enumerate(cases)
    ]
    baseline = run_evaluation("development", max_cases=12, workspace_dir=reports_dir)
    model.invoke.side_effect = [model_response(cases[0])] + [
        model_response(case) for case in cases
    ]
    proposal = investigation["proposal"]
    model.with_structured_output.return_value.invoke.return_value = proposal
    app = AppTest.from_file(str(APP), default_timeout=20).run()

    app.sidebar.selectbox(key="dashboard_language").set_value(language).run()
    labels = TRANSLATIONS[language]
    _navigate(app, "investigation")
    assert app.selectbox(key="dashboard_investigation_baseline").value == baseline.experiment_id
    assert model.invoke.call_count == 12
    _click(app, labels["investigate"])
    assert not app.exception
    assert not app.error
    assert app.subheader[0].value == labels["awaiting_decision"]
    assert app.text[0].value == proposal.hypothesis
    pending, report = app.session_state.dashboard_investigation
    assert pending.experiment_id == app.code[0].value
    assert report["diagnostics"][0]["cases"][0]["status"] == "passed"

    _navigate(app, "experiments")
    app.sidebar.selectbox(key="dashboard_language").set_value("עברית").run()
    _navigate(app, "investigation")
    assert not app.exception
    assert app.header[0].value == "חקירה"
    assert app.subheader[0].value == "ממתין להחלטה אנושית"
    assert app.session_state.dashboard_investigation[0] == pending
    model.with_structured_output.return_value.invoke.assert_called_once()
    assert model.invoke.call_count == 25


def test_dashboard_budget_tracks_split_and_rejects_stale_displayed_limit(
    saved_experiment, approval_inputs, tmp_path, monkeypatch
):
    _, _, _, model = saved_experiment
    cases, _, _, _ = approval_inputs
    reports_dir = tmp_path / "evaluation"
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    development = datasets_dir / "development.json"
    monkeypatch.setattr(workspace, "EXPERIMENTS_DIR", reports_dir)
    monkeypatch.setattr(dataset_loader, "DATASET_DIR", datasets_dir)
    development.write_bytes(dataset_loader.ADAPTER.dump_json(cases))
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.sidebar.radio(key="dashboard_screen").set_value("experiments").run()
    assert any("12 development cases" in caption.value for caption in app.caption)

    additional = cases[1].model_copy(update={
        "case_id": "N99", "document": cases[1].document + "\nAdditional synthetic note."
    })
    expanded = [*cases, additional]
    development.write_bytes(dataset_loader.ADAPTER.dump_json(expanded))
    next(button for button in app.button if button.label == "Run development baseline").click().run()
    assert app.error[0].value == "The development baseline could not be completed."
    assert any("13 development cases" in caption.value for caption in app.caption)
    model.invoke.assert_not_called()
    assert not reports_dir.exists()

    model.invoke.side_effect = [model_response(case) for case in expanded]
    next(button for button in app.button if button.label == "Run development baseline").click().run()
    assert not app.exception
    assert not app.error
    assert json.loads(app.json[0].value)["total_cases"] == 13
    app.sidebar.radio(key="dashboard_screen").set_value("investigation").run()
    assert any("up to 13 regression cases" in caption.value for caption in app.caption)
    assert model.invoke.call_count == 13


def test_dashboard_shares_expanded_edition_across_screens_and_languages(
    saved_experiment, tmp_path, monkeypatch
):
    _, _, _, model = saved_experiment
    reports_dir = tmp_path / "evaluation"
    monkeypatch.setattr(workspace, "EXPERIMENTS_DIR", reports_dir)
    original_cases = dataset_loader.load_cases("development")
    model.invoke.side_effect = [model_response(case) for case in original_cases]
    original = run_evaluation("development", max_cases=12, workspace_dir=reports_dir)
    model.invoke.reset_mock()

    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.sidebar.radio(key="dashboard_screen").set_value("experiments").run()
    assert app.sidebar.selectbox(key="dashboard_dataset_edition_en").value == "Original"
    assert app.selectbox(key="dashboard_report_id").value == original.experiment_id
    app.sidebar.selectbox(key="dashboard_dataset_edition_en").set_value("Expanded").run()
    assert any("24 development cases" in caption.value for caption in app.caption)
    assert any("No local evaluation reports" in item.value for item in app.info)
    model.invoke.assert_not_called()

    expanded_cases = dataset_loader.load_cases("development", dataset_edition="expanded")
    model.invoke.side_effect = [model_response(case) for case in expanded_cases]
    next(button for button in app.button if button.label == "Run development baseline").click().run()
    assert not app.exception
    assert not app.error
    expanded_id = app.selectbox(key="dashboard_report_id").value
    payload = json.loads((reports_dir / expanded_id / "evaluation_report.json").read_bytes())
    assert payload["configuration"]["dataset_edition"] == "expanded"
    assert payload["metrics"]["total_cases"] == 24

    app.sidebar.selectbox(key="dashboard_language").set_value("עברית").run()
    assert not app.exception
    assert app.sidebar.selectbox(key="dashboard_dataset_edition_he").value == "מורחבת"
    assert app.session_state.dashboard_selected_edition == "expanded"
    app.sidebar.selectbox(key="dashboard_language").set_value("English").run()
    assert not app.exception
    assert app.sidebar.selectbox(key="dashboard_dataset_edition_en").value == "Expanded"

    app.sidebar.radio(key="dashboard_screen").set_value("overview").run()
    app.sidebar.selectbox(key="dashboard_language").set_value("עברית").run()
    _navigate(app, "investigation")
    assert not app.exception
    assert app.sidebar.selectbox(key="dashboard_dataset_edition_he").value == "מורחבת"
    assert app.session_state.dashboard_selected_edition == "expanded"
    assert app.selectbox(key="dashboard_investigation_baseline").value == expanded_id
    assert any("24 מקרי רגרסיה" in caption.value for caption in app.caption)
    model.with_structured_output.assert_not_called()
    assert model.invoke.call_count == 24


@pytest.mark.parametrize(
    "stored,expected_count,expected_id",
    [("Original", 12, "original"), ("Expanded", 24, "expanded"), ("מורחבת", 24, "expanded")]
)
def test_dashboard_recovers_saved_labels_from_either_language(
    saved_experiment, stored, expected_count, expected_id
):
    _, _, _, model = saved_experiment
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.session_state.dashboard_selected_edition = stored

    app.sidebar.radio(key="dashboard_screen").set_value("experiments").run()

    assert not app.exception
    assert app.session_state.dashboard_selected_edition == expected_id
    assert any(f"{expected_count} development cases" in part.value for part in app.caption)
    model.invoke.assert_not_called()


@pytest.mark.parametrize("language", TRANSLATIONS)
@pytest.mark.parametrize(
    "decision,expected,version",
    [("approve", "activated", 1), ("reject", "rejected", 0), ("revise", "revision_requested", 0)]
)
def test_dashboard_approval_actions_and_invalid_id(saved_experiment, language, decision, expected, version):
    pending, report_path, _, model = saved_experiment
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.sidebar.selectbox(key="dashboard_language").set_value(language).run()
    labels = TRANSLATIONS[language]

    _load_experiment(app, "missing")
    assert not app.exception
    assert app.error[0].value == labels["load_error"]
    _load_experiment(app, pending.experiment_id)
    assert not app.exception
    assert not app.error
    assert app.button(key=f"dashboard_{decision}").label == labels[decision]
    app.button(key=f"dashboard_{decision}").click().run()

    assert not app.exception
    assert app.subheader[0].value == labels[expected]
    assert app.metric[0].value == str(version)
    assert json.loads(report_path.read_bytes())["active_version"] == version
    model.invoke.assert_not_called()


def test_unknown_saved_edition_requires_explicit_selection(saved_experiment):
    _, _, _, model = saved_experiment
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.session_state.dashboard_selected_edition = "unknown-edition"

    app.sidebar.radio(key="dashboard_screen").set_value("experiments").run()

    assert not app.exception
    assert app.warning
    assert not any(button.label == "Run development baseline" for button in app.button)
    app.sidebar.selectbox(key="dashboard_dataset_edition_en").set_value("Expanded").run()
    assert not app.exception
    assert app.session_state.dashboard_selected_edition == "expanded"
    assert any("24 development cases" in part.value for part in app.caption)
    model.invoke.assert_not_called()


def test_navigation_through_all_screens_survives_live_language_switches(saved_experiment):
    _, _, _, model = saved_experiment
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    for screen in ("overview", "playground", "investigation", "experiments", "approval", "history"):
        _navigate(app, screen)
        app.sidebar.selectbox(key="dashboard_language").set_value("עברית").run()
        assert not app.exception
        assert app.header[0].value == TRANSLATIONS["עברית"][screen]
        app.sidebar.selectbox(key="dashboard_language").set_value("English").run()
        assert not app.exception
        assert app.header[0].value == TRANSLATIONS["English"][screen]
    model.invoke.assert_not_called()


@pytest.mark.parametrize("language", TRANSLATIONS)
def test_invalid_document_is_reported_before_calling_model(saved_experiment, language):
    _, _, _, model = saved_experiment
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.sidebar.selectbox(key="dashboard_language").set_value(language).run()
    _navigate(app, "playground")
    app.text_input(key="dashboard_document_id").set_value("invalid document ID")
    _click(app, TRANSLATIONS[language]["process_order"])
    assert not app.exception
    assert app.error[0].value == TRANSLATIONS[language]["processing_error"]
    model.invoke.assert_not_called()


@pytest.mark.parametrize("language", TRANSLATIONS)
def test_investigation_can_create_baseline_and_handles_connection_failure(
    saved_experiment, approval_inputs, tmp_path, monkeypatch, language
):
    _, _, _, model = saved_experiment
    cases, _, _, _ = approval_inputs
    reports = tmp_path / "reports"
    monkeypatch.setattr(workspace, "EXPERIMENTS_DIR", reports)
    model.invoke.side_effect = type("OpenAIConnectionError", (Exception,), {})()
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.sidebar.selectbox(key="dashboard_language").set_value(language).run()
    labels = TRANSLATIONS[language]
    _navigate(app, "investigation")

    app.button(key="dashboard_create_investigation_baseline").click().run()
    assert not app.exception
    assert app.error[0].value == labels["baseline_connection_error"]
    assert not list(reports.glob("*/evaluation_report.json"))
    model.invoke.assert_called_once()

    model.invoke.side_effect = [model_response(case) for case in cases]
    app.button(key="dashboard_create_investigation_baseline").click().run()
    assert not app.exception
    assert not app.error
    report_id = app.selectbox(key="dashboard_investigation_baseline").value
    assert (reports / report_id / "evaluation_report.json").is_file()
    _click(app, labels["investigate"])
    assert app.subheader[0].value == labels["no_evidence"]
    assert model.invoke.call_count == 13
    model.with_structured_output.assert_not_called()


@pytest.mark.parametrize("language", TRANSLATIONS)
def test_identical_configuration_comparison_is_identified_as_repeatability(
    saved_experiment, approval_inputs, tmp_path, monkeypatch, language
):
    _, _, _, model = saved_experiment
    cases, _, _, _ = approval_inputs
    monkeypatch.setattr(workspace, "EXPERIMENTS_DIR", tmp_path / "evaluation")
    model.invoke.side_effect = [model_response(case) for case in cases] * 2
    for _ in range(2):
        run_evaluation("development", max_cases=12, workspace_dir=workspace.EXPERIMENTS_DIR)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.sidebar.selectbox(key="dashboard_language").set_value(language).run()
    _navigate(app, "experiments")
    labels = TRANSLATIONS[language]
    _click(app, labels["compare"])
    assert not app.exception
    assert not app.error
    assert labels["same_configuration"] in [item.value for item in app.info]
    assert app.session_state.dashboard_comparison.regressed_cases == 0
    assert model.invoke.call_count == 24
