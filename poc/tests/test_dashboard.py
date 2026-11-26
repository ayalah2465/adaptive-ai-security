import json
from contextlib import closing
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from poc.adaptive.configuration import ConfigurationStore
from poc.evaluation import dataset_loader, workspace
from poc.evaluation.experiment import run_evaluation
from poc.tests.test_adaptive_approval_workflow import approval_inputs, model_response
from poc.tests.test_adaptive_persistence import start_experiment


APP = Path(__file__).resolve().parents[1] / "dashboard.py"


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
    app.sidebar.radio(key="dashboard_screen").set_value("approval").run()
    app.text_input(key="dashboard_experiment_id").set_value(experiment_id)
    next(button for button in app.button if button.label == "Load experiment").click().run()


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


def test_confirmed_rollback_changes_next_playground_order(
    saved_experiment, approval_inputs
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
    app.sidebar.radio(key="dashboard_screen").set_value("playground").run()
    next(button for button in app.button if button.label == "Process order").click().run()
    assert app.session_state.dashboard_order.version == 1
    assert json.loads(model.invoke.call_args.args[0][1].content)["document"]

    app.sidebar.radio(key="dashboard_screen").set_value("history").run()
    assert app.metric[0].value == "1"
    next(button for button in app.button if button.label == "Restore configuration").click().run()
    assert app.error
    assert app.metric[0].value == "1"
    app.checkbox(key="dashboard_confirm_rollback").check()
    next(button for button in app.button if button.label == "Restore configuration").click().run()
    assert not app.exception
    assert app.metric[0].value == "2"
    assert app.success[0].value == "Restored as configuration version 2."
    assert not app.checkbox(key="dashboard_confirm_rollback").value

    app.sidebar.radio(key="dashboard_screen").set_value("playground").run()
    next(button for button in app.button if button.label == "Process order").click().run()
    assert not app.exception
    assert app.session_state.dashboard_order.version == 2
    instruction, _ = model.invoke.call_args.args[0]
    with closing(ConfigurationStore(database_path=database)) as store:
        assert store.active.restored_from == 0
        assert instruction.content == store.history[0].candidate.system_prompt
    assert model.invoke.call_count == 2


def test_experiments_generate_and_compare_reports_with_mocked_model(
    saved_experiment, approval_inputs, tmp_path, monkeypatch
):
    _, _, _, model = saved_experiment
    cases, _, _, _ = approval_inputs
    reports_dir = tmp_path / "evaluation"
    monkeypatch.setattr(workspace, "EXPERIMENTS_DIR", reports_dir)
    model.invoke.side_effect = [model_response(case) for case in cases]
    app = AppTest.from_file(str(APP), default_timeout=20).run()

    app.sidebar.radio(key="dashboard_screen").set_value("experiments").run()
    assert any("No local evaluation reports" in info.value for info in app.info)
    next(button for button in app.button if button.label == "Run development baseline").click().run()
    assert not app.exception
    assert not app.error
    baseline_id = app.selectbox(key="dashboard_report_id").value
    assert (reports_dir / baseline_id / "evaluation_report.json").is_file()
    assert json.loads(app.json[0].value)["total_cases"] == 12

    model.invoke.side_effect = [model_response(case) for case in cases]
    candidate = run_evaluation(
        "development", max_cases=12, context_isolation=True, workspace_dir=reports_dir
    )
    app.run()
    app.selectbox(key="dashboard_compare_before").set_value(baseline_id)
    app.selectbox(key="dashboard_compare_after").set_value(candidate.experiment_id)
    next(button for button in app.button if button.label == "Compare reports").click().run()

    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "0"
    comparison = app.session_state.dashboard_comparison
    assert comparison.configuration_changes == {"context_isolation": (False, True)}
    assert comparison.metrics["task_success_rate"].difference == 0
    assert model.invoke.call_count == 24


def test_investigation_displays_real_diagnostic_evidence_and_preserves_pending_state(
    saved_experiment, approval_inputs, tmp_path, monkeypatch
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

    app.sidebar.radio(key="dashboard_screen").set_value("investigation").run()
    assert app.selectbox(key="dashboard_investigation_baseline").value == baseline.experiment_id
    assert model.invoke.call_count == 12
    next(button for button in app.button if button.label == "Start investigation").click().run()
    assert not app.exception
    assert not app.error
    assert app.subheader[0].value == "Awaiting a human decision"
    assert app.text[0].value == proposal.hypothesis
    pending, report = app.session_state.dashboard_investigation
    assert pending.experiment_id == app.code[0].value
    assert report["diagnostics"][0]["cases"][0]["status"] == "passed"

    app.sidebar.radio(key="dashboard_screen").set_value("experiments").run()
    app.sidebar.selectbox(key="dashboard_language").set_value("עברית").run()
    app.sidebar.radio(key="dashboard_screen").set_value("investigation").run()
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
    assert app.sidebar.selectbox(key="dashboard_dataset_edition").value == "original"
    assert app.selectbox(key="dashboard_report_id").value == original.experiment_id
    app.sidebar.selectbox(key="dashboard_dataset_edition").set_value("expanded").run()
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

    app.sidebar.radio(key="dashboard_screen").set_value("overview").run()
    app.sidebar.selectbox(key="dashboard_language").set_value("עברית").run()
    app.sidebar.radio(key="dashboard_screen").set_value("investigation").run()
    assert not app.exception
    assert app.sidebar.selectbox(key="dashboard_dataset_edition").value == "expanded"
    assert app.selectbox(key="dashboard_investigation_baseline").value == expanded_id
    assert any("24 מקרי רגרסיה" in caption.value for caption in app.caption)
    model.with_structured_output.assert_not_called()
    assert model.invoke.call_count == 24
