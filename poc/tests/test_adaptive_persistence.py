import json
from contextlib import closing
from unittest.mock import patch

import pytest
from langchain_core.runnables import RunnableLambda

from poc.adaptive.configuration import ConfigurationStore
from poc.adaptive.experiment import AdaptiveExperiment
from poc.tests.test_adaptive_approval_workflow import approval_inputs, model_response


@pytest.fixture
def start_persisted(tmp_path, approval_inputs):
    cases, baseline, investigation, model = approval_inputs

    def start(store):
        model.invoke.side_effect = [model_response(case) for case in cases]
        thread_id, state = store.start_approval(
            investigation, baseline, base_version=store.active.version, workspace_dir=tmp_path
        )
        assert state["status"] == "awaiting_decision"
        return thread_id, state["__interrupt__"][0].value

    return start


def test_pending_approval_and_activation_survive_reopen(tmp_path, start_persisted, approval_inputs):
    _, _, _, model = approval_inputs
    path = tmp_path / "approvals.db"
    with closing(ConfigurationStore(database_path=path)) as store:
        thread, request = start_persisted(store)
        assert store.active.version == 0

    model.invoke.reset_mock()
    candidate_id = request["candidate_id"]
    with closing(ConfigurationStore(database_path=path)) as store:
        assert store.pending_approval(thread) == request
        with pytest.raises(ValueError, match="does not match the pending approval"):
            store.record_decision(thread, "wrong-candidate", "approve")
        assert store.pending_approval(thread) == request
        assert store.record_decision(thread, candidate_id, "approve")["status"] == "approved"
        assert store.active.version == 0

    with closing(ConfigurationStore(database_path=path)) as store:
        activated = store.activate(thread, candidate_id)
        assert activated.version == 1
        assert activated.candidate.model_dump() == request["candidate"]
        assert store.history[0].version == 0

    with closing(ConfigurationStore(database_path=path)) as store:
        assert store.active == activated
        with pytest.raises(ValueError, match="already been used"):
            store.activate(thread, candidate_id)
        with pytest.raises(ValueError, match="not awaiting a decision"):
            store.record_decision(thread, candidate_id, "reject")
        restored = store.rollback(0)

    with closing(ConfigurationStore(database_path=path)) as store:
        assert store.active == restored
        assert restored.restored_from == 0
        assert [version.version for version in store.history] == [0, 1, 2]
    model.invoke.assert_not_called()


def test_stale_approval_survives_reopen_without_becoming_usable(tmp_path, start_persisted):
    path = tmp_path / "approvals.db"
    with closing(ConfigurationStore(database_path=path)) as store:
        thread, request = start_persisted(store)
        candidate_id = request["candidate_id"]
        store.record_decision(thread, candidate_id, "approve")
        store.rollback(0)

    with closing(ConfigurationStore(database_path=path)) as store:
        with pytest.raises(ValueError, match="stale"):
            store.activate(thread, candidate_id)
        assert store.active.version == 1
        assert len(store.history) == 2


@pytest.fixture
def start_experiment(tmp_path, approval_inputs):
    cases, baseline, investigation, model = approval_inputs

    def start(store):
        model.invoke.side_effect = [model_response(cases[0])] + [
            model_response(case) for case in cases
        ]
        experiment = AdaptiveExperiment(store, workspace_dir=tmp_path)
        pending = experiment.start(
            baseline, RunnableLambda(lambda _: investigation["proposal"]),
            base_version=0, max_rounds=1, diagnostic_budget=1
        )
        assert pending.status == "awaiting_decision"
        return experiment, pending

    return start


def test_experiment_restores_pending_approval_and_original_report(
    tmp_path, start_experiment, approval_inputs
):
    _, _, _, model = approval_inputs
    path = tmp_path / "approvals.db"
    with closing(ConfigurationStore(database_path=path)) as store:
        _, pending = start_experiment(store)
        thread, report_path, _ = store.load_experiment(pending.experiment_id)
        assert not report_path.exists()

    model.invoke.reset_mock()
    with closing(ConfigurationStore(database_path=path)) as store:
        experiment = AdaptiveExperiment(store)
        restored = experiment.restore(pending.experiment_id)
        assert restored == pending
        assert store.pending_approval(thread) == pending.approval
        final = experiment.resume(restored.approval["candidate_id"], "reject")
        assert final.report_path == report_path
        assert final.status == "rejected"
        assert store.active.version == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["experiment_id"] == pending.experiment_id
    assert report["approval_decision"] == "rejected"
    assert len(report["diagnostics"]) == 1
    assert report["regression"] == pending.approval["regression"]
    model.invoke.assert_not_called()


@pytest.mark.parametrize("failure_point", ["before_finalization", "before_publication"])
def test_report_recovery_after_activation_does_not_reactivate(
    tmp_path, start_experiment, approval_inputs, failure_point
):
    _, _, _, model = approval_inputs
    path = tmp_path / "approvals.db"
    with closing(ConfigurationStore(database_path=path)) as store:
        experiment, pending = start_experiment(store)
        _, report_path, _ = store.load_experiment(pending.experiment_id)
        failure = OSError("simulated report interruption")
        patcher = (
            patch.object(store, "finalize_experiment", side_effect=failure)
            if failure_point == "before_finalization"
            else patch("poc.adaptive.experiment.os.link", side_effect=failure)
        )
        with patcher, pytest.raises(OSError, match="simulated report interruption"):
            experiment.resume(pending.approval["candidate_id"], "approve")
        assert store.active.version == 1
        assert not report_path.exists()

    model.invoke.reset_mock()
    with closing(ConfigurationStore(database_path=path)) as store:
        restored = AdaptiveExperiment(store).restore(pending.experiment_id)
        assert restored.status == "activated"
        assert restored.report_path == report_path
        assert [version.version for version in store.history] == [0, 1]
        original = report_path.read_bytes()
        assert json.loads(original)["activated_version"] == 1
        assert AdaptiveExperiment(store).restore(pending.experiment_id) == restored
        assert report_path.read_bytes() == original
        assert store.active.version == 1

    model.invoke.assert_not_called()


def test_recorded_decision_can_finish_activation_after_reopen(
    tmp_path, start_experiment, approval_inputs
):
    _, _, _, model = approval_inputs
    path = tmp_path / "approvals.db"
    with closing(ConfigurationStore(database_path=path)) as store:
        _, pending = start_experiment(store)
        thread, _, _ = store.load_experiment(pending.experiment_id)
        store.record_decision(thread, pending.approval["candidate_id"], "approve")
        assert store.active.version == 0

    model.invoke.reset_mock()
    with closing(ConfigurationStore(database_path=path)) as store:
        restored = AdaptiveExperiment(store).restore(pending.experiment_id)
        assert restored.status == "activated"
        assert store.active.version == 1
        assert json.loads(restored.report_path.read_bytes())["approval_decision"] == "approved"
    model.invoke.assert_not_called()
