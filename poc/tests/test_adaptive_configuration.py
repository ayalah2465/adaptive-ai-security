import json

import pytest

from poc.adaptive.configuration import ConfigurationStore
from poc.target.extractor import CONTEXT_ISOLATION_INSTRUCTION, SYSTEM_PROMPT
from poc.tests.test_adaptive_approval_workflow import approval_inputs, model_response


@pytest.fixture
def start_candidate(tmp_path, approval_inputs):
    cases, baseline, investigation, model = approval_inputs

    def start(store, *, base_version=0):
        model.invoke.side_effect = [model_response(case) for case in cases]
        thread_id, state = store.start_approval(
            investigation, baseline, base_version=base_version, workspace_dir=tmp_path
        )
        assert state["status"] == "awaiting_decision"
        assert state["__interrupt__"][0].value["base_version"] == base_version
        return thread_id, state["candidate_id"]

    return start


def test_approval_activates_exact_candidate_once(start_candidate):
    store = ConfigurationStore()
    assert store.active.candidate.system_prompt == SYSTEM_PROMPT
    thread, candidate_id = start_candidate(store)

    with pytest.raises(ValueError, match="no final approval"):
        store.activate(thread, candidate_id)
    approved = store.record_decision(thread, candidate_id, "approve")
    with pytest.raises(ValueError, match="does not match"):
        store.activate(thread, "wrong-candidate")

    active = store.activate(thread, candidate_id)

    assert active.version == 1
    assert active.candidate_id == candidate_id
    assert active.candidate.model_dump() == approved["candidate"]
    assert store.active == active
    active.candidate.system_prompt = "Local copy changed"
    assert store.active.candidate.system_prompt == approved["candidate"]["system_prompt"]
    with pytest.raises(ValueError, match="already been used"):
        store.activate(thread, candidate_id)


@pytest.mark.parametrize(
    "decision,status",
    [("reject", "rejected"), ("revise", "revision_requested")]
)
def test_unapproved_candidate_leaves_active_version_unchanged(
    start_candidate, decision, status
):
    store = ConfigurationStore()
    thread, candidate_id = start_candidate(store)
    state = store.record_decision(thread, candidate_id, decision)

    assert state["status"] == status
    with pytest.raises(ValueError, match="no final approval"):
        store.activate(thread, candidate_id)
    assert len(store.history) == 1
    assert store.active.version == 0


def test_invalid_decisions_leave_candidate_pending(start_candidate, approval_inputs):
    _, _, _, model = approval_inputs
    store = ConfigurationStore()
    thread, candidate_id = start_candidate(store)

    with pytest.raises(ValueError, match="does not match the pending approval"):
        store.record_decision(thread, "wrong-candidate", "approve")
    with pytest.raises(ValueError, match="Unsupported approval decision"):
        store.record_decision(thread, candidate_id, "unsupported")

    assert store.active.version == 0
    approved = store.record_decision(thread, candidate_id, "approve")
    assert approved["status"] == "approved"
    with pytest.raises(ValueError, match="not awaiting a decision"):
        store.record_decision(thread, candidate_id, "reject")
    assert model.invoke.call_count == 12
    assert store.activate(thread, candidate_id).version == 1


def test_stale_approval_cannot_activate(start_candidate):
    store = ConfigurationStore()
    first = start_candidate(store)
    second = start_candidate(store)
    store.record_decision(*first, "approve")
    store.record_decision(*second, "approve")

    store.activate(*first)

    with pytest.raises(ValueError, match="stale"):
        store.activate(*second)
    assert len(store.history) == 2
    assert store.active.candidate_id == first[1]
    with pytest.raises(ValueError, match="no longer active"):
        start_candidate(store)


def test_rollback_appends_version_and_preserves_history(start_candidate):
    store = ConfigurationStore()
    approval = start_candidate(store)
    store.record_decision(*approval, "approve")
    activated = store.activate(*approval)

    restored = store.rollback(0)

    assert [version.version for version in store.history] == [0, 1, 2]
    assert store.history[1] == activated
    assert restored.restored_from == 0
    assert restored.candidate == store.history[0].candidate
    assert store.active == restored
    with pytest.raises(ValueError, match="Unknown"):
        store.rollback(-1)
    assert len(store.history) == 3


def test_order_uses_approved_snapshot_then_restored_configuration(
    tmp_path, start_candidate, approval_inputs
):
    cases, _, investigation, model = approval_inputs
    store = ConfigurationStore()
    approval = start_candidate(store)
    store.record_decision(*approval, "approve")
    store.activate(*approval)
    documents = tmp_path / "orders"
    documents.mkdir()
    document = "Synthetic purchase order"
    (documents / "order_001.txt").write_text(document, encoding="utf-8")

    def finish_after_rollback(*args, **kwargs):
        store.rollback(0)
        return model_response(cases[0])

    model.invoke.side_effect = finish_after_rollback
    first = store.process_with_active_configuration(
        documents_dir=documents, reports_dir=tmp_path / "first"
    )

    assert first.version == 1
    assert first.result.order == cases[0].expected_output
    instruction, data = model.invoke.call_args.args[0]
    candidate = investigation["proposal"].candidate
    assert instruction.content == f"{candidate.system_prompt}\n\n{CONTEXT_ISOLATION_INSTRUCTION}"
    assert json.loads(data.content) == {"document": document}

    model.invoke.side_effect = None
    model.invoke.return_value = model_response(cases[0])
    second = store.process_with_active_configuration(
        documents_dir=documents, reports_dir=tmp_path / "second"
    )

    assert second.version == 2
    instruction, data = model.invoke.call_args.args[0]
    assert instruction.content == SYSTEM_PROMPT
    assert data.content == document
