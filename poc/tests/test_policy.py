
import pytest

from poc.target import storage

from poc.target.action_log import ActionLog

from poc.target.models import PurchaseOrder

from poc.target.policy import (
    ActionContext,
    PolicyViolation,
    authorize_action,
    resolve_safe_path
)


def test_context_cannot_expand_mandatory_permissions():
    context = ActionContext("N01", frozenset({"delete_file"}))

    with pytest.raises(PolicyViolation, match="Action is not allowed"):
        authorize_action("delete_file", resource="N01", context=context)


def test_storage_blocks_document_outside_case_context(tmp_path):
    (tmp_path / "N02.txt").write_text("Synthetic order", encoding="utf-8")
    context = ActionContext("N01", frozenset({"read_document"}))
    log = ActionLog()

    with pytest.raises(PolicyViolation, match="outside the case context"):
        storage.read_document("N02", log, documents_dir=tmp_path, action_context=context)

    assert [event.status for event in log.events] == ["requested", "blocked"]


@pytest.mark.parametrize(
    "action",
    ["read_document", "save_report"]
)
def test_allowed_actions(action):

    authorize_action(action)


@pytest.mark.parametrize(
    "action",
    ["delete_file", "execute_command", "unknown"]
)
def test_forbidden_actions(action):

    with pytest.raises(PolicyViolation):
        authorize_action(action)


@pytest.mark.parametrize(
    "document_id",
    [
        "../outside",
        "..\\outside",
        "C:\\Windows\\file",
        "",
        "N01.txt"
    ]
)
def test_invalid_document_ids(
    tmp_path,
    document_id
):

    with pytest.raises(PolicyViolation):

        resolve_safe_path(
            tmp_path,
            document_id,
            ".txt"
        )


@pytest.mark.parametrize(
    "document_id",
    ["N01", "I02", "B03", "C04", "order_001"]
)
def test_valid_document_ids(
    tmp_path,
    document_id
):

    path = resolve_safe_path(
        tmp_path,
        document_id,
        ".txt"
    )

    assert path.parent == tmp_path.resolve()

    assert path.name == f"{document_id}.txt"


def test_forbidden_file_type(tmp_path):

    with pytest.raises(PolicyViolation):

        resolve_safe_path(
            tmp_path,
            "N01",
            ".exe"
        )


def test_real_document_read(
    tmp_path,
    monkeypatch
):

    monkeypatch.setattr(
        storage,
        "DOCUMENTS_DIR",
        tmp_path
    )

    document = tmp_path / "N01.txt"

    document.write_text(
        "Synthetic test document",
        encoding="utf-8"
    )

    log = ActionLog()

    result = storage.read_document(
        "N01",
        log
    )

    assert result == "Synthetic test document"
    assert log.count("requested") == 1
    assert log.count("executed") == 1
    assert log.count("blocked") == 0


def test_real_report_write(
    tmp_path,
    monkeypatch
):

    monkeypatch.setattr(
        storage,
        "REPORTS_DIR",
        tmp_path
    )

    order = PurchaseOrder(
        order_id="PO-1001",
        supplier="Alpha Technologies",
        amount="1250.00",
        currency="USD",
        delivery_date="2026-12-15"
    )

    log = ActionLog()

    path = storage.save_report(
        order,
        "N01",
        log
    )

    assert path.exists()

    saved_order = PurchaseOrder.model_validate_json(
        path.read_text(encoding="utf-8")
    )

    assert saved_order == order
    assert log.count("requested") == 1
    assert log.count("executed") == 1


def test_blocked_read():

    log = ActionLog()

    with pytest.raises(PolicyViolation):

        storage.read_document(
            "../outside",
            log
        )

    assert log.count("requested") == 1
    assert log.count("blocked") == 1
    assert log.count("executed") == 0


def test_blocked_write(
    tmp_path,
    monkeypatch
):

    monkeypatch.setattr(
        storage,
        "REPORTS_DIR",
        tmp_path
    )

    order = PurchaseOrder(
        order_id="PO-1001",
        supplier="Alpha Technologies",
        amount="1250.00",
        currency="USD",
        delivery_date="2026-12-15"
    )

    log = ActionLog()

    with pytest.raises(PolicyViolation):

        storage.save_report(
            order,
            "../outside",
            log
        )

    assert log.count("requested") == 1
    assert log.count("blocked") == 1
    assert log.count("executed") == 0

    assert not list(tmp_path.iterdir())


def test_missing_document(
    tmp_path,
    monkeypatch
):

    monkeypatch.setattr(
        storage,
        "DOCUMENTS_DIR",
        tmp_path
    )

    log = ActionLog()

    with pytest.raises(FileNotFoundError):

        storage.read_document(
            "N01",
            log
        )

    assert log.count("failed") == 1
    assert log.count("executed") == 0
    assert log.count("blocked") == 0

@pytest.mark.parametrize(
    "outside",
    [False, True]
)
def test_symbolic_link_blocked(
    tmp_path,
    outside
):

    documents_dir = tmp_path / "documents"

    documents_dir.mkdir()

    target_dir = (
        tmp_path
        if outside
        else documents_dir
    )

    target = target_dir / "N02.txt"

    target.write_text(
        "Synthetic document",
        encoding="utf-8"
    )

    link = documents_dir / "N01.txt"

    try:

        link.symlink_to(target)

    except (OSError, NotImplementedError):

        pytest.skip(
            "Symbolic links unavailable on this system"
        )

    with pytest.raises(PolicyViolation):

        resolve_safe_path(
            documents_dir,
            "N01",
            ".txt"
        )
