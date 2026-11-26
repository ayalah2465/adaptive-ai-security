import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from poc.demo_app import TRANSLATIONS

APP = Path(__file__).resolve().parents[1] / "demo_app.py"


@pytest.fixture
def app():
    demo = AppTest.from_file(str(APP), default_timeout=15).run()
    yield demo
    if "demo_directory" in demo.session_state:
        demo.session_state.demo_directory.cleanup()


def test_switching_language_preserves_pending_candidate_and_allows_approval(app):
    assert not app.exception
    assert app.title[0].value == "Adaptive defense demo"
    assert app.metric[0].value == "0"

    app.button(key="demo_start").click().run()
    assert not app.exception
    assert app.subheader[0].value == "Awaiting a human decision"
    candidate_id = app.session_state.demo_result.approval["candidate_id"]

    app.sidebar.selectbox(key="demo_language").set_value("עברית").run()
    assert not app.exception
    assert app.title[0].value == "הדגמת הגנה אדפטיבית"
    assert app.session_state.demo_result.approval["candidate_id"] == candidate_id
    assert app.button(key="demo_approve").label == "אישור"
    assert app.metric[0].value == "0"

    app.button(key="demo_approve").click().run()
    assert not app.exception
    assert app.subheader[0].value == "המועמד אושר והופעל"
    assert app.metric[0].value == "1"
    report_path = app.session_state.demo_result.report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["candidate_id"] == candidate_id
    assert report["approval_decision"] == "approved"
    assert report["active_version"] == 1
    assert report["regression"]["total_cases"] == 12
    app.run()
    assert app.session_state.demo_result.report_path == report_path
    assert app.metric[0].value == "1"


@pytest.mark.parametrize(
    "button,status",
    [
        ("demo_reject", "rejected"),
        ("demo_revise", "revision_requested")
    ]
)
@pytest.mark.parametrize("language", TRANSLATIONS)
def test_other_decisions_preserve_baseline(app, button, status, language):
    app.sidebar.selectbox(key="demo_language").set_value(language).run()
    app.button(key="demo_start").click().run()
    assert not app.exception
    app.button(key=button).click().run()

    assert not app.exception
    assert app.subheader[0].value == TRANSLATIONS[language][status]
    assert app.metric[0].value == "0"
    report = json.loads(app.session_state.demo_result.report_path.read_text(encoding="utf-8"))
    assert report["approval_decision"] == status
    assert report["active_version"] == 0
