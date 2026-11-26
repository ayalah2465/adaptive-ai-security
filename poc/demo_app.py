import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

import streamlit as st
from langchain_core.runnables import RunnableLambda

from poc.adaptive.configuration import ConfigurationStore
from poc.adaptive.experiment import AdaptiveExperiment
from poc.adaptive.feedback_loop import DefenseCandidate, InvestigationProposal
from poc.evaluation.dataset_loader import load_cases
from poc.evaluation.evaluator import evaluate_case
from poc.evaluation.models import CaseEvaluation
from poc.target.action_log import ActionEvent
from poc.target.extractor import SYSTEM_PROMPT


TRANSLATIONS = {
    "English": {
        "title": "Adaptive defense demo",
        "notice": "Demonstration with simulated model responses and synthetic evaluation cases.",
        "version": "Active configuration version",
        "start": "Run demonstration",
        "awaiting_decision": "Awaiting a human decision",
        "activate": "Approve",
        "reject": "Reject",
        "revise": "Request revision",
        "activated": "Candidate approved and activated",
        "rejected": "Candidate rejected",
        "revision_requested": "Revision requested",
        "report": "Experiment report"
    },
    "עברית": {
        "title": "הדגמת הגנה אדפטיבית",
        "notice": "הדגמה עם תשובות מודל מדומות ומקרי הערכה סינתטיים.",
        "version": "גרסת התצורה הפעילה",
        "start": "הפעלת ההדגמה",
        "awaiting_decision": "ממתין להחלטה אנושית",
        "activate": "אישור",
        "reject": "דחייה",
        "revise": "בקשת תיקון",
        "activated": "המועמד אושר והופעל",
        "rejected": "המועמד נדחה",
        "revision_requested": "התבקש תיקון",
        "report": "דוח הניסוי"
    }
}


@st.cache_resource
def _demo_lock() -> Lock:
    return Lock()


def _start_demo() -> None:
    cases = load_cases("development")
    diagnostic = cases[0]
    baseline = []
    for case in cases:
        output = case.expected_output
        if case.case_id == diagnostic.case_id:
            output = output.model_copy(update={"supplier": "Simulated supplier mismatch"})
        baseline.append(evaluate_case(case, CaseEvaluation(
            case_id=case.case_id,
            category=case.category,
            split=case.split,
            expected_output=case.expected_output,
            actual_output=output,
            status="pending",
            report_verified=True,
            actions=[
                ActionEvent("read_document", "executed", case.case_id),
                ActionEvent("save_report", "executed", case.case_id)
            ]
        )))

    proposal = InvestigationProposal(
        hypothesis="Clarifying which label identifies the supplier may improve extraction.",
        diagnostic_case_ids=[diagnostic.case_id],
        candidate=DefenseCandidate(
            system_prompt=f"{SYSTEM_PROMPT}\nUse the supplier named in the supplier field.",
            context_isolation=True,
            context_aware_tool_restriction=True
        )
    )
    responses = iter([diagnostic, *cases])

    def simulated_response(*args, **kwargs):
        case = next(responses)
        return SimpleNamespace(text=case.expected_output.model_dump_json(), usage_metadata={})

    temporary = TemporaryDirectory(prefix="adaptive-demo-")
    configurations = ConfigurationStore()
    experiment = AdaptiveExperiment(configurations, workspace_dir=Path(temporary.name))
    with _demo_lock(), patch(
        "poc.target.extractor.get_model", return_value=SimpleNamespace(invoke=simulated_response)
    ):
        result = experiment.start(
            baseline, RunnableLambda(lambda _: proposal), base_version=0,
            max_rounds=1, diagnostic_budget=1
        )
    st.session_state.demo_directory = temporary
    st.session_state.demo_configurations = configurations
    st.session_state.demo_experiment = experiment
    st.session_state.demo_result = result


def _decide(decision: str) -> None:
    result = st.session_state.demo_result
    if result.status == "awaiting_decision":
        st.session_state.demo_result = st.session_state.demo_experiment.resume(
            result.approval["candidate_id"], decision
        )


def main() -> None:
    st.set_page_config(page_title="Adaptive defense demo", layout="centered")
    language = st.sidebar.selectbox("Language / שפה", list(TRANSLATIONS), key="demo_language")
    labels = TRANSLATIONS[language]
    if language == "עברית":
        st.html("""
            <style>
            [data-testid="stMain"], [data-testid="stSidebarContent"] {
                direction: rtl;
                text-align: right;
            }
            </style>
        """)
    st.title(labels["title"])
    st.info(labels["notice"])

    configurations = st.session_state.get("demo_configurations")
    st.metric(labels["version"], configurations.active.version if configurations else 0)
    result = st.session_state.get("demo_result")
    if result is None:
        st.button(labels["start"], key="demo_start", on_click=_start_demo)
        return

    st.subheader(labels.get(result.status, result.status))
    if result.approval is not None:
        st.json(result.approval)
        approve, reject, revise = st.columns(3)
        approve.button(labels["activate"], key="demo_approve", on_click=_decide, args=("approve",))
        reject.button(labels["reject"], key="demo_reject", on_click=_decide, args=("reject",))
        revise.button(labels["revise"], key="demo_revise", on_click=_decide, args=("revise",))
    if result.report_path is not None:
        st.subheader(labels["report"])
        st.json(json.loads(result.report_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
