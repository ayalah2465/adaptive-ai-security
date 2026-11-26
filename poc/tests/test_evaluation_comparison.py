from dataclasses import replace

import pytest

from poc.evaluation.comparison import compare_evaluation_reports
from poc.evaluation.dataset_loader import load_case
from poc.evaluation.models import CaseEvaluation
from poc.evaluation.report_storage import save_experiment_report
from poc.evaluation.workspace import create_workspace
from poc.target.metrics import ExtractionMetrics


@pytest.fixture
def report_factory(tmp_path):
    cases = [load_case("N01"), load_case("I01")]
    baseline = [
        CaseEvaluation(
            case_id=case.case_id,
            category=case.category,
            split=case.split,
            expected_output=case.expected_output,
            status="passed",
            execution_seconds=0.5,
            metrics=ExtractionMetrics("mock", 0.1, 5, 2, 7, 1),
            false_positive=False if not case.contains_injection else None,
            attack_succeeded=False if case.contains_injection else None
        )
        for case in cases
    ]
    configuration = {
        "split": "development",
        "model_name": "mock",
        "max_cases": 2,
        "cases_sha256": "a" * 64,
        "system_prompt_sha256": "b" * 64,
        "context_isolation": False,
        "context_aware_tool_restriction": False
    }

    def save(results=None, *, legacy=False, **changes):
        workspace = create_workspace(tmp_path)
        return save_experiment_report(
            workspace, baseline if results is None else results,
            configuration=None if legacy else configuration | changes
        )

    return baseline, save


def test_reports_compare_changes_metrics_and_regressions(report_factory):
    baseline, save = report_factory
    candidate = [
        replace(
            baseline[0], status="failed", failures=["unauthorized_action"],
            execution_seconds=1.0, metrics=ExtractionMetrics("mock", 0.1, 8, 2, 10, 1)
        ),
        baseline[1]
    ]
    before = save()
    after = save(candidate, context_isolation=True, system_prompt_sha256="c" * 64)

    compared = compare_evaluation_reports(before, after)

    assert compared.baseline_experiment_id == before.parent.name
    assert compared.candidate_experiment_id == after.parent.name
    assert compared.split == "development"
    assert compared.configuration_changes == {
        "context_isolation": (False, True), "system_prompt_sha256": ("b" * 64, "c" * 64)
    }
    assert compared.regressed_cases == 1
    assert compared.metrics["task_success_rate"].baseline == 100.0
    assert compared.metrics["task_success_rate"].candidate == 50.0
    assert compared.metrics["task_success_rate"].difference == -50.0
    assert compared.metrics["unauthorized_cases"].difference == 1
    assert compared.metrics["execution_seconds"].difference == 0.5
    assert compared.metrics["total_tokens"].difference == 3
    assert compared.metrics["attack_success_rate"].difference == 0.0


@pytest.mark.parametrize(
    "key,value", [("split", "held_out"), ("model_name", "other"), ("cases_sha256", "c" * 64)]
)
def test_incompatible_configuration_is_rejected(report_factory, key, value):
    baseline, save = report_factory
    candidate = [replace(case, split="held_out") for case in baseline] if key == "split" else baseline
    before, after = save(), save(candidate, **{key: value})

    with pytest.raises(ValueError, match=key):
        compare_evaluation_reports(before, after)


def test_report_with_different_case_order_is_rejected(report_factory):
    baseline, save = report_factory

    with pytest.raises(ValueError, match="case IDs or order"):
        compare_evaluation_reports(save(), save(reversed(baseline)))


def test_legacy_report_without_configuration_is_rejected(report_factory):
    _, save = report_factory

    with pytest.raises(ValueError, match=r"configuration \(missing\)"):
        compare_evaluation_reports(save(legacy=True), save())


def test_unknown_metrics_have_no_difference(report_factory):
    baseline, save = report_factory
    candidate = [baseline[0], replace(baseline[1], attack_succeeded=None, metrics=None)]

    compared = compare_evaluation_reports(save(), save(candidate))

    for name in ("attack_success_rate", "input_tokens", "output_tokens", "total_tokens"):
        assert compared.metrics[name].baseline is not None
        assert compared.metrics[name].candidate is None
        assert compared.metrics[name].difference is None
