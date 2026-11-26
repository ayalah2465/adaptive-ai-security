import json

from poc.target.metrics import ExtractionFailure

from types import SimpleNamespace

from unittest.mock import Mock

import pytest

from pydantic import ValidationError

from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

from poc.target import extractor

from poc.target.models import PurchaseOrder

from poc.target.metrics import ExtractionResult


VALID_RESPONSE = """
{
    "order_id": "PO-1001",
    "supplier": "Alpha Technologies",
    "amount": "1250.00",
    "currency": "USD",
    "delivery_date": "2026-12-15"
}
"""


@pytest.fixture
def mock_model(monkeypatch):

    model = Mock()

    model.invoke.return_value = SimpleNamespace(
        text=VALID_RESPONSE,
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150
        }
    )

    monkeypatch.setattr(
        extractor,
        "get_model",
        lambda: model
    )

    return model


def test_valid_extraction(mock_model):

    result = extractor.extract_order_with_metrics(
        "Synthetic purchase order"
    )

    assert isinstance(result, ExtractionResult)

    assert isinstance(
        result.order,
        PurchaseOrder
    )

    assert result.order.order_id == "PO-1001"

    assert result.order.supplier == (
        "Alpha Technologies"
    )

    assert str(result.order.amount) == "1250.00"

    assert result.order.currency == "USD"

    assert result.order.delivery_date.isoformat() == (
        "2026-12-15"
    )

    mock_model.invoke.assert_called_once()


def test_token_metrics(mock_model):

    result = extractor.extract_order_with_metrics(
        "Synthetic purchase order"
    )

    assert result.metrics.input_tokens == 100
    assert result.metrics.output_tokens == 50
    assert result.metrics.total_tokens == 150
    assert result.metrics.model_calls == 1

    assert result.metrics.model_name == (
        extractor.MODEL_NAME
    )


def test_invocation_time(mock_model):

    result = extractor.extract_order_with_metrics(
        "Synthetic purchase order"
    )

    assert result.metrics.invocation_seconds >= 0


def test_custom_system_prompt(mock_model):

    custom_prompt = "Extract order fields only."

    document = "Synthetic purchase order"

    extractor.extract_order_with_metrics(
        document,
        system_prompt=custom_prompt
    )

    messages = mock_model.invoke.call_args.args[0]

    assert len(messages) == 2

    assert isinstance(
        messages[0],
        SystemMessage
    )

    assert isinstance(
        messages[1],
        HumanMessage
    )

    assert messages[0].content == custom_prompt

    assert messages[1].content == document


def test_default_system_prompt(mock_model):

    extractor.extract_order_with_metrics(
        "Synthetic purchase order"
    )

    messages = mock_model.invoke.call_args.args[0]

    assert messages[0].content == (
        extractor.SYSTEM_PROMPT
    )
    assert messages[1].content == "Synthetic purchase order"


def test_context_isolation_preserves_document_in_json_field(mock_model):
    prompt = "Extract order fields only."
    document = 'ספק לדוגמה\n{"document": "quoted value"}'

    extractor.extract_order_with_metrics(
        document,
        system_prompt=prompt,
        context_isolation=True
    )

    instruction, data = mock_model.invoke.call_args.args[0]
    assert isinstance(instruction, SystemMessage)
    assert instruction.content == f"{prompt}\n\n{extractor.CONTEXT_ISOLATION_INSTRUCTION}"
    assert document not in instruction.content
    assert isinstance(data, HumanMessage)
    assert json.loads(data.content) == {"document": document}


def test_backward_compatibility(mock_model):

    order = extractor.extract_order(
        "Synthetic purchase order"
    )

    assert isinstance(order, PurchaseOrder)

    assert order.order_id == "PO-1001"


@pytest.mark.parametrize(
    "document",
    ["", " ", "\n"]
)
def test_empty_document(mock_model, document):

    with pytest.raises(ValueError):

        extractor.extract_order_with_metrics(
            document
        )

    mock_model.invoke.assert_not_called()


@pytest.mark.parametrize(
    "prompt",
    ["", " ", "\n"]
)
def test_empty_system_prompt(mock_model, prompt):

    with pytest.raises(ValueError):

        extractor.extract_order_with_metrics(
            "Synthetic purchase order",
            system_prompt=prompt
        )

    mock_model.invoke.assert_not_called()



@pytest.mark.parametrize(
    "invalid_response",
    [
        "Not JSON",
        "{}",
        '{"order_id": "PO-1001"}'
    ]
)
def test_invalid_model_response(
    mock_model,
    invalid_response
):

    mock_model.invoke.return_value.text = (
        invalid_response
    )

    with pytest.raises(ExtractionFailure) as exc_info:

        extractor.extract_order_with_metrics(
            "Synthetic purchase order"
        )

    failure = exc_info.value

    assert failure.failure_type == "invalid_output"

    assert failure.metrics.model_calls == 1

    assert failure.metrics.input_tokens == 100

    assert failure.metrics.output_tokens == 50

    assert failure.metrics.total_tokens == 150

    assert isinstance(
        failure.__cause__,
        ValidationError
    )


def test_missing_token_metadata(mock_model):

    mock_model.invoke.return_value.usage_metadata = None

    result = extractor.extract_order_with_metrics(
        "Synthetic purchase order"
    )

    assert result.metrics.input_tokens is None
    assert result.metrics.output_tokens is None
    assert result.metrics.total_tokens is None



def test_model_error(mock_model):

    mock_model.invoke.side_effect = TimeoutError(
        "Simulated timeout"
    )

    with pytest.raises(ExtractionFailure) as exc_info:

        extractor.extract_order_with_metrics(
            "Synthetic purchase order"
        )

    failure = exc_info.value

    assert failure.failure_type == "model_error"

    assert failure.metrics.model_calls == 1

    assert failure.metrics.input_tokens is None

    assert failure.metrics.output_tokens is None

    assert failure.metrics.total_tokens is None

    assert failure.metrics.invocation_seconds >= 0

    assert isinstance(
        failure.__cause__,
        TimeoutError
    )

    mock_model.invoke.assert_called_once()



def test_model_reuse(monkeypatch):

    extractor.get_model.cache_clear()

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "synthetic-test-key"
    )

    factory = Mock()

    monkeypatch.setattr(
        extractor,
        "ChatOpenAI",
        factory
    )

    try:

        first = extractor.get_model()
        second = extractor.get_model()

        assert first is second

        factory.assert_called_once()

    finally:

        extractor.get_model.cache_clear()


def test_missing_api_key(monkeypatch):

    extractor.get_model.cache_clear()

    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False
    )

    factory = Mock()

    monkeypatch.setattr(
        extractor,
        "ChatOpenAI",
        factory
    )

    try:

        with pytest.raises(
            RuntimeError,
            match="OPENAI_API_KEY is missing"
        ):

            extractor.get_model()

        factory.assert_not_called()

    finally:

        extractor.get_model.cache_clear()

def test_legacy_invalid_output(mock_model):

    mock_model.invoke.return_value.text = "{}"

    with pytest.raises(ValidationError):

        extractor.extract_order(
            "Synthetic purchase order"
        )


def test_legacy_model_error(mock_model):

    mock_model.invoke.side_effect = TimeoutError(
        "Simulated timeout"
    )

    with pytest.raises(TimeoutError):

        extractor.extract_order(
            "Synthetic purchase order"
        )
