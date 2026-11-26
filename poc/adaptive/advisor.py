from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda

from poc.target import extractor

from .advisor_prompts import AdvisorPromptVariant, apply_advisor_prompt
from .feedback_loop import InvestigationProposal


def create_advisor(
    *, prompt_variant: AdvisorPromptVariant = "basic"
) -> Runnable[list[BaseMessage], InvestigationProposal]:
    """Bind the configured model only when the graph requests a proposal."""
    if prompt_variant not in ("basic", "evidence_driven"):
        raise ValueError("Unknown advisor prompt variant")

    def propose(messages: list[BaseMessage]) -> InvestigationProposal:
        prompt = apply_advisor_prompt(messages, prompt_variant)
        advisor = extractor.get_model().with_structured_output(
            InvestigationProposal, method="json_schema", strict=True
        )
        return InvestigationProposal.model_validate(advisor.invoke(prompt))

    return RunnableLambda(propose)
