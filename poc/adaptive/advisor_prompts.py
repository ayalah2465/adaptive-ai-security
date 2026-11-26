from typing import Literal

from langchain_core.messages import BaseMessage, SystemMessage


AdvisorPromptVariant = Literal["basic", "evidence_driven"]

EVIDENCE_DRIVEN_GUIDANCE = (
    "Tie the hypothesis to a specific observation from the supplied evidence. "
    "If the evidence establishes only a symptom, state what remains uncertain; "
    "do not infer a specific field, document content, or expected value that was not supplied. "
    "Choose the smallest set of existing diagnostic cases that can distinguish plausible "
    "explanations within the remaining budget. If a prior attempt exists, explain what "
    "the new proposal changes and how its result would support or weaken the hypothesis."
)


def apply_advisor_prompt(
    messages: list[BaseMessage], variant: AdvisorPromptVariant
) -> list[BaseMessage]:
    if variant == "basic":
        return messages

    if not messages or not isinstance(messages[0], SystemMessage):
        raise ValueError("The advisor requires a leading system message")
    instruction = messages[0]
    if not isinstance(instruction.content, str):
        raise ValueError("The advisor system message must contain text")

    return [
        instruction.model_copy(update={
            "content": f"{instruction.content}\n\n{EVIDENCE_DRIVEN_GUIDANCE}"
        }),
        *messages[1:]
    ]
