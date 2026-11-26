
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class TestState(TypedDict):
    counter: int
    approved: bool


def increment(state: TestState):
    counter = state["counter"] + 1

    print(f"Iteration: {counter}")

    return {"counter": counter}


def check_counter(state: TestState):
    if state["counter"] < 3:
        return "increment"

    return "human_approval"


def human_approval(state: TestState):
    decision = interrupt({
        "message": "Approve the operation?",
        "counter": state["counter"]
    })

    return {"approved": decision}


def check_approval(state: TestState):
    if state["approved"]:
        return "approved"

    return "rejected"


def approved(state: TestState):
    print("Operation approved.")

    return {}


def rejected(state: TestState):
    print("Operation rejected.")

    return {}


builder = StateGraph(TestState)

builder.add_node("increment", increment)
builder.add_node("human_approval", human_approval)
builder.add_node("approved", approved)
builder.add_node("rejected", rejected)

builder.add_edge(START, "increment")

builder.add_conditional_edges(
    "increment",
    check_counter,
    {
        "increment": "increment",
        "human_approval": "human_approval"
    }
)

builder.add_conditional_edges(
    "human_approval",
    check_approval,
    {
        "approved": "approved",
        "rejected": "rejected"
    }
)

builder.add_edge("approved", END)
builder.add_edge("rejected", END)

graph = builder.compile(
    checkpointer=InMemorySaver()
)

config = {
    "configurable": {
        "thread_id": "environment-test-001"
    }
}

print("Starting LangGraph smoke test...")

result = graph.invoke(
    {
        "counter": 0,
        "approved": False
    },
    config=config
)

assert result["counter"] == 3
assert "__interrupt__" in result

print("\nGraph paused successfully.")
print(result["__interrupt__"])

while True:
    answer = input("\nApprove? (y/n): ").strip().lower()

    if answer in ("y", "n"):
        break

    print("Please enter y or n.")

decision = answer == "y"

final_result = graph.invoke(
    Command(resume=decision),
    config=config
)

assert final_result["counter"] == 3
assert final_result["approved"] == decision
assert not graph.get_state(config).next

print("\nFinal state:", final_result)
print("LangGraph smoke test PASSED.")
