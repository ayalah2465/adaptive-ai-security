
from dataclasses import dataclass, field
from typing import Literal


ActionStatus = Literal[
    "requested",
    "executed",
    "blocked",
    "failed"
]


@dataclass(frozen=True)
class ActionEvent:

    action: str
    status: ActionStatus
    resource: str


@dataclass
class ActionLog:

    events: list[ActionEvent] = field(
        default_factory=list
    )

    def record(
        self,
        action: str,
        status: ActionStatus,
        resource: str
    ) -> None:

        self.events.append(
            ActionEvent(
                action=action,
                status=status,
                resource=resource
            )
        )

    def count(
        self,
        status: ActionStatus
    ) -> int:

        return sum(
            event.status == status
            for event in self.events
        )
