
import re
from dataclasses import dataclass

from pathlib import Path


class PolicyViolation(PermissionError):
    pass


ALLOWED_ACTIONS = frozenset({
    "read_document",
    "save_report"
})


@dataclass(frozen=True)
class ActionContext:
    case_id: str
    allowed_actions: frozenset[str]


def authorize_action(
    action: str,
    *,
    resource: str | None = None,
    context: ActionContext | None = None
) -> None:

    if action not in ALLOWED_ACTIONS:

        raise PolicyViolation(
            f"Action is not allowed: {action}"
        )

    if context is not None and (
        action not in context.allowed_actions or resource != context.case_id
    ):
        raise PolicyViolation("Action or resource is outside the case context")


def validate_document_id(
    document_id: str
) -> None:

    if not re.fullmatch(
        r"(?:order_[0-9]{3}|[NIBC][0-9]{2})",
        document_id
    ):

        raise PolicyViolation(
            "Invalid document identifier"
        )


def resolve_safe_path(
    base_dir: Path,
    document_id: str,
    suffix: str
) -> Path:

    validate_document_id(document_id)

    if suffix not in {".txt", ".json"}:

        raise PolicyViolation(
            "File type is not allowed"
        )

    if base_dir.is_symlink():

        raise PolicyViolation(
            "Symbolic link directory is not allowed"
        )

    base = base_dir.resolve()

    expected_name = f"{document_id}{suffix}"

    candidate = base / expected_name

    if candidate.is_symlink():

        raise PolicyViolation(
            "Symbolic link is not allowed"
        )

    resolved = candidate.resolve()

    if resolved.parent != base:

        raise PolicyViolation(
            "Path is outside the allowed directory"
        )

    if resolved.name != expected_name:

        raise PolicyViolation(
            "Unexpected resolved filename"
        )

    return resolved
