from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator
)

from poc.target.models import PurchaseOrder


Category = Literal[
    "normal",
    "injection",
    "benign_suspicious",
    "complex"
]

DatasetSplit = Literal[
    "development",
    "held_out"
]

AllowedAction = Literal[
    "read_document",
    "save_report"
]


class TestCase(BaseModel):

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(
        pattern=r"^[NIBC]\d{2}$"
    )

    category: Category

    split: DatasetSplit

    description: str = Field(min_length=1)

    document: str = Field(min_length=1)

    expected_output: PurchaseOrder

    allowed_actions: list[AllowedAction] = Field(
        default_factory=lambda: [
            "read_document",
            "save_report"
        ]
    )

    contains_injection: bool = False

    @model_validator(mode="after")
    def validate_category(self):

        prefixes = {
            "normal": "N",
            "injection": "I",
            "benign_suspicious": "B",
            "complex": "C"
        }

        expected_prefix = prefixes[self.category]

        if not self.case_id.startswith(expected_prefix):
            raise ValueError(
                "Case ID does not match category"
            )

        if self.contains_injection != (
            self.category == "injection"
        ):
            raise ValueError(
                "Injection flag does not match category"
            )

        return self