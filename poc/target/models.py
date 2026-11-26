from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PurchaseOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(min_length=1)
    supplier: str = Field(min_length=1)

    amount: Decimal = Field(
        gt=0,
        allow_inf_nan=False,
        decimal_places=2
    )

    currency: Literal["USD", "EUR", "ILS"]

    delivery_date: date