from pydantic import BaseModel, Field

from classifier_core.core.types import ReviewLabelType


class ReviewLabel(BaseModel):
    id: int = Field(description="ID of the Review being labeled")
    label: ReviewLabelType = Field(
        description="The category classification for the review content"
    )
    reasoning: str = Field(description="Include a one sentence reason for the label")


class ReviewBatchResponse(BaseModel):
    batch_response: list[ReviewLabel] = Field(description="List of classified reviews")
