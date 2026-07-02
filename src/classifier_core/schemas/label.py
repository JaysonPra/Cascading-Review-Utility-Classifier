from pydantic import BaseModel, Field


class ReviewLabel(BaseModel):
    label_str: str = Field(
        description="Must be either 'High Utility', 'Low Utility', or 'Spam'"
    )
    reasoning: str = Field(description="Include a one sentence reason for the label")


class ReviewBatchResponse(BaseModel):
    batch_response: list[ReviewLabel] = Field(description="List of classified reviews")
