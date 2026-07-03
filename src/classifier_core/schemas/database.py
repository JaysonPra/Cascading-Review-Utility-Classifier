from sqlmodel import Field, SQLModel

from classifier_core.core.types import ReviewLabel


class Review(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    content: str = Field(min_length=0)
    score: int = Field(ge=1, le=5)
    label: ReviewLabel = Field(index=True)
