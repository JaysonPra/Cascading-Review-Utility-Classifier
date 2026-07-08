from pathlib import Path

import yaml
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from classifier_core.core.config import settings
from classifier_core.schemas.database import Review
from classifier_core.schemas.label import ReviewBatchResponse


class LabelingJobConfig(BaseModel):
    batch_size: int = Field(default=20, ge=1)
    num_reviews: int = Field(default=100, ge=1)
    system_instruction: str

    @classmethod
    def load_from_yaml(cls, file_path: Path) -> LabelingJobConfig:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


def build_batch_prompt(review_batch: list[Review], system_instructions: str) -> str:
    serialized_reviews = "\n".join(f"{r.id}. {r.content}" for r in review_batch)

    return f"{system_instructions}\nReviews to process:\n{serialized_reviews}"


def label_batch_reviews(client: genai.Client, prompt_content: str) -> str | None:
    response = client.models.generate_content(
        contents=prompt_content,
        model=settings.label_model,
        config=types.GenerateContentConfig(
            response_schema=ReviewBatchResponse,
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    return response.text
