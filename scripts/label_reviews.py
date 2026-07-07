from __future__ import annotations

from pathlib import Path

import yaml
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from sqlmodel import Session

from classifier_core.core.config import settings
from classifier_core.core.crud import get_unlabeled_reviews_batch, save_review_label
from classifier_core.core.db import get_session
from classifier_core.schemas.database import Review
from classifier_core.schemas.label import ReviewBatchResponse


class LabelingJobConfig(BaseModel):
    batch_size: int = Field(default=20, ge=1)
    count: int = Field(default=5, ge=1)
    system_instruction: str

    @classmethod
    def load_from_yaml(cls, file_path: Path) -> LabelingJobConfig:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


def build_batch_prompt(review_batch: list[Review], system_instructions: str) -> str:
    serialized_reviews = "\n".join(f"{r.id}. {r.content}" for r in review_batch)

    return f"{system_instructions}\nReviews to process:\n{serialized_reviews}"


def label_review(client: genai.Client, prompt_content: str) -> str | None:
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


def start_labeling(
    session: Session,
    client: genai.Client,
    batch_size: int = 20,
    last_id: int = 0,
    prompt: str = "",
) -> tuple[str | None, int | None]:
    review_batch = get_unlabeled_reviews_batch(
        session=session, limit=batch_size, last_id=last_id
    )

    if not review_batch:
        return None, None

    batch_prompt = build_batch_prompt(review_batch, prompt)

    batch_labels = label_review(client, batch_prompt)

    return batch_labels, review_batch[-1].id  # type: ignore


if __name__ == "__main__":
    count: int = 5
    batch_size: int = 20
    prompt: str = ""
    last_id: int = 0
    client = genai.Client()

    with get_session() as session:
        for _ in range(count):
            if not last_id:
                break

            batch_labels, current_last_id = start_labeling(
                session, client, batch_size, last_id, prompt
            )

            if current_last_id is None or batch_labels is None:
                break

            last_id = current_last_id

            validated_batch = ReviewBatchResponse.model_validate(
                batch_labels
            ).batch_response

            for review in validated_batch:
                save_review_label(session, id=review.id, label=review.label)
