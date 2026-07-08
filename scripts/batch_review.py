from __future__ import annotations

from pathlib import Path

from google import genai
from sqlmodel import Session

from classifier_core.core.crud import get_unlabeled_reviews_batch, save_review_label
from classifier_core.core.db import get_session
from classifier_core.extras.label_utils import (
    LabelingJobConfig,
    build_batch_prompt,
    label_review,
)
from classifier_core.schemas.label import ReviewBatchResponse


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


def run_labeling_pipeline(config_path: Path) -> None:
    job_config = LabelingJobConfig.load_from_yaml(config_path)
    client = genai.Client()

    last_id: int | None = 0

    with get_session() as session:
        for _ in range(job_config.count):
            batch_labels, current_last_id = start_labeling(
                session,
                client,
                job_config.batch_size,
                last_id,
                job_config.system_instruction,
            )

            if current_last_id is None or batch_labels is None:
                break

            validated_batch = ReviewBatchResponse.model_validate(
                batch_labels
            ).batch_response

            for review in validated_batch:
                save_review_label(session, review.id, review.label)

            last_id = current_last_id
