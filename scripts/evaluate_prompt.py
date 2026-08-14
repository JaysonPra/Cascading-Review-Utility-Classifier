from collections.abc import Generator

from google import genai
from sklearn.metrics import cohen_kappa_score
from sqlmodel import Session

from classifier_core.core.constants import CONFIG_DIR
from classifier_core.core.crud import (
    get_reviews_with_llm_labels,
    get_reviews_with_manual_labels,
    save_batch_review_label,
)
from classifier_core.core.db import get_session
from classifier_core.extras.label_utils import (
    LabelingJobConfig,
    build_batch_prompt,
    label_batch_reviews,
)
from classifier_core.schemas.database import Review
from classifier_core.schemas.label import ReviewBatchResponse, ReviewLabelType


def chunk_reviews(
    reviews: list[Review],
    chunk_size: int,
) -> Generator[list[Review], None, None]:
    for i in range(0, len(reviews), chunk_size):
        yield reviews[i : i + chunk_size]


def get_label_dict(batch_reviews: ReviewBatchResponse) -> dict[int, ReviewLabelType]:
    reviews = batch_reviews.batch_response

    return {review.id: review.label for review in reviews}


def start_labeling_job(
    job_config: LabelingJobConfig, client: genai.Client, session: Session
) -> None:
    unlabeled_reviews = get_reviews_with_manual_labels(session, job_config.num_reviews)

    for batch in chunk_reviews(unlabeled_reviews, job_config.batch_size):
        batch_prompt = build_batch_prompt(batch, job_config.system_instruction)

        response = label_batch_reviews(client, batch_prompt)
        validated_response = ReviewBatchResponse.model_validate_json(response)

        label_dict = get_label_dict(validated_response)
        save_batch_review_label(session, label_dict)


def evaluate_label(session: Session) -> float:
    labeled_reviews = get_reviews_with_llm_labels(session)

    manual_labels = [review.manual_label.value for review in labeled_reviews]  # type: ignore
    llm_labels = [review.label.value for review in labeled_reviews]  # type: ignore

    all_labels = [e.value for e in ReviewLabelType]

    return cohen_kappa_score(manual_labels, llm_labels, labels=all_labels)


if __name__ == "__main__":
    file_path = CONFIG_DIR / "labeling_job.yaml"
    system_instructions = LabelingJobConfig.load_from_yaml(file_path)

    client = genai.Client()

    with get_session() as session:
        # start_labeling_job(system_instructions, client, session)
        score = evaluate_label(session)
