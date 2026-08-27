import argparse
from collections.abc import Generator

import mlflow
from google import genai
from sklearn.metrics import cohen_kappa_score
from sqlmodel import Session

from classifier_core.core.constants import CONFIG_DIR
from classifier_core.core.crud import (
    get_reviews_with_llm_labels,
    get_reviews_with_manual_labels,
    get_unlabeled_reviews,
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
    """Yields batches of reviews of a specified chunk size."""
    for i in range(0, len(reviews), chunk_size):
        yield reviews[i : i + chunk_size]


def get_label_dict(batch_reviews: ReviewBatchResponse) -> dict[int, ReviewLabelType]:
    """Maps review IDs to their predicted LLM labels."""
    reviews = batch_reviews.batch_response

    return {review.id: review.label for review in reviews}


def start_labeling_job(
    job_config: LabelingJobConfig,
    reviews: list[Review],
    client: genai.Client,
    session: Session,
) -> None:
    """Executes the batch LLM labeling pipeline for manually annotated reviews."""
    for batch in chunk_reviews(reviews, job_config.batch_size):
        batch_prompt = build_batch_prompt(batch, job_config.system_instruction)

        response = label_batch_reviews(client, batch_prompt)
        validated_response = ReviewBatchResponse.model_validate_json(response)

        label_dict = get_label_dict(validated_response)
        save_batch_review_label(session, label_dict)


def evaluate_label(session: Session) -> float:
    """Calculates Cohen's Kappa score between manual and LLM labels."""
    labeled_reviews = get_reviews_with_llm_labels(session)

    manual_labels = [review.manual_label.value for review in labeled_reviews]  # type: ignore
    llm_labels = [review.label.value for review in labeled_reviews]  # type: ignore

    all_labels = [e.value for e in ReviewLabelType]

    return cohen_kappa_score(manual_labels, llm_labels, labels=all_labels)


def run_experiment(
    system_instructions: LabelingJobConfig,
    reviews: list[Review],
    client: genai.Client,
    session: Session,
) -> None:
    """Runs a system prompt evaluation experiment and logs metrics to MLflow."""
    mlflow.set_experiment(experiment_name="System Prompt Evaluation")  # type: ignore

    instructions_dict: dict[str, int | str] = {
        "batch_size": system_instructions.batch_size,
        "num_reviews": system_instructions.num_reviews,
        "system_prompt": system_instructions.system_instruction,
    }

    with mlflow.start_run():
        start_labeling_job(system_instructions, reviews, client, session)
        score = evaluate_label(session)

        mlflow.log_params(instructions_dict)
        mlflow.log_metric("cohen_kappa_score", score)


if __name__ == "__main__":
    file_path = CONFIG_DIR / "labeling_job.yaml"
    system_instructions = LabelingJobConfig.load_from_yaml(file_path)
    client = genai.Client()

    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    with get_session() as session:
        if args.apply:
            reviews = get_unlabeled_reviews(session)
            start_labeling_job(system_instructions, reviews, client, session)
        else:
            reviews = get_reviews_with_manual_labels(
                session, limit=system_instructions.num_reviews
            )
            run_experiment(system_instructions, reviews, client, session)
