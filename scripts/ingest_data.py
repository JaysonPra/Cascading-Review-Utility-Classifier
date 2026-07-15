import pickle
from collections.abc import Generator
from typing import Any

from google_play_scraper import Sort, reviews  # type: ignore
from google_play_scraper.features.reviews import _ContinuationToken  # type: ignore
from loguru import logger

from classifier_core.core.constants import DATA_DIR, TOKEN_PATH
from classifier_core.core.crud import insert_batch_reviews
from classifier_core.core.db import get_session, init_db
from classifier_core.core.logger import setup_console_logger, setup_ingestion_logger
from classifier_core.schemas.database import Review


def save_token(token: _ContinuationToken) -> None:
    """Saves the Google Play scraping continuation token."""
    if token:
        DATA_DIR.mkdir(exist_ok=True)

        with open(TOKEN_PATH, "wb") as file:
            pickle.dump(token, file)
            logger.success("Token saved successfully...")


def read_token() -> _ContinuationToken | None:
    """Loads the serialized continuation token from disk if it exists."""
    try:
        with open(TOKEN_PATH, "rb") as file:
            continuation_token = pickle.load(file)
            logger.success("Continuation Token loaded successfully...")
            return continuation_token

    except FileNotFoundError:
        logger.info(f"Continuation Token file not found in: {str(TOKEN_PATH)}")
        return None


def start_ingestion(
    continuation_token: _ContinuationToken | None = None,
) -> tuple[dict[str, Any], _ContinuationToken]:
    """Fetch a raw review payload from the Google Play API.

    If no token is provided, the last saved token is loaded from disk
    before requesting the review.
    """
    if continuation_token is None:
        continuation_token = read_token()

    review, continuation_token = reviews(  # type: ignore
        "com.valvesoftware.android.steam.community",
        sort=Sort.NEWEST,
        continuation_token=continuation_token,  # type: ignore
        count=1,
        lang="en",
    )

    raw_payload: dict[str, Any] = review[0]  # type: ignore

    return raw_payload, continuation_token


def stream_reviews(
    initial_token: _ContinuationToken | None = None,
) -> Generator[tuple[Review, _ContinuationToken], None, None]:
    """Yield validated reviews from Google Play.

    Reviews are fetched lazily using the continuation token returned by the
    previous request. Iteration stops when no further reviews are available.
    """
    continuation_token = initial_token

    while True:
        review, continuation_token = start_ingestion(continuation_token)
        validated_review = Review.model_validate(review)

        yield validated_review, continuation_token

        if not continuation_token:
            logger.info("No more reviews...")
            break


if __name__ == "__main__":
    setup_console_logger()
    setup_ingestion_logger()
    init_db()

    batch_reviews: list[Review] = []
    initial_token: _ContinuationToken | None = read_token()
    current_token: _ContinuationToken | None = None

    with get_session() as session:
        try:
            for review, current_token in stream_reviews(initial_token):
                batch_reviews.append(review)
                logger.debug(f"Review {len(batch_reviews)}")

                if len(batch_reviews) == 100:
                    insert_batch_reviews(session, batch_reviews)
                    batch_reviews.clear()
                    logger.success("Batch of 100 reviews inserted successfully...")

        except KeyboardInterrupt:
            logger.info("Ingestion interrupted. Saving token...")

            if batch_reviews:
                insert_batch_reviews(session, batch_reviews)
                logger.success("Remaining reviews in memory inserted successfully...")

            if current_token:
                save_token(current_token)
