from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from classifier_core.core.types import ReviewLabelType
from classifier_core.schemas.database import Review


def insert_batch_reviews(session: Session, reviews: list[Review]) -> None:
    """Bulk inserts a list of Review records into the database with transaction rollback on failure."""
    if len(reviews) == 0:
        logger.info("No reviews to add...")
        return

    try:
        session.add_all(reviews)
        session.commit()

        logger.success("Session committed succesfully...")

    except SQLAlchemyError as e:
        logger.exception(f"Session committed unsuccessfully: {e}")

        session.rollback()


def get_reviews_with_manual_labels(session: Session, limit: int = 100) -> list[Review]:
    """Fetches a chronologically ordered batch of reviews with manual labels."""
    statement = select(Review).where(Review.manual_label is not None).limit(limit)

    try:
        return list(session.exec(statement).all())
    except SQLAlchemyError:
        logger.exception("Failed to fetch reviews batch from database")
        return []


def save_batch_review_label(
    session: Session, updates: dict[int, ReviewLabelType]
) -> None:
    """Saves the labels of a batch of reviews"""
    if not updates:
        return

    try:
        statement = select(Review).where(Review.id in updates.keys())
        reviews = session.exec(statement).all()

        for review in reviews:
            review.label = updates[review.id]  # type: ignore

        session.commit()

    except SQLAlchemyError:
        session.rollback()
        logger.exception("Failed to batch update review labels in the database")
