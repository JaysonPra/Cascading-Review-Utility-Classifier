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


def get_unlabeled_manual_reviews(session: Session, limit: int = 100) -> list[Review]:
    """Fetches a chronologically ordered batch of unlabeled manual labels."""
    statement = select(Review).where(Review.manual_label is not None).limit(limit)

    try:
        return list(session.exec(statement).all())
    except SQLAlchemyError:
        logger.exception("Failed to fetch reviews batch from database")
        return []


def get_review(session: Session, id: int) -> Review | None:
    """Fetches a single review for the specified review ID"""
    statement = select(Review).where(Review.id == id)

    try:
        return session.exec(statement).one()
    except SQLAlchemyError:
        logger.exception("Failed to fetch review from database...")


def save_review_label(session: Session, id: int, label: ReviewLabelType) -> None:
    """Saves the label of the review for the specified review ID"""
    review = get_review(session, id)

    if not review:
        return

    try:
        review.label = label
        session.commit()

    except SQLAlchemyError:
        session.rollback()
        logger.exception("Failed to fetch review from database")
