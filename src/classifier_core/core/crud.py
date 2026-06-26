from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session

from classifier_core.schemas.database import Review


def insert_batch_reviews(session: Session, reviews: list[Review]) -> None:
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
