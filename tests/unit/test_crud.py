from sqlmodel import Session, select

from classifier_core.core.crud import (
    get_review,
    insert_batch_reviews,
    save_review_label,
)
from classifier_core.core.types import ReviewLabelType
from classifier_core.schemas.database import Review


def test_insert_batch_reviews(db_session: Session):
    test_reviews = [
        Review(
            content="Good",
            score=5,
        ),
        Review(
            content="Bad",
            score=1,
        ),
    ]

    insert_batch_reviews(session=db_session, reviews=test_reviews)

    results = db_session.exec(select(Review)).all()

    assert len(results) == 2
    assert results[0].content == "Good"


def test_insert_batch_reviews_rollback_on_error(db_session: Session):
    duplicate_reviews = [
        Review(id=1, content="Good", score=5),
        Review(id=1, content="Bad", score=1),
    ]

    insert_batch_reviews(db_session, duplicate_reviews)

    results = db_session.exec(select(Review)).all()
    assert len(results) == 0


def test_get_review(db_session: Session):
    test_review = Review(id=12, content="Very nice product!", score=4)
    db_session.add(test_review)
    db_session.commit()

    fetched_review = get_review(db_session, id=12)

    assert fetched_review is not None
    assert fetched_review.id == 42
    assert fetched_review.content == "Very nice product!"


def test_get_review_not_found(db_session: Session):
    fetched_review = get_review(db_session, id=14)

    assert fetched_review is None


def test_save_review_label(db_session: Session):
    test_review = Review(id=42, content="Spam message here", score=1)
    db_session.add(test_review)
    db_session.commit()

    save_review_label(db_session, id=42, label=ReviewLabelType.SPAM)

    updated_review = db_session.exec(select(Review).where(Review.id == 42)).one()
    assert updated_review.label == "Spam"


def test_save_review_label_non_existent(db_session: Session):
    save_review_label(db_session, id=24, label=ReviewLabelType.HIGH_UTILITY)

    results = db_session.exec(select(Review)).all()
    assert len(results) == 0
