from sqlmodel import Session, select

from classifier_core.core.crud import insert_batch_reviews
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
