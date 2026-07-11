from sqlmodel import Session, select

from classifier_core.core.crud import (
    get_batch_reviews,
    get_reviews_with_manual_labels,
    insert_batch_reviews,
    save_batch_review_label,
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


def test_get_reviews_with_manual_labels(db_session: Session):
    mixed_reviews = [
        Review(content="Great", score=5, manual_label=ReviewLabelType.HIGH_UTILITY),
        Review(content="Okay", score=3, manual_label=None),
        Review(content="Terrible", score=1, manual_label=ReviewLabelType.LOW_UTILITY),
    ]
    db_session.add_all(mixed_reviews)
    db_session.commit()

    results = get_reviews_with_manual_labels(db_session, limit=10)

    assert len(results) == 2
    assert any(r.content == "Great" for r in results)
    assert any(r.content == "Terrible" for r in results)
    assert not any(r.content == "Okay" for r in results)


def test_get_batch_reviews(db_session: Session):
    batch = [Review(content=f"Review {i}", score=3) for i in range(5)]
    db_session.add_all(batch)
    db_session.commit()

    results = get_batch_reviews(db_session, limit=3)

    assert len(results) == 3


def test_save_batch_review_label(db_session: Session):
    initial_reviews = [
        Review(content="Loved it", score=5),
        Review(content="Hated it", score=1),
    ]
    db_session.add_all(initial_reviews)
    db_session.commit()

    updates = {
        initial_reviews[0].id: ReviewLabelType.HIGH_UTILITY,
        initial_reviews[1].id: ReviewLabelType.LOW_UTILITY,
    }

    save_batch_review_label(db_session, updates)  # type: ignore

    db_session.expire_all()

    updated_10 = db_session.get(Review, initial_reviews[0].id)
    updated_11 = db_session.get(Review, initial_reviews[1].id)

    assert updated_10 is not None
    assert updated_11 is not None

    assert updated_10.label == ReviewLabelType.HIGH_UTILITY
    assert updated_11.label == ReviewLabelType.LOW_UTILITY
