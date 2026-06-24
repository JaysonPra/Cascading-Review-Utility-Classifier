from sqlmodel import Session, inspect


def test_database_schema_initialization(test_engine):
    inspector = inspect(test_engine)

    assert "review" in inspector.get_table_names()


def test_session(db_session: Session):
    assert isinstance(db_session, Session)
    assert db_session.is_active
