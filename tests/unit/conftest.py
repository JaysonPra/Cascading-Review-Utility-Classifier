import pathlib

import pytest
from sqlmodel import Session, SQLModel, create_engine

import classifier_core.core.db as db_module


@pytest.fixture(scope="function")
def test_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    test_url = "sqlite:///:memory:"

    local_test_engine = create_engine(
        test_url, echo=False, connect_args={"check_same_thread": False}
    )

    monkeypatch.setattr(db_module, "engine", local_test_engine)

    SQLModel.metadata.create_all(local_test_engine)

    yield local_test_engine


@pytest.fixture(scope="function")
def db_session(test_engine):  # type: ignore
    with Session(test_engine) as session:  # type: ignore
        yield session
