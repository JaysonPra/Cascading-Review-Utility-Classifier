from collections.abc import Generator

from sqlmodel import Session, create_engine

from classifier_core.core.config import settings
from classifier_core.schemas.database import Review, SQLModel  # noqa # type: ignore

engine = create_engine(settings.database_url, echo=False)


def get_session() -> Generator[Session, None]:
    with Session(engine) as session:
        yield session


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
