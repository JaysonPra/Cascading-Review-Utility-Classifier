from typing import cast

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = cast(str, None)


settings = Settings()
