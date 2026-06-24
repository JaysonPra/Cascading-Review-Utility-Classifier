from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///fallback_development.db"


settings = Settings()
