from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    app_name: str = "CV Supporter"
    database_url: str = ""
    database_host: str = ""
    database_password: str = ""
    ollama_address: str = ""
    ollama_host: str = "http://localhost:11434"
    chat_model: str = "qwen3:1.7b"
    embedding_model: str = "nomic-embed-text"
    chat_context_tokens: int = 4096
    chat_max_output_tokens: int = 600
    ollama_timeout_seconds: int = 300
    session_ttl_seconds: int = 1800
    cleanup_interval_seconds: int = 5
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    enable_study_routes: bool = False

    @model_validator(mode="after")
    def resolve_service_addresses(self):
        if not self.database_url and self.database_host:
            self.database_url = URL.create(
                "postgresql+psycopg", username="rag_user", password=self.database_password,
                host=self.database_host, port=5432, database="rag_db",
            ).render_as_string(hide_password=False)
        if self.ollama_address:
            self.ollama_host = "http://" + self.ollama_address
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
