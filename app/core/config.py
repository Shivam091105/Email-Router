"""
Centralized application configuration.

Every setting the app needs (database URL, log level, API keys added in
later phases) is declared here as a typed field on a single Settings class,
and loaded from environment variables / a .env file. No other module should
read os.environ directly — they should import `settings` from here instead.

Why this pattern:
- Single source of truth for configuration.
- Pydantic validates types at startup (e.g. a malformed DATABASE_URL fails
  fast with a clear error, instead of surfacing as a confusing error deep
  inside SQLAlchemy later).
- Swapping environments (dev/test/prod) is just swapping the .env file —
  no code changes.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Application ---
    app_env: str = "development"
    log_level: str = "INFO"

    # --- Database ---
    database_url: str = "postgresql://postgres:postgres@localhost:5432/email_router"

    # --- LLM / embeddings ---
    # Groq is the default LLM provider: free tier, no credit card required.
    groq_api_key: str | None = None
    llm_model_name: str = "openai/gpt-oss-20b"

    # Embeddings run locally by default (sentence-transformers, no API key
    # needed). huggingfacehub_api_token is only used if you explicitly
    # switch to HuggingFaceInferenceEmbeddings/HuggingFaceLLMClient.
    huggingfacehub_api_token: str | None = None
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_persist_dir: str = "chroma_db"
    retrieval_top_k: int = 3
    confidence_threshold: float = 0.80

    # --- Email integration ---
    imap_host: str | None = None
    imap_user: str | None = None
    imap_password: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    max_attachment_size_mb: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# A single, importable instance. Created once at import time.
settings = Settings()