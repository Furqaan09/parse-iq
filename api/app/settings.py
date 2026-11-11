from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# -----------------
# Important paths
# -----------------
API_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = API_DIR / "storage"
STATIC_DIR = API_DIR / "static"  # built React app

ENV_FILE = API_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    database_url: str
    openai_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-5.6-luna"


# Create a single instance at import
settings = Settings()
