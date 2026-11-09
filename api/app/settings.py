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
    hf_token: SecretStr | None = None
    hf_llm_model: str = "Qwen/Qwen2.5-14B-Instruct-1M"
    hf_llm_provider: str = "auto"


# Create a single instance at import
settings = Settings()
