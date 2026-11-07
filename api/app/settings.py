from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Path to the .env file, relative to this file
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    database_url: str
    hf_token: SecretStr | None = None
    hf_llm_model: str = "Qwen/Qwen2.5-14B-Instruct-1M"
    hf_llm_provider: str = "auto"


# Create a single instance at import
settings = Settings()
