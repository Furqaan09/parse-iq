from pathlib import Path
from typing import Literal, Self

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# -----------------
# Important paths
# -----------------
API_DIR = Path(__file__).resolve().parents[1]
UPLOADS_DIR = API_DIR / "storage" / "uploads"  # local storage backend only
STATIC_DIR = API_DIR / "static"  # built React app

ENV_FILE = API_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    database_url: str
    openai_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-5.6-luna"

    storage_backend: Literal["local", "s3"] = "local"
    s3_bucket: str | None = None
    aws_region: str | None = None

    @model_validator(mode="after")
    def _s3_needs_bucket_and_region(self) -> Self:
        if self.storage_backend == "s3":
            missing = [n.upper() for n in ("s3_bucket", "aws_region") if not getattr(self, n)]
            if missing:
                raise ValueError(f"STORAGE_BACKEND=s3 also needs {', '.join(missing)}")
        return self


# Create a single instance at import
settings = Settings()
