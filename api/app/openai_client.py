from functools import cache

from openai import OpenAI

from app.settings import settings


# -------------------------------------------------
# OpenAI client with cache for embeddings and LLM
# -------------------------------------------------
@cache
def openai_client() -> OpenAI:
    if settings.openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=settings.openai_api_key.get_secret_value(), timeout=30.0)
