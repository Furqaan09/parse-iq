from huggingface_hub import InferenceClient

from app.settings import settings


def generate_with_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Hosted LLM generation through Hugging Face Inference Providers.
    Keeping the old function name avoids changing rag.py right away.
    """
    # Check if the HF_TOKEN is set in the environment
    if settings.hf_token is None:
        raise RuntimeError("HF_TOKEN is not set (env or api/.env).")

    # Initialize the InferenceClient with the HF API key and provider
    client = InferenceClient(
        provider=settings.hf_llm_provider, api_key=settings.hf_token.get_secret_value()
    )

    # Generate a response
    completion = client.chat.completions.create(
        model=settings.hf_llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
        temperature=0.0,
    )

    return completion.choices[0].message.content.strip()
