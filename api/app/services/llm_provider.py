import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-1M")
HF_LLM_PROVIDER = os.getenv("HF_LLM_PROVIDER", "auto")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set in the environment.")

client = InferenceClient(provider=HF_LLM_PROVIDER, api_key=HF_TOKEN)

def generate_with_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Hosted LLM generation through Hugging Face Inference Providers.
    Keeping the old function name avoids changing rag.py right away.
    """
    completion = client.chat.completions.create(
        model=HF_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
        temperature=0.0,
    )

    return completion.choices[0].message.content.strip()