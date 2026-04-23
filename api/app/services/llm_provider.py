import json
import subprocess
import sys
from pathlib import Path


def generate_with_local_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Runs the local LLM in a separate Python process.
    This avoids PyTorch/Transformers crashing inside the FastAPI server process.
    """
    script_path = Path(__file__).resolve().parents[1] / "workers" / "llm_worker.py"

    payload = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }

    result = subprocess.run(
        [sys.executable, str(script_path)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"LLM worker failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM worker returned invalid JSON.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        ) from e

    return data["answer"]
