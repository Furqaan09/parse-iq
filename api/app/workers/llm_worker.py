import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
device = "mps"

def main():
    """
    This process is called by the main server process to handle LLM inference.
    Worker process that reads a prompt from stdin, generates a response using the LLM,
    and writes the response to stdout.
    """
    payload = json.loads(sys.stdin.read())
    system_prompt = payload["system_prompt"]
    user_prompt = payload["user_prompt"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=True,
    )
    model.to(device)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=384,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs.shape[-1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    print(json.dumps({"answer": answer}))

if __name__ == "__main__":
    main()