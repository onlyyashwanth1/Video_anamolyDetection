import torch
from transformers import AutoProcessor, AutoModelForCausalLM

model_path = "/home/nverma/models/gemma-4-31B-it"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_path)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
)

print("Model loaded on:", next(model.parameters()).device)
print("Type 'quit' to exit.\n")

while True:
    q = input("Prompt: ")
    if q.strip().lower() == "quit":
        break

    messages = [{"role": "user", "content": q}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=150)

    # only decode the newly generated tokens, not the prompt
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(new_tokens, skip_special_tokens=True)

    print("\n=== RESPONSE ===")
    print(response)
    print()