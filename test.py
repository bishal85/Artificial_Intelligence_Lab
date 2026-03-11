import os
from pathlib import Path

# RX 6600 (gfx1032) may need this for some ROCm wheel builds.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    # Default to ROCm/CUDA when available; allow manual CPU override.
    if os.environ.get("FORCE_CPU", "0") == "1":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_ref: str | Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModelForCausalLM.from_pretrained(model_ref).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def generate_once(model, tokenizer, prompt: str, device: str, seed: int) -> str:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


model_path = Path("./shakespeare_model")
if not model_path.exists():
    raise FileNotFoundError(f"Model directory not found: {model_path.resolve()}")

device = pick_device()
prompt = "ROMEO: Shall I speak more, or shall I hear this?"
seed = 42

print(f"Using device: {device}", flush=True)
print("Loading fine-tuned model...", flush=True)
ft_tokenizer, ft_model = load_model_and_tokenizer(model_path, device)
print("Loading base model (gpt2)...", flush=True)
base_tokenizer, base_model = load_model_and_tokenizer("gpt2", device)

print("Generating with fine-tuned model...", flush=True)
ft_text = generate_once(ft_model, ft_tokenizer, prompt, device, seed)
print("Generating with base gpt2...", flush=True)
base_text = generate_once(base_model, base_tokenizer, prompt, device, seed)

print("\n--- Fine-tuned (shakespeare_model) ---")
print(ft_text)
print("\n--- Base (gpt2) ---")
print(base_text)
