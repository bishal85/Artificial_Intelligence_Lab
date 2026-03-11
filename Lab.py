import os

# RX 6600 (gfx1032) can require this override with some ROCm wheel builds.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer,Trainer,TrainingArguments,DataCollatorForLanguageModeling


def _can_train_on_cuda():
    if not torch.cuda.is_available():
        return False
    try:
        # Exercise the same kind of kernel operation Trainer uses internally.
        probe = torch.tensor([1, -100], device="cuda")
        _ = probe.ne(-100).sum().item()
        torch.cuda.synchronize()
        return True
    except Exception as exc:
        print(f"CUDA/HIP probe failed, falling back to CPU: {exc}")
        return False


model_name="gpt2"
tokenizer=GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token
use_cuda = _can_train_on_cuda()
device = "cuda" if use_cuda else "cpu"
model=GPT2LMHeadModel.from_pretrained(model_name).to(device)
# Load Tiny Shakespeare from a plain text file (dataset scripts are deprecated).
dataset = load_dataset(
    "text",
    data_files={
        "train": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    },
)
def tokenize_function(example):
    return tokenizer(example["text"],truncation=True, max_length=128)
tokenized_dataset=dataset.map(tokenize_function,batched=True,remove_columns=["text"])
# Drop blank lines that tokenize to zero-length, which crash GPT-2 forward().
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)
# 4. Data Collator (Standard for Causal LM)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. Training Arguments (Optimized for RX 6600)
training_args = TrainingArguments(
    output_dir="./gpt2-shakespeare",
    num_train_epochs=4,             # Shakespeare is small, so 4-5 epochs works well
    per_device_train_batch_size=4,  # Adjust if you get OOM (Out of Memory)
    save_steps=10000,
    logging_steps=50,
    fp16=False,                  # Enable mixed precision only when CUDA is stable
    use_cpu=not use_cuda,
    learning_rate=5e-5,
)

# 6. Initialize and Run Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)

print(f"Starting fine-tuning on {device}...")
trainer.train()
trainer.save_model("./shakespeare_model")
