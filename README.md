# Shakespeare Fine-Tuning (GPT-2)

This repository fine-tunes GPT-2 on Tiny Shakespeare and compares generated text against base GPT-2.

## Repository Structure

```text
.
├── Lab.py                    # Training script
├── test.py                   # Inference + comparison script (fine-tuned vs base GPT-2)
├── README.md
├── Screenshots/
│   ├── .gitkeep
│   └── README.md
├── gpt2-shakespeare/         # Training checkpoints (generated)
└── shakespeare_model/        # Saved fine-tuned model (generated)
```

## How To Run

### 1. Train

```bash
python3 Lab.py
```

This saves the fine-tuned model to `./shakespeare_model`.

### 2. Compare Base vs Fine-Tuned

```bash
python3 test.py
```

`test.py` will:
- Load `./shakespeare_model`
- Load base `gpt2`
- Generate from both using the same prompt and seed
- Print both outputs for direct comparison

## Before vs After (Sample Output)

Prompt:

```text
ROMEO: Shall I speak more, or shall I hear this?
```

Fine-tuned (`shakespeare_model`):

```text
ROMEO: Shall I speak more, or shall I hear this? for you know, the prince is come. What, my lord, come hither:--ROMIO: my son, he's here,--O, mercy!--BENIO! you are gone, you must! I am: here! O, what, that!!--ROM: I, I! Romeo! come! his son! what! he is gone! it is in his
```

Base (`gpt2`):

```text
ROMEO: Shall I speak more, or shall I hear this?

JOSEPH: Oh, I have no choice, Mr. Mather, but to begin with, let me tell you the truth. As you know, my father had died in the battle of Waterloo, and he died a little while ago. He was buried at the Royal Cemetery in London. In that cemetery he had the name of John Cusack, who was in his
```


