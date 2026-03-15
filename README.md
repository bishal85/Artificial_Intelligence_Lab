# Artificial Intelligence Lab Projects

This repository contains two local AI workflows:

1. `Lab.py`: fine-tunes GPT-2 on the Tiny Shakespeare dataset.
2. `test.py`: compares text generation from the fine-tuned model against base GPT-2.
3. `Lab2.py`: expands an image prompt with LangChain + GPT-2 and generates an image with Stable Diffusion.

The code was developed in a local Python virtual environment and tested on an AMD RX 6600 system with ROCm-compatible PyTorch.

## Project Structure

```text
.
├── Lab.py                     # GPT-2 fine-tuning on Tiny Shakespeare
├── test.py                    # Fine-tuned vs base GPT-2 comparison
├── Lab2.py                    # Prompt expansion + Stable Diffusion image generation
├── README.md
├── requirements.txt           # Python packages used by the scripts (install torch separately)
├── Screenshots/
│   ├── .gitkeep
│   └── README.md
├── gpt2-shakespeare/          # Training checkpoints (generated, ignored)
├── shakespeare_model/         # Saved fine-tuned model (generated, ignored)
└── task2_output.png           # Generated image output from Lab2.py (ignored)
```

## Features

- Fine-tunes `gpt2` on Tiny Shakespeare text.
- Saves the trained model locally for reuse.
- Generates Shakespeare-style text from the fine-tuned model.
- Compares the fine-tuned model with base GPT-2 using the same prompt and seed.
- Uses LangChain Expression Language to expand a simple image topic into a richer prompt.
- Generates an image from the expanded prompt using `runwayml/stable-diffusion-v1-5`.
- Detects whether the GPU is actually usable and falls back to CPU when needed.

## Environment Notes

This project uses Hugging Face models and will download model weights the first time each script runs.

For AMD/ROCm systems:

- `Lab.py`, `test.py`, and `Lab2.py` set `HSA_OVERRIDE_GFX_VERSION=10.3.0` for RX 6600 compatibility with some wheel builds.
- `Lab2.py` should not be used with a CUDA-only `xformers` wheel in a ROCm PyTorch environment.
- If you see an error mentioning `libcudart.so.12` or `xformers`, remove `xformers` from the environment:

```bash
python -m pip uninstall -y xformers
```

## Setup

Create and activate a virtual environment, then install PyTorch for your hardware platform first.

Install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

If you are on ROCm, install the ROCm build of PyTorch before installing the rest of the packages.

## How To Run

### 1. Fine-Tune GPT-2

```bash
python Lab.py
```

What it does:

- Loads Tiny Shakespeare from a text source.
- Tokenizes the dataset.
- Fine-tunes `gpt2`.
- Saves the trained model to `./shakespeare_model`.

### 2. Compare Fine-Tuned vs Base GPT-2

```bash
python test.py
```

What it does:

- Loads `./shakespeare_model`.
- Loads base `gpt2`.
- Uses the same prompt and seed for both models.
- Prints both generated outputs for direct comparison.

### 3. Generate an Image with Stable Diffusion

```bash
python Lab2.py
```

What it does:

- Uses GPT-2 through LangChain to expand a simple topic into a richer image prompt.
- Loads the Stable Diffusion v1.5 pipeline.
- Generates an image from the expanded prompt.
- Saves the result as `task2_output.png`.

If you want to force CPU execution:

```bash
FORCE_CPU=1 python Lab2.py
```

## Sample Text Output

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

## GitHub Upload

This repository already has a Git remote configured:

`origin -> https://github.com/bishal85/Artificial_Intelligence_Lab.git`

Typical commands to publish the latest changes:

```bash
git add README.md .gitignore requirements.txt Lab2.py
git commit -m "Add project documentation and Lab2 ROCm notes"
git push origin main
```

## Notes

- Generated model folders and local virtual environment files are ignored and should not be committed.
- `task2_output.png` is treated as a generated artifact and is also ignored by default.
- The first run of each script may take time because Hugging Face assets need to download.
