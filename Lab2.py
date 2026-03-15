import os
import sys

import torch

# AMD RX 6600 GPU Override
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")


def _load_image_dependencies():
    try:
        from diffusers import StableDiffusionPipeline
    except Exception as exc:
        extra_help = ""
        error_text = str(exc)
        if "libcudart.so.12" in error_text or "xformers" in error_text.lower():
            extra_help = (
                "\nDetected an incompatible xformers install for this PyTorch build. "
                f"Remove it with: {sys.executable} -m pip uninstall -y xformers"
            )
        raise SystemExit(
            "Stable Diffusion dependencies are missing or incompatible. "
            "Run: python -m pip install -U diffusers transformers accelerate\n"
            f"{extra_help}\n"
            f"Original error: {exc}"
        ) from exc
    return StableDiffusionPipeline


def _load_langchain_dependencies():
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_community.llms import HuggingFacePipeline
    except Exception as exc:
        raise SystemExit(
            "LangChain dependencies are missing. "
            "Run: python -m pip install -U langchain-core langchain-community\n"
            f"Original error: {exc}"
        ) from exc
    return ChatPromptTemplate, HuggingFacePipeline, StrOutputParser


def _pick_device():
    if os.environ.get("FORCE_CPU", "0") == "1":
        return "cpu", -1, torch.float32

    if not torch.cuda.is_available():
        return "cpu", -1, torch.float32

    try:
        probe = torch.tensor([1], device="cuda")
        _ = probe.item()
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"GPU probe failed, falling back to CPU: {exc}")
        return "cpu", -1, torch.float32

    return "cuda", 0, torch.float16


def main():
    StableDiffusionPipeline = _load_image_dependencies()
    ChatPromptTemplate, HuggingFacePipeline, StrOutputParser = (
        _load_langchain_dependencies()
    )

    device, pipeline_device, diffusion_dtype = _pick_device()

    # 1. Setup Local LLM for Prompt Expansion
    model_id = "gpt2"
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 50},
        device=pipeline_device,
    )

    # 2. Setup the LCEL Chain
    prompt = ChatPromptTemplate.from_template(
        "Enhance this image prompt for Stable Diffusion: {topic}. "
        "Include lighting, style, and 8k details. Enhanced Prompt:"
    )
    expansion_chain = prompt | llm | StrOutputParser()

    # 3. Stable Diffusion Engine
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=diffusion_dtype,
    ).to(device)
    pipe.enable_attention_slicing()

    # 4. Execution
    user_input = "Nature"
    print("Expanding prompt...")
    enhanced_prompt = expansion_chain.invoke({"topic": user_input}).strip()

    print(f"Generating image on {device} for: {enhanced_prompt}")
    image = pipe(enhanced_prompt).images[0]
    image.save("task2_output.png")
    print("Saved task2_output.png")


if __name__ == "__main__":
    main()
