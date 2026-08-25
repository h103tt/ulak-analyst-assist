"""Shared local Ollama chat model config, used by agent.py (main ReAct loop),
refine.py (answer refinement pass), and vector_embed.py (query expansion) so
they all talk to the same served model instead of each standing up their own
separately-configured client."""

import os

from langchain_ollama import ChatOllama

# Overridable for local testing against whatever model is actually pulled
# (e.g. OLLAMA_MODEL=qwen2.5:7b) without editing this file.
DEFAULT_MODEL = "qwen3.5:9b"


def get_model() -> ChatOllama:
    kwargs = dict(
        model=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
        temperature=0.5,
        top_k=20,
        top_p=0.15,
        num_ctx=32768,
    )
    # Some GPU/driver/Ollama-CUDA-build combinations crash on load (e.g. a
    # "shared object initialization failed" / stack-buffer-overrun abort from
    # a PDL feature-support check on older GPUs). Set OLLAMA_NUM_GPU=0 to
    # force CPU-only inference as a workaround -- unset by default so this
    # doesn't change behavior on machines where GPU offload works fine.
    num_gpu = os.environ.get("OLLAMA_NUM_GPU")
    if num_gpu is not None:
        kwargs["num_gpu"] = int(num_gpu)
    return ChatOllama(**kwargs)
