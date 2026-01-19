"""
llm_vllm.py – vLLM client (OpenAI-compatible)
=============================================
* Connects to local vLLM server (multi-GPU backend)
* Replaces HF Transformers inference
* Compatible with query_server interface
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI

# -------------------------------
@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40  # vLLM doesn't use top_k but kept for compatibility
    repetition_penalty: float = 1.05
    seed: Optional[int] = None
    stream: bool = False

# -------------------------------
class LLM:
    """vLLM OpenAI-compatible client."""

    def __init__(self, model: str, server_url: str = "http://localhost:8000/v1"):
        self.model = model
        self.client = OpenAI(
            base_url=server_url,
            api_key="EMPTY"  # required dummy key for vLLM
        )

    def chat(self, system: str, user: str, cfg: GenerationConfig | None = None, ) -> str:
        cfg = cfg or GenerationConfig()
        # For non-chat models like MPT, fallback to generate()
        if "mpt" in self.model.lower() or "deepseek-coder" in self.model.lower():
            prompt = f"{system.strip()}\n{user.strip()}"
            return self.generate(prompt, cfg)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            seed=cfg.seed,
        )
        return response.choices[0].message.content


    def generate(self, prompt: str, cfg: GenerationConfig | None = None) -> str:
        cfg = cfg or GenerationConfig()
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            seed=cfg.seed,
        )
        return response.choices[0].text

# -------------------------------
from functools import lru_cache

@lru_cache(maxsize=2)
def get_llm(model_id: str, server_url: str = "http://localhost:8000/v1") -> LLM:
    return LLM(model=model_id, server_url=server_url)
