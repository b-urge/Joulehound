"""The workload under measurement: a governed agent task.

Task: "decode a multi-spectral QR code and act on its contents."
  - multispecqr  is the compute-heavy TOOL the agent invokes (REAL, wired in)
  - llama.cpp    is the agent brain (real when a model is present, else fallback)
  - pollard      governs/budgets the agent loop (the optimization lever)
  - power.py     measures; nexus_ml scores

Configs:
  baseline  -- naive loop: re-decodes the QR every step, calls the brain every step
  optimized -- pollard-governed: decode once + cache, gate redundant brain calls

LLM setup (on the device):
  pip install llama-cpp-python   # compiles llama.cpp with NEON on Arm
  export JOULEHOUND_MODEL=/path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf
If the model or library is absent, llm_call falls back to a deterministic
compute stub so the pipeline still runs anywhere (CI, laptop).
"""
from __future__ import annotations
import os

from multispecqr import encode_layers, decode_layers

# ---------------------------------------------------------------- QR tool --
# Six sensor readings packed into ONE multi-spectral QR image. Generated once
# at module load: this is the agent's "input document."
_PAYLOADS = [f"sensor-{i}:reading={i * 17}" for i in range(6)]
_QR_IMG = encode_layers(_PAYLOADS, version=6)


def decode_qr(payload: str = "", layers: int = 6) -> dict:
    """REAL multispecqr decode: recover 6 independent payloads from one image.
    Deterministic and compute-heavy (~seconds) -> clean energy signal."""
    decoded = decode_layers(_QR_IMG, num_layers=layers)
    return {"layers": layers, "token": decoded[0], "payloads": decoded}


# --------------------------------------------------------------- LLM brain --
_MODEL_PATH = os.environ.get("JOULEHOUND_MODEL", "")
_LLM = None


def _get_llm():
    global _LLM
    if _LLM is None and _MODEL_PATH and os.path.exists(_MODEL_PATH):
        from llama_cpp import Llama
        _LLM = Llama(model_path=_MODEL_PATH, n_ctx=1024, verbose=False)
    return _LLM


def llm_call(prompt: str, max_tokens: int = 64, work: int = 200_000) -> str:
    """Agent brain. Real llama.cpp inference when JOULEHOUND_MODEL is set;
    deterministic compute fallback otherwise (CI / laptop without a model)."""
    llm = _get_llm()
    if llm is not None:
        out = llm(prompt, max_tokens=max_tokens, temperature=0.0)
        return out["choices"][0]["text"].strip()
    acc = 0
    for i in range(work):
        acc = (acc + i * 2654435761) & 0xFFFFFFFF
    return f"decision:{acc % 3}"


# ----------------------------------------------------------------- configs --
_PROMPT = ("You are monitoring sensors. Readings: {readings}. "
           "Reply with one word: NORMAL or ALERT.")


def run_baseline(steps: int = 4) -> dict:
    """Naive agent: re-decodes the QR every step, brain call every step."""
    results = []
    for _ in range(steps):
        qr = decode_qr()                              # heavy decode EVERY step
        decision = llm_call(_PROMPT.format(readings=qr["payloads"]))
        results.append(decision)
    return {"config": "baseline", "steps": steps, "results": results}


def run_optimized(steps: int = 4) -> dict:
    """pollard-governed: decode once + cache, gate redundant brain calls,
    stop early once the decision is stable."""
    from pollard import Budget, Runtime
    results = []
    cached_qr = None
    with Runtime("runs.db").run("joulehound", budget=Budget(tokens=20_000)):
        for _ in range(steps):
            if cached_qr is None:
                cached_qr = decode_qr()               # decode ONCE
            decision = llm_call(
                _PROMPT.format(readings=cached_qr["payloads"]),
                work=120_000,
            )
            results.append(decision)
            if len(results) >= 2 and len(set(results)) == 1:
                break                                 # stable -> stop early
    return {"config": "optimized", "steps": len(results), "results": results}


CONFIGS = {"baseline": run_baseline, "optimized": run_optimized}
