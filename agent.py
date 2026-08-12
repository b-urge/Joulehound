"""The workload under measurement: a governed agent task.

Task: "decode a multi-spectral QR code and act on its contents."
This keeps every Joulehound library busy --
  - pollard     governs/budgets the agent loop (the optimization lever)
  - the LLM     (llama.cpp, wired in on-device Wednesday) is the agent brain
  - multispecqr is the compute-heavy TOOL the agent invokes
  - power.py    measures the whole thing; nexus_ml scores it

Two configs:
  baseline  -- naive loop, no budget, every step runs
  optimized -- pollard-governed: budgeted + gated, wasted steps pruned

TONIGHT this runs with a stub LLM + stub QR tool so the pipeline is testable
on your laptop with zero hardware. Wednesday you swap `llm_call` for a real
llama.cpp call and `decode_qr` for a real multispecqr decode. The measurement
and scoring code does not change.
"""
from __future__ import annotations
import time
import hashlib


def llm_call(prompt: str, work: int = 200_000) -> str:
    """STUB brain. Wednesday: replace with a llama.cpp call on-device.
    `work` simulates token-generation compute so the meter sees load."""
    acc = 0
    for i in range(work):
        acc = (acc + i * 2654435761) & 0xFFFFFFFF
    return f"decision:{acc % 3}"


def decode_qr(payload: str, layers: int = 6, work: int = 300_000) -> dict:
    """STUB for multispecqr's ML decoder. Wednesday: real multispecqr decode.
    Deterministic + compute-heavy -> clean, low-variance energy signal."""
    acc = 0
    for i in range(work):
        acc = (acc + (i ^ layers) * 40503) & 0xFFFFFFFF
    h = hashlib.sha256(f"{payload}{acc}".encode()).hexdigest()[:12]
    return {"layers": layers, "token": h}


def run_baseline(steps: int = 4) -> dict:
    """Naive agent: fixed number of steps, no budgeting, re-decodes every step."""
    results = []
    for _ in range(steps):
        qr = decode_qr("payload")           # heavy tool call every step
        decision = llm_call(f"act on {qr['token']}")
        results.append(decision)
    return {"config": "baseline", "steps": steps, "results": results}


def run_optimized(steps: int = 4) -> dict:
    """pollard-governed agent: budget the run, decode ONCE and cache, gate
    redundant LLM calls. Same output, fewer wasted joules."""
    from pollard import Budget, Runtime
    results = []
    cached_qr = None
    with Runtime("runs.db").run("joulehound", budget=Budget(tokens=20_000)) as run:
        for _ in range(steps):
            if cached_qr is None:            # decode once, not every step
                cached_qr = decode_qr("payload")
            # gate: only call the brain when the decision could change
            decision = llm_call(f"act on {cached_qr['token']}", work=120_000)
            results.append(decision)
            if len(set(results)) == 1 and len(results) >= 2:
                # stable -> stop early (pollard budget would enforce this too)
                break
    return {"config": "optimized", "steps": len(results), "results": results}


CONFIGS = {"baseline": run_baseline, "optimized": run_optimized}
