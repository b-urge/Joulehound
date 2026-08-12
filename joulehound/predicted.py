"""NEXUS-style predicted energy for the Joulehound workload.

Why this module exists
----------------------
nexus-ml-metrics v0.0.1 ships ``predict_energy()`` as a stub (it raises
NotImplementedError) and its beta-coefficient table is EMPTY pending upstream
validation of the Horowitz source data. So we do what the package's own API
invites: populate ``BetaCoefficientTable`` via its public ``set()`` method
with the canonical Horowitz (ISSCC 2014, DOI 10.1109/ISSCC.2014.6757323)
per-operation energies, then count THIS workload's operations from its actual
structure in agent.py.

Honesty notes (also in the README):
  * beta values here are per-operation energies in picojoules (45 nm,
    Horowitz slide values). The alpha calibration (joules per transistor-op on
    THIS silicon) is unknown, so ABSOLUTE predicted joules are not meaningful.
  * The killer chart is normalized (baseline = 1.0), where alpha cancels.
    The RATIO between configs is the prediction under test.
  * DRAM/SRAM figures are per 32-bit access; we count per-value accesses,
    which overstates absolute energy identically in both configs (ratio-safe).
"""
from __future__ import annotations

import os

from nexus_ml.core.coefficients import BetaCoefficient, BetaCoefficientTable
from nexus_ml.core import OperationType as OT, Precision as P

# --------------------------------------------------------------------------
# Horowitz (ISSCC 2014) per-operation energy, picojoules. 45 nm process.
# source='horowitz_2014' = read directly off the published data;
# confidence='derived'   = composed from those numbers (e.g. MAC = mul + add).
# --------------------------------------------------------------------------
_HOROWITZ_PJ = [
    (OT.ADD,          P.INT8,  0.03,  "horowitz_2014", "verified"),
    (OT.ADD,          P.INT32, 0.1,   "horowitz_2014", "verified"),
    (OT.ADD,          P.FP16,  0.4,   "horowitz_2014", "verified"),
    (OT.ADD,          P.FP32,  0.9,   "horowitz_2014", "verified"),
    (OT.MULTIPLY,     P.INT8,  0.2,   "horowitz_2014", "verified"),
    (OT.MULTIPLY,     P.INT32, 3.1,   "horowitz_2014", "verified"),
    (OT.MULTIPLY,     P.FP16,  1.1,   "horowitz_2014", "verified"),
    (OT.MULTIPLY,     P.FP32,  3.7,   "horowitz_2014", "verified"),
    (OT.MAC,          P.INT8,  0.23,  "derived",       "derived"),   # mul+add
    (OT.MAC,          P.FP16,  1.5,   "derived",       "derived"),
    (OT.COMPARISON,   P.INT8,  0.03,  "derived",       "derived"),   # ~= add
    (OT.CACHE_ACCESS, P.INT8,  5.0,   "horowitz_2014", "verified"),  # 8KB SRAM, 32b
    (OT.MEMORY_READ,  P.INT8,  640.0, "horowitz_2014", "verified"),  # DRAM, 32b
    (OT.MEMORY_WRITE, P.INT8,  640.0, "derived",       "derived"),
]

TABLE = BetaCoefficientTable()
for _op, _prec, _pj, _src, _conf in _HOROWITZ_PJ:
    TABLE.set(BetaCoefficient(
        op_type=_op, precision=_prec, value=_pj,
        source=_src, confidence=_conf,
        notes="Registered by Joulehound via the public set() API; "
              "upstream defaults are empty in v0.0.1.",
    ))

# --------------------------------------------------------------------------
# Workload operation counts, derived from agent.py's actual structure.
# --------------------------------------------------------------------------
# Measured from the real artifact: encode_layers(6 payloads, version=6)
# returns a 490x490 RGB PIL image.
IMG_PIXELS = 490 * 490            # 240,100 positions
IMG_VALUES = IMG_PIXELS * 3       # 720,300 uint8 subpixel values
LAYERS = 6

# llama.cpp fallback loop intensity (agent.py llm_call work=...)
BASELINE_LLM_WORK = 200_000
OPTIMIZED_LLM_WORK = 120_000

# Real-model path (device): first-order transformer cost model.
_GEN_TOKENS = 64                                  # agent.py max_tokens
_Q4_BYTES_PER_PARAM = 0.56                        # Q4_K_M on-disk density


def _decode_counts() -> dict:
    """One multispecqr threshold decode of the 6-layer, version-6 image."""
    return {
        (OT.MEMORY_READ,  P.INT8): IMG_VALUES,           # image in from DRAM
        (OT.CACHE_ACCESS, P.INT8): LAYERS * IMG_PIXELS,  # per-layer sweep, SRAM
        (OT.COMPARISON,   P.INT8): LAYERS * IMG_PIXELS,  # threshold compares
        (OT.ADD,          P.INT32): IMG_PIXELS,          # module accumulation
    }


def _llm_counts(work: int) -> dict:
    """One brain call.

    Fallback (no JOULEHOUND_MODEL): the deterministic loop in agent.py is
    literally `work` iterations of one INT32 multiply + one INT32 add.

    Real model: weights stream from DRAM once per generated token — the
    memory-bound reality of on-device LLM inference, and exactly the cost
    FLOPs-thinking misses.
    """
    model_path = os.environ.get("JOULEHOUND_MODEL", "")
    if model_path and os.path.exists(model_path):
        params = float(os.environ.get("JOULEHOUND_MODEL_PARAMS", 3.09e9))
        tokens = _GEN_TOKENS + 1                      # +1 ~= prefill lump
        weight_words = params * _Q4_BYTES_PER_PARAM / 4.0   # 32-bit words
        return {
            (OT.MAC,         P.INT8): int(params * tokens),
            (OT.MEMORY_READ, P.INT8): int(weight_words * tokens),
        }
    return {
        (OT.MULTIPLY, P.INT32): work,
        (OT.ADD,      P.INT32): work,
    }


def _merge(into: dict, add: dict, times: int = 1) -> dict:
    for key, count in add.items():
        into[key] = into.get(key, 0) + count * times
    return into


def config_counts(config: str, steps_executed: int) -> dict:
    """Operation counts for one full run of a config.

    baseline  : decode EVERY step + brain call EVERY step (steps_executed).
    optimized : decode ONCE + gated brain calls (steps_executed already
                reflects pollard's early stop).
    """
    counts: dict = {}
    if config == "baseline":
        _merge(counts, _decode_counts(), times=steps_executed)
        _merge(counts, _llm_counts(BASELINE_LLM_WORK), times=steps_executed)
    elif config == "optimized":
        _merge(counts, _decode_counts(), times=1)
        _merge(counts, _llm_counts(OPTIMIZED_LLM_WORK), times=steps_executed)
    else:
        raise ValueError(f"unknown config: {config}")
    return counts


def predict_config_energy(config: str, steps_executed: int) -> float:
    """NEXUS-style predicted energy in joules (alpha-uncalibrated: use ratios).

    energy = sum over ops of count x beta(op, precision), beta in pJ.
    """
    total_pj = 0.0
    for (op, prec), count in config_counts(config, steps_executed).items():
        total_pj += count * TABLE.get(op, prec).value
    return total_pj * 1e-12


def breakdown(config: str, steps_executed: int) -> dict:
    """Per-domain split (pJ) for the README table: where the energy model
    says the joules actually go. Memory dominance is the whole point."""
    mem_ops = {OT.MEMORY_READ, OT.MEMORY_WRITE, OT.CACHE_ACCESS}
    out = {"memory_pj": 0.0, "compute_pj": 0.0}
    for (op, prec), count in config_counts(config, steps_executed).items():
        bucket = "memory_pj" if op in mem_ops else "compute_pj"
        out[bucket] += count * TABLE.get(op, prec).value
    return out


if __name__ == "__main__":
    for cfg, steps in (("baseline", 4), ("optimized", 2)):
        e = predict_config_energy(cfg, steps)
        b = breakdown(cfg, steps)
        print(f"{cfg:10s} steps={steps}  pred={e:.6e} J  "
              f"mem={b['memory_pj']:.3e} pJ  compute={b['compute_pj']:.3e} pJ")
    base = predict_config_energy("baseline", 4)
    opt = predict_config_energy("optimized", 2)
    print(f"predicted optimized/baseline ratio: {opt / base:.3f}")
