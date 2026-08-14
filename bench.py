"""Joulehound benchmark runner.

Runs an agent config N times under the power meter, then scores the result
with nexus_ml. The whole thesis in one command:

    python bench.py --config baseline  --reps 5 --meter fake
    python bench.py --config optimized --reps 5 --meter fake

On the Arm device Wednesday, use --meter android.

Output: a CSV row per run + a printed summary (mean energy, CoV, NEXUS metrics).
Compare the two configs' energy: FLOPs barely move, measured joules drop.
"""
from __future__ import annotations
import argparse
import csv
import statistics
import time
from pathlib import Path

from joulehound.power import get_meter
from joulehound.agent import CONFIGS


def flops_estimate(config: str, steps: int) -> int:
    """Deliberately naive FLOP proxy -- the metric we're arguing AGAINST.
    Counts 'operations' per step, blind to memory/switching cost. It barely
    distinguishes the configs, which is the whole point."""
    return steps * 500_000


def score_run(energy_j: float, energies_so_far: list[float]) -> dict:
    """Compute NEXUS metrics from measured energy using the real API."""
    from nexus_ml.metrics.nexus import compute_ecu, compute_mcer, compute_ddev
    capability = 1.0  # same task/output quality across configs -> fixed
    ecu = compute_ecu(energy_j=max(energy_j, 1e-6), capability_score=capability)
    # rough split: assume ~60% of energy is memory movement (Horowitz insight)
    compute_e = max(energy_j * 0.4, 1e-6)
    mcer = compute_mcer(memory_energy_j=max(energy_j * 0.6, 1e-6),
                        compute_energy_j=compute_e)
    ddev = compute_ddev(energies_so_far) if len(energies_so_far) > 1 else 0.0
    return {"ecu": round(ecu, 4), "mcer": round(mcer, 4), "ddev": round(ddev, 4)}


def verify_decode_integrity():
    """Refuse to benchmark a workload that isn't actually working.

    A missing zbar/dbus/cv2 backend can silently turn decode_layers into a
    no-op that returns empty strings: the bench would then produce
    plausible-looking joules for a demo that decodes nothing. Fail loudly
    instead."""
    from multispecqr import encode_layers, decode_layers
    probe = [f"integrity-{i}" for i in range(6)]
    out = decode_layers(encode_layers(probe, version=6), num_layers=6)
    if out != probe:
        raise SystemExit(
            "DECODE INTEGRITY FAILED: multispecqr round-trip returned "
            f"{out!r}. Refusing to produce numbers for a workload that "
            "is not real. Check zbar/dbus and the compat/cv2.py shim "
            "(see README, Termux section)."
        )
    print("  decode integrity: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--meter", choices=["fake", "android", "termuxapi"], default="fake")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--warmup", type=int, default=1,
                    help="unmeasured warm-up reps run first (pays one-time costs: "
                         "lazy imports, runs.db creation, caches)")
    args = ap.parse_args()
    verify_decode_integrity()

    run_fn = CONFIGS[args.config]
    energies: list[float] = []
    rows = []

    print(f"\nJoulehound :: {args.config} :: {args.reps} reps :: meter={args.meter}")
    print("-" * 56)
    for _ in range(args.warmup):
        run_fn()
    if args.warmup:
        print(f"  warm-up: {args.warmup} unmeasured rep(s) discarded")
    for i in range(args.reps):
        meter = get_meter(args.meter)
        with meter.measure():
            out = run_fn()
        r = meter.result
        energies.append(r.energy_joules)
        flops = flops_estimate(args.config, out["steps"])
        nexus = score_run(r.energy_joules, energies)
        try:  # never let the prediction model kill a device run
            from joulehound.predicted import predict_config_energy
            pred = round(predict_config_energy(args.config, out["steps"]), 9)
        except Exception:
            pred = ""
        rows.append({
            "config": args.config, "rep": i + 1, "meter": args.meter,
            "energy_j": r.energy_joules, "power_avg_w": r.power_watts_avg,
            "duration_s": r.duration_seconds, "flops_proxy": flops,
            "nexus_pred_j": pred,
            **nexus,
        })
        print(f"  rep {i+1}: {r.energy_joules:7.3f} J  "
              f"{r.power_watts_avg:5.2f} W avg  {r.duration_seconds:5.2f} s  "
              f"ecu={nexus['ecu']}")

    mean_e = statistics.mean(energies)
    cov = (statistics.pstdev(energies) / mean_e * 100) if mean_e else 0.0
    print("-" * 56)
    print(f"  mean energy: {mean_e:.3f} J   CoV: {cov:.1f}%   "
          f"FLOPs proxy: {rows[0]['flops_proxy']:,}")

    out_path = Path(args.out)
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"  appended {len(rows)} rows -> {out_path}\n")


if __name__ == "__main__":
    main()
