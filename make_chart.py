"""Generate the killer chart from results.csv.

    python make_chart.py                 # reads results.csv, writes chart.png
    python make_chart.py results.csv out.png

Three metrics, both configs, everything normalized to baseline = 1.0:
FLOPs proxy vs NEXUS-predicted energy vs measured energy. The point of the
chart: the FLOPs bar and the measured bar disagree; the physics bar and the
measured bar don't. Laptop-only tool (needs matplotlib) — run it on the CSV
scp'd back from the device.
"""
from __future__ import annotations

import csv
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE_COLOR = "#8892a0"
OPTIMIZED_COLOR = "#2fa36b"


def load(path: str) -> dict:
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        sys.exit(f"{path} is empty — run bench.py first.")
    by_cfg: dict[str, list[dict]] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r)
    missing = {"baseline", "optimized"} - set(by_cfg)
    if missing:
        sys.exit(f"{path} is missing config(s): {missing} — run both configs.")
    return by_cfg


def col(rows: list[dict], name: str) -> list[float]:
    return [float(r[name]) for r in rows if r.get(name) not in ("", None)]


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "chart.png"
    by_cfg = load(csv_path)
    base, opt = by_cfg["baseline"], by_cfg["optimized"]

    metrics = [("FLOPs proxy", "flops_proxy"),
               ("NEXUS-predicted\nenergy", "nexus_pred_j"),
               ("Measured\nenergy", "energy_j")]

    labels, base_vals, opt_vals, base_err, opt_err = [], [], [], [], []
    for label, key in metrics:
        b, o = col(base, key), col(opt, key)
        if not b or not o:
            continue  # e.g. nexus_pred_j absent — chart still works
        b_mean, o_mean = statistics.mean(b), statistics.mean(o)
        labels.append(label)
        base_vals.append(1.0)
        opt_vals.append(o_mean / b_mean)
        scale = b_mean
        base_err.append(statistics.pstdev(b) / scale if len(b) > 1 else 0.0)
        opt_err.append(statistics.pstdev(o) / scale if len(o) > 1 else 0.0)

    meter = base[0].get("meter", "?")
    reps = f"{len(base)}x baseline / {len(opt)}x optimized reps"

    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
    ax.bar([i - width / 2 for i in x], base_vals, width, yerr=base_err,
           capsize=4, color=BASELINE_COLOR, label="baseline")
    bars = ax.bar([i + width / 2 for i in x], opt_vals, width, yerr=opt_err,
                  capsize=4, color=OPTIMIZED_COLOR, label="optimized")
    for rect, v in zip(bars, opt_vals):
        ax.annotate(f"\u2212{(1 - v) * 100:.0f}%",
                    (rect.get_x() + rect.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=12, fontweight="bold",
                    xytext=(0, 6), textcoords="offset points")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("relative to baseline (= 1.0)", fontsize=11)
    ax.set_ylim(0, max(1.22, 1.0 + max(base_err, default=0.0) + 0.10))
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Same task, two configs: what each metric thinks it costs",
                 fontsize=14, pad=14)
    ax.legend(frameon=False, fontsize=11)

    drops = dict(zip(labels, opt_vals))
    flops_d = (1 - drops.get("FLOPs proxy", 1)) * 100
    meas_d = (1 - drops.get("Measured\nenergy", 1)) * 100
    caption = (f"FLOPs claims \u2212{flops_d:.0f}%. The battery says "
               f"\u2212{meas_d:.0f}%.")
    pred_key = "NEXUS-predicted\nenergy"
    if pred_key in drops:
        pred_d = (1 - drops[pred_key]) * 100
        caption += f" The physics model called \u2212{pred_d:.0f}%."
    fig.text(0.5, 0.015, f"{caption}   ({reps}, meter={meter})",
             ha="center", fontsize=10, style="italic", color="#444444")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    print(f"  {caption}  [{reps}, meter={meter}]")


if __name__ == "__main__":
    main()
