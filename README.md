# Joulehound

**FLOPs lie about energy. Joulehound sniffs out what your AI agent actually costs — measured on Arm silicon.**

Most efficiency claims for on-device AI are reported in FLOPs. But a single memory access costs ~100× more energy than a compute op, so FLOP counts systematically mislead. Joulehound optimizes a local *agentic* AI workload, measures the real energy on an Arm device, and shows that the [NEXUS physics-grounded metric](https://pypi.org/project/nexus-ml-metrics/) (FLAIRS 2026) tracks measured joules while FLOPs don't.

## The experiment

One agent task — *"decode a multi-spectral QR code and act on its contents"* — run in two configurations, N reps each, under a live power meter:

| | Baseline | Optimized |
|---|---|---|
| Agent loop | naive, unbounded | [pollard](https://pypi.org/project/pollard/)-governed: budgeted + gated |
| Tool calls | decode every step | decode once, cache, gate redundant calls |
| Measured | energy, power, duration | same |

**The result:** FLOPs barely distinguish the two configs; measured energy drops substantially, and NEXUS predicts it.

## Stack

- **[nexus-ml-metrics](https://pypi.org/project/nexus-ml-metrics/)** — physics-grounded energy scoring (the thesis)
- **[pollard](https://pypi.org/project/pollard/)** — governed execution trees for the agent loop (the optimization lever)
- **[multispecqr](https://pypi.org/project/multispecqr/)** — compute-heavy ML tool the agent invokes
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — on-device LLM brain (NEON-optimized on Arm)

## Run it

```bash
pip install -r requirements.txt

# Laptop, no hardware needed (synthetic power, proves the pipeline):
python bench.py --config baseline  --reps 5 --meter fake
python bench.py --config optimized --reps 5 --meter fake

# On the Arm device (real measured energy from battery sysfs):
python bench.py --config baseline  --reps 5 --meter android
python bench.py --config optimized --reps 5 --meter android
```

Results append to `results.csv`.

### Measuring on Android (Termux)

1. Install **Termux from F-Droid** (not the Play Store).
2. `pkg install python git && pip install -r requirements.txt`
3. Phone **unplugged**, airplane mode + WiFi, brightness fixed (USB charging poisons the reading).
4. Run with `--meter android`. Power comes from `/sys/class/power_supply/battery/current_now` and `voltage_now`.

## Reproduce on Arm64 in CI

The [`arm64-bench`](.github/workflows/arm64-bench.yml) workflow runs the whole benchmark on a GitHub-hosted **Arm64** runner. Hit "Run workflow" — no hardware required to verify the pipeline on real Arm silicon.

## Status

Agent loop, pollard governance, power harness, nexus scoring, and Arm64 CI all working. On-device llama.cpp inference and real multispecqr decode swap in via `joulehound/agent.py` (`llm_call` / `decode_qr`) — the measurement and scoring code is unchanged.

## License

MIT
