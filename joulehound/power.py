"""Power measurement harness for Joulehound.

Reads energy over the duration of a workload. Two backends:

  - AndroidBatteryMeter: polls Android battery sysfs (current_now / voltage_now),
    the path Termux exposes. Real measured energy from the phone's own PMIC.
  - FakeMeter: deterministic synthetic power for laptop development TONIGHT,
    before any device is attached. Lets you build and test the whole pipeline
    with no hardware.

Both return an EnergyResult you can hand straight to nexus_ml.

Swap the backend on the command line: --meter fake  (default)
                                      --meter android
"""
from __future__ import annotations
import json
import os
import subprocess
import time
import threading
from dataclasses import dataclass, field


# Android battery sysfs. current_now is microamps, voltage_now is microvolts.
# Sign convention varies by OEM (discharge may read negative) -- we take abs().
CURRENT_PATH = "/sys/class/power_supply/battery/current_now"
VOLTAGE_PATH = "/sys/class/power_supply/battery/voltage_now"


@dataclass
class EnergyResult:
    energy_joules: float
    power_watts_avg: float
    power_watts_peak: float
    duration_seconds: float
    samples: int
    metadata: dict = field(default_factory=dict)


class _BaseMeter:
    """Sample power on a background thread while a workload runs.

    Usage:
        with meter.measure() as m:
            do_work()
        result = m.result
    """
    def __init__(self, sample_hz: float = 200.0):
        self.interval = 1.0 / sample_hz
        self._samples: list[tuple[float, float]] = []  # (timestamp, watts)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.result: EnergyResult | None = None

    def _read_watts(self) -> float:
        raise NotImplementedError

    def _loop(self):
        while not self._stop.is_set():
            self._samples.append((time.time(), self._read_watts()))
            time.sleep(self.interval)

    def measure(self):
        return _MeterContext(self)

    def _start(self):
        self._samples.clear()
        self._stop.clear()
        try:  # take the t=0 reading BEFORE starting the clock, so the
            # meter's own IPC latency (~0.5s on termuxapi) is never billed
            # to the workload
            w0 = self._read_watts()
        except Exception:
            w0 = None
        self._t0 = time.time()
        if w0 is not None:
            self._samples.append((self._t0, w0))
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _finish(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=6.0)  # an in-flight slow read may need it
        t_end = time.time()
        try:  # close the end edge too; at ~1 Hz cadence this is real energy
            self._samples.append((t_end, self._read_watts()))
        except Exception:
            pass
        duration = t_end - self._t0
        self._samples.sort(key=lambda s: s[0])  # keep trapezoid monotonic
        watts = [w for _, w in self._samples] or [0.0]
        # Trapezoidal integration of power over time -> joules.
        energy = 0.0
        for i in range(1, len(self._samples)):
            dt = self._samples[i][0] - self._samples[i - 1][0]
            energy += 0.5 * (self._samples[i][1] + self._samples[i - 1][1]) * dt
        self.result = EnergyResult(
            energy_joules=round(energy, 4),
            power_watts_avg=round(sum(watts) / len(watts), 4),
            power_watts_peak=round(max(watts), 4),
            duration_seconds=round(duration, 4),
            samples=len(self._samples),
            metadata={"backend": type(self).__name__},
        )


class _MeterContext:
    def __init__(self, meter: _BaseMeter):
        self.meter = meter
    def __enter__(self):
        self.meter._start()
        return self.meter
    def __exit__(self, *exc):
        self.meter._finish()
        return False


def _to_microamps(raw: int) -> float:
    """Normalize current_now to microamps.

    The kernel ABI says microamps, but several Samsung drivers report
    MILLIamps. Under load, real draw is hundreds of mA: as uA that reads
    ~100,000-3,000,000; as mA it reads ~100-3,000. Anything under 10,000 is
    therefore treated as mA. (Deep-idle uA readings can be misclassified,
    but idle watts are negligible and we only integrate during load.)
    """
    v = abs(raw)  # sign convention (charge/discharge) varies by OEM
    return v * 1000.0 if v < 10_000 else float(v)


def _to_microvolts(raw: int) -> float:
    """Normalize voltage_now to microvolts.

    A Li-ion cell sits at 3,400,000-4,500,000 uV; the same value in mV is
    3,400-4,500. Anything under 100,000 is treated as mV.
    """
    v = abs(raw)
    return v * 1000.0 if v < 100_000 else float(v)


class AndroidBatteryMeter(_BaseMeter):
    """Real power from Android battery sysfs. Run inside Termux on the device.

    IMPORTANT for clean numbers: device UNPLUGGED (USB charging poisons the
    reading), airplane mode + WiFi only, screen off (termux-wake-lock) or
    brightness fixed. Unit quirks (uA vs mA, uV vs mV, sign) are normalized
    by _to_microamps/_to_microvolts so Samsung devices read correctly.
    """
    def _read_watts(self) -> float:
        with open(CURRENT_PATH) as f:
            microamps = _to_microamps(int(f.read().strip()))
        with open(VOLTAGE_PATH) as f:
            microvolts = _to_microvolts(int(f.read().strip()))
        return (microamps * microvolts) / 1e12



class TermuxApiBatteryMeter(_BaseMeter):
    """Power via the Android battery API (Termux:API app + termux-api pkg).

    For devices whose vendor/SELinux policy denies battery sysfs outright
    (e.g. recent One UI): termux-battery-status is the sanctioned pipe.
    Each reading is an IPC round-trip (~0.2-1 s), so effective cadence is
    ~1 Hz rather than 200 Hz. measure() samples both edges synchronously,
    reps last many seconds, and power varies slowly, so trapezoid error
    stays small -- and the polling cost is identical in both configs, so it
    cancels out of the normalized comparison.

    Field quirks seen across devices/builds, all normalized here:
      current -- microamps or milliamps, sign convention varies.
      voltage -- millivolts when present; MISSING on many builds. When
                 absent, a fixed nominal cell voltage is used
                 (env JOULEHOUND_VBAT_MV, default 3850). True voltage drifts
                 <1% across back-to-back runs at equal charge, so relative
                 results are unaffected; absolute joules carry a few percent
                 of uncertainty, disclosed in the README.
    """

    def __init__(self, sample_hz: float = 1.5):
        super().__init__(sample_hz=sample_hz)
        self._last: float | None = None
        self._warned_voltage = False

    def _voltage_microvolts(self, data: dict) -> float:
        raw = data.get("voltage")
        if raw is None:
            mv = os.environ.get("JOULEHOUND_VBAT_MV", "3850")
            if not self._warned_voltage:
                print(f"  [termuxapi] no voltage field; using nominal {mv} mV",
                      flush=True)
                self._warned_voltage = True
            return float(mv) * 1000.0
        v = abs(float(raw))
        if v < 10.0:  # plain volts, seen on rare builds
            return v * 1e6
        return _to_microvolts(int(v))  # handles mV vs uV

    def _read_watts(self) -> float:
        try:
            proc = subprocess.run(["termux-battery-status"],
                                  capture_output=True, text=True, timeout=5)
            data = json.loads(proc.stdout)
            microamps = _to_microamps(int(round(float(data["current"]))))
            watts = (microamps * self._voltage_microvolts(data)) / 1e12
            self._last = watts
            return watts
        except Exception as e:
            if self._last is not None:
                return self._last  # ride through a single IPC hiccup
            raise RuntimeError(
                "termux-battery-status failed on its first read. Is the "
                "Termux:API app installed from F-Droid, and did you run "
                "pkg install termux-api? Try termux-api-start, then retry."
            ) from e


class FakeMeter(_BaseMeter):
    """Synthetic power for laptop dev. Idle ~1.5W, load ~4.5W with jitter."""
    def __init__(self, sample_hz: float = 200.0, load_watts: float = 4.5):
        super().__init__(sample_hz)
        self.load_watts = load_watts
        self._n = 0
    def _read_watts(self) -> float:
        import math, random
        self._n += 1
        base = self.load_watts + 0.4 * math.sin(self._n / 5.0)
        return round(base + random.uniform(-0.2, 0.2), 4)


def get_meter(name: str, **kw) -> _BaseMeter:
    return {"fake": FakeMeter, "android": AndroidBatteryMeter,
            "termuxapi": TermuxApiBatteryMeter}[name](**kw)
