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
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._t0 = time.time()
        self._thread.start()

    def _finish(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        duration = time.time() - self._t0
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


class AndroidBatteryMeter(_BaseMeter):
    """Real power from Android battery sysfs. Run inside Termux on the phone.

    IMPORTANT for clean numbers: phone UNPLUGGED (USB charging poisons the
    reading), airplane mode + WiFi only, screen brightness fixed.
    """
    def _read_watts(self) -> float:
        with open(CURRENT_PATH) as f:
            amps = abs(int(f.read().strip())) / 1e6
        with open(VOLTAGE_PATH) as f:
            volts = int(f.read().strip()) / 1e6
        return amps * volts


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
    return {"fake": FakeMeter, "android": AndroidBatteryMeter}[name](**kw)
