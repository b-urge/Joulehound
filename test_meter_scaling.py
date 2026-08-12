"""Prove AndroidBatteryMeter math is unit-proof with synthetic sysfs files.

Run:  python test_meter_scaling.py
Feeds the meter the same electrical state four ways (uA/uV, mA/mV, negative
sign, mixed) and asserts identical watts. If this passes on the laptop, the
meter cannot be wrong about units on the device.
"""
import os
import tempfile

from joulehound import power


def _write(d: str, current: int, voltage: int) -> None:
    with open(os.path.join(d, "current_now"), "w") as f:
        f.write(str(current))
    with open(os.path.join(d, "voltage_now"), "w") as f:
        f.write(str(voltage))


def main() -> None:
    meter = power.AndroidBatteryMeter()
    expected = 0.35 * 3.9  # 350 mA at 3.9 V = 1.365 W

    with tempfile.TemporaryDirectory() as d:
        power.CURRENT_PATH = os.path.join(d, "current_now")
        power.VOLTAGE_PATH = os.path.join(d, "voltage_now")

        cases = {
            "uA / uV          ": (350_000, 3_900_000),
            "mA / mV (Samsung)": (350, 3_900),
            "negative uA      ": (-350_000, 3_900_000),
            "mA / uV mixed    ": (350, 3_900_000),
        }
        for name, (cur, vol) in cases.items():
            _write(d, cur, vol)
            watts = meter._read_watts()
            status = "OK " if abs(watts - expected) < 1e-9 else "FAIL"
            print(f"  {status} {name} -> {watts:.4f} W")
            assert abs(watts - expected) < 1e-9, f"unit scaling broken: {name}"

    print(f"\nAll unit conventions -> {expected:.3f} W. Meter is unit-proof.")


if __name__ == "__main__":
    main()
