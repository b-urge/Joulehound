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


def termux_stub_test() -> None:
    """Unit-proof the Termux:API meter against a stubbed binary."""
    import json

    d = tempfile.mkdtemp()
    stub = os.path.join(d, "termux-battery-status")
    payload = os.path.join(d, "payload.json")
    with open(stub, "w") as f:
        f.write("#!/bin/sh\ncat '%s'\n" % payload)
    os.chmod(stub, 0o755)
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = d + os.pathsep + old_path
    os.environ["JOULEHOUND_VBAT_MV"] = "3900"
    expected = 0.35 * 3.9
    cases = {
        "uA + mV (Tab S9+ dialect)": {"current": -350000, "voltage": 3900},
        "mA + missing voltage     ": {"current": -350},
        "mA + plain volts         ": {"current": 350, "voltage": 3.9},
    }
    try:
        for name, data in cases.items():
            with open(payload, "w") as f:
                json.dump({"status": "DISCHARGING", **data}, f)
            m = power.TermuxApiBatteryMeter()
            w = m._read_watts()
            ok = abs(w - expected) < 1e-9
            print(f"  {'OK ' if ok else 'FAIL'} {name} -> {w:.4f} W")
            assert ok, f"termuxapi scaling broken: {name}"
    finally:
        os.environ["PATH"] = old_path
        os.environ.pop("JOULEHOUND_VBAT_MV", None)
    print("Termux:API meter is unit-proof.")


if __name__ == "__main__":
    main()
    print()
    termux_stub_test()
