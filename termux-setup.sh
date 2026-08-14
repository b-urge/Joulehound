#!/data/data/com.termux/files/usr/bin/bash
# Joulehound device setup for Termux on Arm Android.
# Battle-tested on a Samsung Galaxy Tab S9+ (Snapdragon 8 Gen 2), Termux
# python 3.14, Aug 2026. Every flag below exists because the default path
# fails on-device; see README "Termux field notes" for the full war story.
set -e

echo "== packages (prebuilt: avoids compiling numpy/pillow/ninja on-device) =="
pkg update -y
pkg install -y python clang cmake make git openssh python-numpy python-pillow \
               termux-api zbar dbus tmux ninja
# zbar's compiled lib links libdbus but doesn't declare it -- hence dbus.
# ninja from pkg pre-empts pip building ninja from source (fails: no spawn.h).

echo "== python deps (the Python-3.14 wheel-desert workarounds) =="
# nexus-ml-metrics declares pandas but never imports it; pandas has no
# Android wheel and its source build dies. Skip deps, install what's used.
pip install --no-deps nexus-ml-metrics
pip install pollard pyzbar qrcode
# multispecqr 0.4.1 is pure python but its metadata caps Requires-Python
# at <3.13; Termux ships 3.14. Without the override pip silently falls back
# to a skeleton 0.0.1a0 that lacks encode_layers/decode_layers.
pip install --no-deps --ignore-requires-python "multispecqr==0.4.1"

echo "== opencv shim =="
# Termux/TUR have no prebuilt OpenCV. multispecqr's decoder tries
# cv2.QRCodeDetector first and falls back to pyzbar per layer; the repo's
# compat/cv2.py makes the import succeed and defers every layer to pyzbar.
grep -q "Joulehound/compat" ~/.profile 2>/dev/null || \
  echo 'export PYTHONPATH=$HOME/Joulehound/compat' >> ~/.profile
grep -q "Joulehound/compat" ~/.bashrc 2>/dev/null || \
  echo 'export PYTHONPATH=$HOME/Joulehound/compat' >> ~/.bashrc
export PYTHONPATH=$HOME/Joulehound/compat

echo "== REMINDER: Termux:API =="
echo "The termuxapi power meter needs the Termux:API *app* installed from"
echo "F-Droid (f-droid.org/en/packages/com.termux.api/) -- the pkg alone is"
echo "not enough. Also set Termux + Termux:API to Battery -> Unrestricted."

echo "== optional: local LLM brain (llama.cpp), ~20-40 min compile =="
read -r -p "Install llama-cpp-python now? [y/N] " yn
if [ "$yn" = "y" ] || [ "$yn" = "Y" ]; then
  pip install -v llama-cpp-python
  # PEP 738: python >=3.13 reports sys.platform == "android"; llama-cpp-python
  # <=0.3.34's loader only knows linux/darwin/win32 and raises
  # "Unsupported platform". Patch the one condition.
  python - <<'PY'
import pathlib, sysconfig
sp = pathlib.Path(sysconfig.get_paths()["purelib"]) / "llama_cpp/_ctypes_extensions.py"
s = sp.read_text()
old = 'elif sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):'
new = 'elif sys.platform.startswith(("linux", "android", "freebsd")):'
if old in s:
    sp.write_text(s.replace(old, new, 1))
    print("patched llama-cpp loader for android (PEP 738)")
else:
    print("loader pattern not found -- newer version may already handle android")
PY
  echo "Point the agent at a GGUF: export JOULEHOUND_MODEL=~/models/<model>.gguf"
fi

echo ""
echo "Setup complete. Verify, then measure:"
echo "  python test_meter_scaling.py"
echo "  python bench.py --config baseline  --reps 5 --warmup 3 --meter termuxapi"
echo "  python bench.py --config optimized --reps 5 --warmup 3 --meter termuxapi"
echo "Clean-run protocol: UNPLUGGED, airplane mode (WiFi back on if using SSH),"
echo "screen off (termux-wake-lock). See README."
