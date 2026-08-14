"""cv2 compatibility shim for Termux (Joulehound).

Termux/Android has no prebuilt OpenCV, and multispecqr imports cv2 at module
level. Its decoder tries cv2.QRCodeDetector FIRST and falls back to pyzbar
(zbar) per layer -- a fallback the library itself provides. This shim makes
the import succeed and the primary attempt report "no decode", so every
layer takes the pyzbar path. Decode results are identical; only the decoder
engine differs, and it differs equally in both benchmark configs.

Anything beyond QRCodeDetector raises immediately and loudly: this file is
NOT OpenCV. Activate only on the device:
    export PYTHONPATH=$HOME/Joulehound/compat
"""


class QRCodeDetector:
    def detectAndDecode(self, img):
        return "", None, None  # defer to multispecqr's pyzbar fallback


def __getattr__(name):
    raise ImportError(
        f"cv2 shim: '{name}' is not provided. This is Joulehound's Termux "
        "compatibility stub, not real OpenCV. Install opencv-python for "
        "full functionality."
    )
