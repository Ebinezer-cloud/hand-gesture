"""Make sure models/hand_landmarker.task exists and is valid.  Exit code 0 = OK, 1 = failed."""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from jarvis.modelfetch import fetch  # noqa: E402

dest = os.path.abspath(os.path.join(ROOT, "models", "hand_landmarker.task"))
sys.exit(0 if fetch(dest) else 1)
