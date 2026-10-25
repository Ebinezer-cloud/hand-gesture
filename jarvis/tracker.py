"""Thin wrapper around MediaPipe's Hand Landmarker (Tasks API)."""
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from . import modelfetch

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def ensure_model(path):
    """Make sure the ~8 MB model file exists (downloads it once, with progress)."""
    if not modelfetch.fetch(path):
        print("\n[jarvis] could not download the hand model.")
        print("         Open this link in your browser, save the file into the 'models' folder")
        print("         (or your Downloads folder) and run again:")
        print(f"         {modelfetch.MODEL_URL}")
        print(f"         target: {path}")
        sys.exit(1)
    return path


class HandTracker:
    def __init__(self, model_path, max_hands=2, detect_conf=0.6, track_conf=0.5):
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detect_conf,
            min_hand_presence_confidence=detect_conf,
            min_tracking_confidence=track_conf,
        )
        self.lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = 0

    def detect(self, bgr):
        """Returns list of (label, score, landmarks[21]) for each hand found."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts:  # timestamps must strictly increase
            ts = self._last_ts + 1
        self._last_ts = ts
        res = self.lm.detect_for_video(image, ts)
        hands = []
        for i, pts in enumerate(res.hand_landmarks):
            cat = res.handedness[i][0]
            hands.append((cat.category_name, cat.score, pts))
        return hands

    def close(self):
        self.lm.close()
