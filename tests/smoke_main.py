"""Runs main.main() headless with a fake camera + fake tracker to catch runtime errors."""
import os, sys
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".")); sys.path.insert(0, os.path.join(HERE, ".."))
import fake_hand as fh
import main as app

class FakeCap:
    def __init__(self, *a, **k): self.n = 0
    def isOpened(self): return True
    def set(self, *a): return True
    def read(self):
        import time; time.sleep(0.004)
        self.n += 1
        return True, np.full((720, 1280, 3), 70, np.uint8)
    def release(self): pass

class FakeTracker:
    def __init__(self, *a, **k): self.i = 0
    def detect(self, frame):
        self.i += 1
        makers = [fh.open_palm, fh.fist, fh.pinch, fh.one_finger]
        if self.i % 90 > 80: return []                     # hand disappears
        m = makers[(self.i // 20) % 4]
        hands = [("Right", 0.9, m(cx=0.6, cy=0.75))]
        if self.i > 100: hands.append(("Left", 0.9, fh.open_palm(cx=0.25, cy=0.75)))
        return hands
    def close(self): pass

keys = iter([-1] * 30 + [ord("1"), ord("2"), ord("d"), ord("h"), ord("h"), ord("v"), ord("s")] + [-1] * 150 + [ord("q")])
cv2.VideoCapture = FakeCap
cv2.imshow = lambda *a: None
cv2.namedWindow = lambda *a: None
cv2.setWindowProperty = lambda *a: None
cv2.getWindowProperty = lambda *a: 1
cv2.waitKey = lambda *a: next(keys)
cv2.destroyAllWindows = lambda: None
app.HandTracker = FakeTracker
app.ensure_model = lambda p: p
sys.argv = ["main.py", "--no-voice"]
app.main()
print("SMOKE OK")
