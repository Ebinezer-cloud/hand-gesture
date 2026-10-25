import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import fake_hand as fh
from jarvis import gestures as G

ASPECT = 1.78  # 16:9 camera

CASES = [
    ("open palm", fh.open_palm, G.OPEN_PALM),
    ("fist", fh.fist, G.FIST),
    ("one finger", fh.one_finger, G.ONE_FINGER),
    ("pinch (others open)", fh.pinch, G.PINCH),
    ("pinch (others curled)", fh.pinch_fist, G.PINCH),
]

fails = 0
for rot in (0.0, math.pi / 2, -0.6, math.pi):
    for scale in (0.14, 0.05):
        for name, make, want in CASES:
            got, info = G.classify(make(rot=rot, scale=scale), ASPECT)
            ok = got == want
            fails += not ok
            if not ok or (rot == 0.0 and scale == 0.14):
                print(f"{'PASS' if ok else 'FAIL'}  {name:24s} rot={rot:5.2f} scale={scale}  -> {got}  "
                      f"count={info['count']} pinch={info['pinch']:.2f}")

# finger counting for the HUD
_, i = G.classify(fh.open_palm(), ASPECT)
assert i["count"] == 5, i
_, i = G.classify(fh.fist(), ASPECT)
assert i["count"] == 0, i
_, i = G.classify(fh.one_finger(), ASPECT)
assert i["count"] == 1, i

# smoother
sm = G.GestureSmoother(size=7, need=4)
seq = [G.FIST] * 3 + [G.OPEN_PALM] * 7
out = [sm.update(g) for g in seq]
assert out[2] == G.UNKNOWN and out[-1] == G.OPEN_PALM, out
print("\nfailures:", fails)
sys.exit(1 if fails else 0)
