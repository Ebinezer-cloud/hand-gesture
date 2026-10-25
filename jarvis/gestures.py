"""
Finger / gesture classification from 21 MediaPipe hand landmarks.

Everything is based on RATIOS between landmark distances, so it works at any
distance from the camera and with the hand rotated in any direction.

Gestures
    OPEN_PALM   all 4 fingers extended              -> human heart
    FIST        all 4 fingers folded                -> car
    PINCH       thumb tip touches index tip         -> shadow mantis face
    ONE_FINGER  only the index finger extended      -> butterfly
"""
import math
from collections import Counter, deque
from types import SimpleNamespace

# landmark ids
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {  # name -> (mcp, pip, tip)
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}

OPEN_PALM, FIST, PINCH, ONE_FINGER, UNKNOWN = "OPEN_PALM", "FIST", "PINCH", "ONE_FINGER", "UNKNOWN"

LABELS = {
    OPEN_PALM: "OPEN PALM",
    FIST: "CLOSED FIST",
    PINCH: "PINCH",
    ONE_FINGER: "ONE FINGER",
    UNKNOWN: "SCANNING",
}

# ---- tunable thresholds (use `python main.py --debug` to see live values) ----
FINGER_EXTENDED_RATIO = 1.45   # (tip->wrist) / (mcp->wrist) above this = extended
THUMB_OUT_RATIO = 0.62         # (thumb tip -> index mcp) / palm size above this = thumb out
PINCH_RATIO = 0.30             # (thumb tip -> index tip) / palm size below this = pinch
PINCH_INDEX_MIN_RATIO = 1.12   # index must not be fully curled (otherwise it's a fist)


def _d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def palm_size(lm):
    return max(_d(lm[WRIST], lm[MIDDLE_MCP]), 1e-6)


def finger_ratios(lm):
    """Return {finger: tip-to-wrist / mcp-to-wrist}. ~1.9 = straight, ~0.9 = folded."""
    out = {}
    for name, (mcp, pip, tip) in FINGERS.items():
        base = max(_d(lm[mcp], lm[WRIST]), 1e-6)
        out[name] = _d(lm[tip], lm[WRIST]) / base
    return out


def fingers_extended(lm):
    """Return dict finger -> bool for the four fingers plus the thumb."""
    ratios = finger_ratios(lm)
    state = {}
    for name, (mcp, pip, tip) in FINGERS.items():
        # tip must be far from the wrist AND farther than the middle joint
        state[name] = ratios[name] > FINGER_EXTENDED_RATIO and _d(lm[tip], lm[WRIST]) > _d(lm[pip], lm[WRIST])
    ps = palm_size(lm)
    thumb_far = _d(lm[THUMB_TIP], lm[INDEX_MCP]) / ps > THUMB_OUT_RATIO
    thumb_straight = _d(lm[THUMB_TIP], lm[PINKY_MCP]) > _d(lm[THUMB_IP], lm[PINKY_MCP])
    state["thumb"] = thumb_far and thumb_straight
    return state


def to_square(lm, aspect):
    """MediaPipe x/y are normalised by width/height separately, so on a 16:9 frame
    distances are stretched. Multiply x by width/height to get isotropic units."""
    return [SimpleNamespace(x=p.x * aspect, y=p.y) for p in lm]


def classify(lm, aspect=1.0):
    """lm: list of 21 landmarks with .x/.y (normalised). aspect = frame width / height.
    Returns (gesture, info dict)."""
    lm = to_square(lm, aspect)
    ext = fingers_extended(lm)
    ratios = finger_ratios(lm)
    ps = palm_size(lm)
    pinch_dist = _d(lm[THUMB_TIP], lm[INDEX_TIP]) / ps
    count = sum(1 for v in ext.values() if v)

    four = [ext["index"], ext["middle"], ext["ring"], ext["pinky"]]
    info = {
        "extended": ext,
        "ratios": ratios,
        "pinch": pinch_dist,
        "count": count,
    }

    if pinch_dist < PINCH_RATIO and ratios["index"] > PINCH_INDEX_MIN_RATIO:
        return PINCH, info
    if all(four):
        return OPEN_PALM, info
    if ext["index"] and not (ext["middle"] or ext["ring"] or ext["pinky"]):
        return ONE_FINGER, info
    if not any(four):
        return FIST, info
    return UNKNOWN, info


class GestureSmoother:
    """Majority vote over the last N frames so the overlay doesn't flicker."""

    def __init__(self, size=7, need=4):
        self.hist = deque(maxlen=size)
        self.need = need
        self.stable = UNKNOWN

    def update(self, gesture):
        self.hist.append(gesture)
        top, n = Counter(self.hist).most_common(1)[0]
        if n >= self.need:
            self.stable = top
        return self.stable

    def reset(self):
        self.hist.clear()
        self.stable = UNKNOWN
