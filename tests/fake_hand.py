"""Synthetic 21-point hands for testing the classifier without a camera."""
import math
from types import SimpleNamespace

# hand-local coordinates (x right, y up), palm size (wrist -> middle MCP) = 1.0
MCP = {"index": (-0.30, 0.95), "middle": (0.0, 1.0), "ring": (0.28, 0.95), "pinky": (0.52, 0.80)}
DIR = {"index": (-0.08, 1.0), "middle": (0.0, 1.0), "ring": (0.07, 1.0), "pinky": (0.2, 1.0)}


def _norm(v):
    l = math.hypot(*v)
    return (v[0] / l, v[1] / l)


def build(extended=(), thumb_out=False, pinch=False, cx=0.5, cy=0.65, scale=0.14, rot=0.0):
    """extended: iterable of finger names that are straight. Returns list of 21 points (x,y in 0..1)."""
    pts = [None] * 21
    pts[0] = (0.0, 0.0)
    # thumb
    pts[1], pts[2] = (-0.30, 0.25), (-0.52, 0.48) if thumb_out else (-0.48, 0.45)
    if thumb_out:
        pts[3], pts[4] = (-0.78, 0.68), (-1.00, 0.85)
    else:
        pts[3], pts[4] = (-0.35, 0.70), (-0.10, 0.55)
    ids = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    for name, base in ids.items():
        mcp = MCP[name]
        d = _norm(DIR[name])
        pts[base] = mcp
        if name in extended:
            pts[base + 1] = (mcp[0] + d[0] * 0.45, mcp[1] + d[1] * 0.45)
            pts[base + 2] = (mcp[0] + d[0] * 0.70, mcp[1] + d[1] * 0.70)
            pts[base + 3] = (mcp[0] + d[0] * 0.95, mcp[1] + d[1] * 0.95)
        else:  # folded: pip goes up, tip curls back toward the palm
            pts[base + 1] = (mcp[0] + d[0] * 0.38, mcp[1] + d[1] * 0.38)
            pts[base + 2] = (mcp[0] + d[0] * 0.30, mcp[1] + d[1] * 0.05)
            pts[base + 3] = (mcp[0] + d[0] * 0.05, mcp[1] - 0.30)
    if pinch:  # index bent toward the thumb, thumb tip meets index tip
        i = MCP["index"]
        pts[6] = (i[0] - 0.05, i[1] + 0.40)
        pts[7] = (i[0] - 0.20, i[1] + 0.52)
        pts[8] = (-0.52, 1.28)
        pts[2], pts[3], pts[4] = (-0.50, 0.46), (-0.62, 0.85), (-0.54, 1.25)
    out = []
    c, s = math.cos(rot), math.sin(rot)
    for (x, y) in pts:
        xr, yr = x * c - y * s, x * s + y * c
        out.append(SimpleNamespace(x=cx + xr * scale, y=cy - yr * scale * 1.78, z=0.0))
    return out


def open_palm(**k):  return build(extended=("index", "middle", "ring", "pinky"), thumb_out=True, **k)
def fist(**k):       return build(extended=(), thumb_out=False, **k)
def one_finger(**k): return build(extended=("index",), thumb_out=False, **k)
def pinch(**k):      return build(extended=("middle", "ring", "pinky"), pinch=True, **k)
def pinch_fist(**k): return build(extended=(), pinch=True, **k)
