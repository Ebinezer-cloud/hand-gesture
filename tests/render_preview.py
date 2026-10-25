"""Renders the full HUD + hologram pipeline offscreen with synthetic hands (no camera/model needed).
    python tests/render_preview.py            -> writes tests/preview_<gesture>.png and preview_grid.png
"""
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import fake_hand as fh
from jarvis import gestures as G
from jarvis import hud
from jarvis.sprites import Holo

W, H = 1280, 720
holo = Holo(os.path.join(HERE, "..", "assets"))


def fake_camera():
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:H, 0:W]
    base = (60 + 40 * (yy / H))[..., None] * np.array([0.8, 0.9, 1.0])[None, None, :]
    return np.clip(base + rng.normal(0, 4, (H, W, 3)), 0, 255).astype(np.uint8)


def render(name, make, label="Right", frames=40):
    holo.states.clear()
    log = deque([(time.strftime("%H:%M:%S"), "SYSTEM ONLINE")], maxlen=20)
    out = None
    for i in range(frames):
        t = i / 30.0 + 1.0
        lm = make(cx=0.5, cy=0.78, scale=0.13)
        frame = hud.filter_frame(fake_camera())
        fx = np.zeros_like(frame)
        st = holo.state(label)
        g, info = G.classify(lm, W / H)
        stable = st.smoother.update(g)
        if stable != st.last_gesture:
            log.append((time.strftime("%H:%M:%S"), f"{label.upper()}: {G.LABELS[stable]}"))
            st.last_gesture = stable
        hi = holo.draw(frame, st, lm, stable, True, t)
        hud.draw_hand(fx, lm, W, H, stable, info["extended"], t, f"HAND 1 // {label.upper()} // {G.LABELS[stable]}", info)
        if hi:
            hud.draw_hologram_frame(fx, hi, t)
        hud.draw_hud(frame, fx, t, 30.0, [{"gesture": stable, "count": info["count"]}], log,
                     {"voice": True}, hi["name"] if hi else None)
        out = hud.compose_fx(frame, fx)
    path = os.path.join(HERE, f"preview_{name}.png")
    cv2.imwrite(path, out)
    return out


imgs = [render("open_palm", fh.open_palm), render("fist", fh.fist),
        render("pinch", fh.pinch), render("one_finger", fh.one_finger)]
grid = np.vstack([np.hstack(imgs[:2]), np.hstack(imgs[2:])])
cv2.imwrite(os.path.join(HERE, "preview_grid.png"), cv2.resize(grid, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
print("ok")
