"""Iron-Man style HUD: tint, vignette, scanlines, neon glow, reticles, reactor, panels."""
import math
import time

import cv2
import numpy as np

from . import gestures as G
from .tracker import HAND_CONNECTIONS

# BGR colours
CYAN = (255, 225, 60)
CYAN_DIM = (170, 140, 30)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
RED = (70, 70, 255)
GREEN = (120, 255, 120)

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

_cache = {}


def _mask_for(shape):
    """Precomputed vignette x scanline multiplier (uint8 BGR)."""
    key = shape[:2]
    if key in _cache:
        return _cache[key]
    h, w = key
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    vig = np.clip(1.15 - 0.55 * d ** 2, 0.35, 1.0)
    scan = np.ones((h, 1), np.float32)
    scan[::4] = 0.86
    m = (vig * scan * 255).astype(np.uint8)
    m3 = cv2.merge([m, m, m])
    _cache[key] = m3
    return m3


def txt(img, s, org, scale=0.5, color=CYAN, thick=1):
    cv2.putText(img, s, (int(org[0]), int(org[1])), FONT, scale, color, thick, AA)


def scaled(color, k):
    return tuple(int(c * k) for c in color)


def filter_frame(frame):
    """Cool blue tint + vignette + scanlines on the camera image."""
    key = ("tint", frame.shape)
    if key not in _cache:
        t = np.empty_like(frame)
        t[:] = (70, 38, 8)
        _cache[key] = t
    out = cv2.addWeighted(frame, 0.86, _cache[key], 0.14, 0)
    return cv2.multiply(out, _mask_for(frame.shape), scale=1 / 255.0)


def compose_fx(frame, fx):
    """Add the neon layer with a soft glow."""
    h, w = fx.shape[:2]
    small = cv2.resize(fx, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    glow = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    out = cv2.add(frame, cv2.convertScaleAbs(glow, alpha=1.6))
    return cv2.add(out, fx)


def panel(frame, x, y, w, h, alpha=0.45):
    """Darkened translucent rectangle behind a panel."""
    x1, y1 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    roi = frame[y:y1, x:x1]
    roi[:] = (roi * (1 - alpha)).astype(np.uint8)


def brackets(fx, x0, y0, x1, y1, color=CYAN, ln=22, th=2):
    for (x, y, sx, sy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        cv2.line(fx, (x, y), (x + sx * ln, y), color, th, AA)
        cv2.line(fx, (x, y), (x, y + sy * ln), color, th, AA)


def reactor(fx, cx, cy, t, r=62):
    cv2.circle(fx, (cx, cy), r, CYAN_DIM, 1, AA)
    cv2.circle(fx, (cx, cy), int(r * 0.62), CYAN, 1, AA)
    for i in range(48):
        a = i * math.pi / 24
        l = 10 if i % 4 == 0 else 5
        p0 = (cx + (r + 2) * math.cos(a), cy + (r + 2) * math.sin(a))
        p1 = (cx + (r + 2 + l) * math.cos(a), cy + (r + 2 + l) * math.sin(a))
        cv2.line(fx, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), CYAN_DIM, 1, AA)
    for k, (rad, speed, span, col) in enumerate([(r - 8, 60, 70, CYAN), (int(r * 0.78), -90, 50, ORANGE), (int(r * 0.5), 140, 110, CYAN)]):
        base = (t * speed) % 360
        for j in range(3):
            cv2.ellipse(fx, (cx, cy), (rad, rad), 0, base + j * 120, base + j * 120 + span, col, 3, AA)
    pulse = 0.6 + 0.4 * math.sin(t * 3)
    cv2.circle(fx, (cx, cy), int(r * 0.22), scaled(WHITE, pulse), -1, AA)
    cv2.circle(fx, (cx, cy), int(r * 0.30), CYAN, 2, AA)


def draw_hand(fx, lm, W, H, gesture, ext, t, label, debug_info=None):
    """Neon skeleton + reticle + label for one hand."""
    pts = [(int(p.x * W), int(p.y * H)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(fx, pts[a], pts[b], CYAN, 2, AA)
    tips = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}
    for i, p in enumerate(pts):
        if i in tips:
            on = ext.get(tips[i], False)
            cv2.circle(fx, p, 9 if on else 6, ORANGE if on else CYAN_DIM, -1 if on else 2, AA)
            if on:
                cv2.circle(fx, p, 15, ORANGE, 1, AA)
        else:
            cv2.circle(fx, p, 3, WHITE, -1, AA)

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = 26
    x0, y0, x1, y1 = min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad
    brackets(fx, x0, y0, x1, y1, CYAN, 26, 2)
    # spinning reticle on the palm
    px = int(sum(pts[i][0] for i in (0, 5, 9, 13, 17)) / 5)
    py = int(sum(pts[i][1] for i in (0, 5, 9, 13, 17)) / 5)
    rad = int(max(x1 - x0, y1 - y0) * 0.22)
    for j in range(4):
        cv2.ellipse(fx, (px, py), (rad, rad), 0, (t * 120) % 360 + j * 90, (t * 120) % 360 + j * 90 + 40, ORANGE, 2, AA)
    cv2.circle(fx, (px, py), 3, ORANGE, -1, AA)
    txt(fx, label, (x0, max(y0 - 8, 14)), 0.5, CYAN, 1)
    if debug_info:
        r = debug_info["ratios"]
        lines = [f"pinch {debug_info['pinch']:.2f}"] + [f"{k[:3]} {v:.2f}" for k, v in r.items()]
        for i, s in enumerate(lines):
            txt(fx, s, (x1 + 8, y0 + 16 + i * 16), 0.42, GREEN, 1)
    return px, py


def draw_hologram_frame(fx, info, t):
    """Projector base, scan ring, particles, telemetry call-outs and caption around the model."""
    from .sprites import TITLE, telemetry
    a = info["alpha"]
    if a < 0.05:
        return
    fh, fw = fx.shape[:2]
    col = scaled(CYAN, a)
    dim = scaled(CYAN_DIM, a)
    amber = scaled(ORANGE, a)
    cx, cy = int(info["cx"]), int(info["cy"])
    w, h = info["w"] // 2, info["h"] // 2
    holo = info.get("holo", True)

    brackets(fx, cx - w, cy - h, cx + w, cy + h, dim, 18, 1)
    if info["palm"] is not None:  # beam down to the hand
        pxy = (int(info["palm"][0]), int(info["palm"][1]))
        cv2.line(fx, (cx, cy + h + 10), pxy, dim, 1, AA)
        cv2.circle(fx, pxy, 5, amber, 1, AA)

    if holo:
        # ---- projector base: concentric rotating rings under the model
        base_y = cy + h + 10
        rx = int(max(w, 60) * 0.85)
        ry = max(8, int(rx * 0.16))
        for k, (sc, spd, c) in enumerate([(1.0, 40, col), (0.72, -70, amber), (0.45, 110, col)]):
            r1, r2 = int(rx * sc), max(3, int(ry * sc))
            cv2.ellipse(fx, (cx, base_y), (r1, r2), 0, 0, 360, dim, 1, AA)
            for j in range(3):
                ang = (t * spd + j * 120) % 360
                cv2.ellipse(fx, (cx, base_y), (r1, r2), 0, ang, ang + 70, c, 2, AA)
        cv2.line(fx, (cx - rx, base_y), (cx - int(w * 0.9), cy + h - 4), scaled(CYAN_DIM, a * 0.6), 1, AA)
        cv2.line(fx, (cx + rx, base_y), (cx + int(w * 0.9), cy + h - 4), scaled(CYAN_DIM, a * 0.6), 1, AA)

        # ---- scan ring sweeping up through the model
        f = (t * 0.55) % 1.6 - 0.3
        sy = int(cy - h + f * 2 * h)
        if cy - h <= sy <= cy + h:
            cv2.ellipse(fx, (cx, sy), (int(w * 1.02), max(5, int(w * 0.12))), 0, 0, 360, col, 1, AA)

        # ---- particles rising from the base
        for i in range(18):
            speed = 0.16 + 0.26 * ((i * 0.37) % 1)
            ph = (t * speed + (i * 0.618) % 1) % 1
            px = cx + int((((i * 0.7548) % 1) - 0.5) * 1.6 * rx + 9 * math.sin(t * 2 + i))
            py = int(base_y - ph * (2 * h + 40))
            cv2.circle(fx, (px, py), 1 if i % 3 else 2, scaled(CYAN, a * (1 - ph)), -1, AA)

        # ---- telemetry call-outs with leader lines (typed in as the model appears)
        shown = min(1.0, max(0.0, (a - 0.55) / 0.45))
        for i, line in enumerate(telemetry(info["name"], t)):
            side = -1 if i % 2 == 0 else 1
            row = i // 2
            ax, ay = cx + side * int(w * 0.42), int(cy - h * 0.30 + row * h * 0.62)
            ly = int(np.clip(cy - h * 0.55 + row * h * 0.95, 70, fh - 130))
            (tw, _), _ = cv2.getTextSize(line, FONT, 0.42, 1)
            lx = cx + side * (w + 34)
            x_text = int(min(lx, fw - tw - 24)) if side > 0 else int(max(lx - tw, 24))
            elbow_x = x_text - 10 if side > 0 else x_text + tw + 10
            far_x = x_text + tw if side > 0 else x_text
            pts = np.array([(ax, ay), (elbow_x, ly + 6), (far_x, ly + 6)], np.int32)
            cv2.polylines(fx, [pts], False, dim, 1, AA)
            cv2.circle(fx, (ax, ay), 3, amber, -1, AA)
            txt(fx, line[:int(len(line) * shown + 0.999)], (x_text, ly), 0.42, col, 1)
    else:
        f = (t * 0.55) % 1.6 - 0.3
        sy = int(cy - h + f * 2 * h)
        if cy - h <= sy <= cy + h:
            cv2.line(fx, (cx - w, sy), (cx + w, sy), scaled(CYAN, a * 0.5), 1, AA)

    name = TITLE.get(info["name"], info["name"].upper())
    lock = f"MODEL LOCK {int(94 + 5 * math.sin(t * 2))}%"
    ty = cy - h - 26 if cy - h - 40 > 10 else cy - h + 22   # caption above, or inside the top
    txt(fx, name, (cx - w + 4, ty), 0.55, amber, 1)
    txt(fx, lock, (cx - w + 4, ty + 17), 0.4, col, 1)


def draw_hud(frame, fx, t, fps, hands_info, log, flags, active_name):
    """Static HUD: title, panels, legend, reactor, log. hands_info: list of dicts."""
    H, W = frame.shape[:2]
    # corner frame
    m = 14
    brackets(fx, m, m, W - m, H - m, CYAN, 40, 2)
    # moving scan band
    y = int((t * 160) % H)
    cv2.line(fx, (m, y), (W - m, y), (60, 50, 10), 2, AA)

    # title
    panel(frame, 26, 24, 330, 62)
    txt(fx, "J.A.R.V.I.S.", (40, 58), 0.95, CYAN, 2)
    txt(fx, "GESTURE RECOGNITION INTERFACE", (42, 78), 0.4, CYAN_DIM, 1)

    # status panel (left)
    lines = [
        ("SYSTEM", "ONLINE", GREEN),
        ("CAMERA", f"{W}x{H}", CYAN),
        ("FPS", f"{fps:4.1f}", CYAN),
        ("HANDS", str(len(hands_info)), ORANGE if hands_info else CYAN_DIM),
        ("VOICE", "ON" if flags.get("voice") else "OFF", GREEN if flags.get("voice") else CYAN_DIM),
    ]
    for i, hi in enumerate(hands_info[:2]):
        lines.append((f"H{i + 1} GESTURE", G.LABELS[hi["gesture"]], ORANGE))
        lines.append((f"H{i + 1} FINGERS", str(hi["count"]), ORANGE))
    ph = 20 + 22 * len(lines)
    panel(frame, 26, 100, 300, ph)
    cv2.rectangle(fx, (26, 100), (326, 100 + ph), CYAN_DIM, 1, AA)
    for i, (k, v, c) in enumerate(lines):
        txt(fx, k, (40, 126 + i * 22), 0.48, CYAN_DIM, 1)
        txt(fx, v, (190, 126 + i * 22), 0.5, c, 1)

    # reactor + log (right)
    reactor(fx, W - 100, 110, t)
    log_lines = list(log)[-6:]
    panel(frame, W - 330, 200, 304, 30 + 20 * max(len(log_lines), 1))
    cv2.rectangle(fx, (W - 330, 200), (W - 26, 200 + 30 + 20 * max(len(log_lines), 1)), CYAN_DIM, 1, AA)
    txt(fx, "EVENT LOG", (W - 318, 220), 0.45, CYAN, 1)
    for i, (ts, msg) in enumerate(log_lines):
        txt(fx, f"{ts} {msg}"[:36], (W - 318, 242 + i * 20), 0.4, CYAN_DIM if i < len(log_lines) - 1 else WHITE, 1)

    # legend (bottom)
    items = [
        (G.OPEN_PALM, "OPEN PALM", "HEART", "heart"),
        (G.FIST, "FIST", "CAR", "car"),
        (G.PINCH, "PINCH", "MANTIS", "mantis"),
        (G.ONE_FINGER, "ONE FINGER", "BUTTERFLY", "butterfly"),
    ]
    bw, bh, gap = 190, 48, 12
    x = (W - (bw * 4 + gap * 3)) // 2
    y = H - bh - 44
    for _, gl, ml, key in items:
        on = key == active_name
        col = ORANGE if on else CYAN_DIM
        panel(frame, x, y, bw, bh, 0.55)
        cv2.rectangle(fx, (x, y), (x + bw, y + bh), col, 2 if on else 1, AA)
        txt(fx, gl, (x + 12, y + 20), 0.48, col, 1)
        txt(fx, "-> " + ml, (x + 12, y + 39), 0.5, WHITE if on else CYAN_DIM, 1)
        x += bw + gap

    # footer
    now = time.strftime("%H:%M:%S  %d %b %Y")
    txt(fx, now, (30, H - 24), 0.5, CYAN, 1)
    hint = "Q quit  H hud  M style  V voice  S shot  F fullscreen  1-4 demo"
    (tw, _), _ = cv2.getTextSize(hint, FONT, 0.42, 1)
    txt(fx, hint, (W - tw - 30, H - 24), 0.42, CYAN_DIM, 1)
