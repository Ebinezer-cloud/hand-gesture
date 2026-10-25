"""Hologram sprites: loading, per-model animation, and per-hand fade/position state."""
import math
import os
import random
import time

import cv2
import numpy as np

from . import gestures as G

# gesture -> sprite file name (assets/<name>.png)
SPRITE_FOR = {
    G.OPEN_PALM: "heart",
    G.FIST: "car",
    G.PINCH: "mantis",
    G.ONE_FINGER: "butterfly",
}
TITLE = {
    "heart": "HUMAN HEART",
    "car": "VEHICLE MODEL",
    "mantis": "SHADOW MANTIS",
    "butterfly": "BUTTERFLY",
}
SPEECH = {
    "heart": "Cardiac model online.",
    "car": "Vehicle model loaded.",
    "mantis": "Shadow mantis engaged.",
    "butterfly": "Releasing butterfly.",
}


# how much of the model's original colour survives in the hologram (0 = pure cyan)
COLOR_MIX = {"heart": 0.40, "car": 0.18, "mantis": 0.32, "butterfly": 0.30}


def telemetry(name, t):
    """Fake live read-outs shown around each hologram (pure UI flavour)."""
    if name == "heart":
        bpm = 72 + int(3 * math.sin(t * 0.9))
        return [f"HEART RATE  {bpm} BPM", "SPO2  98 %", "RHYTHM  SINUS / NORMAL",
                f"OUTPUT  {4.9 + 0.2 * math.sin(t * 1.1):.1f} L/MIN"]
    if name == "car":
        return ["ENGINE  ONLINE", "POWER  640 HP", "0-100 KM/H  2.9 S", "CHASSIS SCAN  OK"]
    if name == "mantis":
        return ["SPECIES  UNKNOWN", "THREAT  ELEVATED", f"BIO-SIGNATURE  {int(86 + 6 * math.sin(t * 1.7))} %",
                "OPTICS  COMPOUND x2"]
    return ["SPECIES  LEPIDOPTERA", "WINGSPAN  7.4 CM", "FLAP RATE  1.9 HZ", "STATUS  AIRBORNE"]


def build_holo(img, mix):
    """Turn a colour sprite into a Jarvis-style hologram image.
    Returns BGRA: BGR = additive light (premultiplied), A = how strongly it darkens what's behind."""
    bgr = img[..., :3].astype(np.float32) / 255.0
    a_full = img[..., 3].astype(np.float32) / 255.0
    a = np.clip((a_full - 0.35) / 0.65, 0, 1)      # shape only: drops the soft glow baked into the PNGs
    gray8 = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    lum = np.clip(gray8.astype(np.float32) / 255.0 * 1.5, 0, 1)

    cyan = np.array([1.0, 0.80, 0.16], np.float32)        # BGR: light blue-cyan
    edge_col = np.array([1.0, 0.96, 0.60], np.float32)    # near-white cyan
    body = cyan[None, None, :] * (0.16 + 0.55 * lum)[..., None]
    body = (body * (1 - mix) + bgr * mix * 0.85) * a[..., None]   # only inside the shape

    # wireframe edges: internal detail (Canny) + silhouette outline
    masked = (gray8.astype(np.float32) * a).astype(np.uint8)
    edges = cv2.Canny(masked, 28, 100)
    alpha8 = (a * 255).astype(np.uint8)
    sil = cv2.subtract(alpha8, cv2.erode(alpha8, np.ones((5, 5), np.uint8)))
    edges = cv2.dilate(np.maximum(edges, sil), np.ones((2, 2), np.uint8)).astype(np.float32) / 255.0
    edges *= (a > 0.25)
    glow = cv2.GaussianBlur(edges, (0, 0), 5)

    # faint 3D-mesh grid inside the shape
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    grid = (((xx % 16) < 1.5) | ((yy % 16) < 1.5)).astype(np.float32) * 0.20 * a

    light = body * 0.85 + edges[..., None] * edge_col + glow[..., None] * cyan * 0.85 + grid[..., None] * cyan
    cover = np.clip(a * 0.55 + edges + glow * 0.6, 0, 1)
    out = np.dstack([np.clip(light, 0, 1) * 255, cover * 255]).astype(np.uint8)
    return out


def blit_holo(frame, spr, cx, cy, k, t):
    """Additive hologram blit with scanlines, moving highlight band, flicker and a
    bottom-up 'materialise' reveal controlled by k (0..1)."""
    h, w = spr.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + w, frame.shape[1]), min(y0 + h, frame.shape[0])
    if fx1 <= fx0 or fy1 <= fy0 or k <= 0.01:
        return
    s = spr[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    rows = np.arange(fy0 - y0, fy1 - y0, dtype=np.float32)
    rel = rows / max(h, 1)
    scan = 0.78 + 0.22 * (((rows // 2) % 2))                              # fine scanlines
    band_c = (t * 0.55) % 1.6 - 0.3
    band = 1.0 + 1.3 * np.exp(-((rel - band_c) ** 2) / 0.003)              # moving bright band
    reveal = np.clip((k * 1.25 - (1.0 - rel)) * 7.0, 0, 1) if k < 0.999 else 1.0   # bottom-up
    flick = 0.93 + 0.05 * math.sin(t * 53.0) + 0.03 * math.sin(t * 31.0)
    if (t % 3.1) < 0.06:                                                     # rare glitch dip
        flick *= 0.55
    wgt = (scan * band * reveal * flick * min(1.0, k * 1.6)).astype(np.float32)[:, None, None]
    light = s[..., :3].astype(np.float32) * wgt * 1.15
    cover = (s[..., 3:4].astype(np.float32) / 255.0) * 0.38 * min(1.0, k * 1.6)
    roi = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    frame[fy0:fy1, fx0:fx1] = np.clip(roi * (1 - cover) + light, 0, 255).astype(np.uint8)


def rotate_bgra(img, angle):
    if abs(angle) < 0.05:
        return img
    h, w = img.shape[:2]
    diag = int(math.hypot(w, h)) + 2
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[0, 2] += (diag - w) / 2
    m[1, 2] += (diag - h) / 2
    return cv2.warpAffine(img, m, (diag, diag), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def blit(frame, spr, cx, cy, alpha=1.0):
    """Alpha-blend a BGRA sprite centred on (cx, cy), clipped to the frame."""
    h, w = spr.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + w, frame.shape[1]), min(y0 + h, frame.shape[0])
    if fx1 <= fx0 or fy1 <= fy0:
        return
    s = spr[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    a = (s[..., 3:4].astype(np.float32) / 255.0) * alpha
    roi = frame[fy0:fy1, fx0:fx1]
    roi[:] = (s[..., :3].astype(np.float32) * a + roi.astype(np.float32) * (1 - a)).astype(np.uint8)


class Sprite:
    def __init__(self, name, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        self.name = name
        self.img = img
        self.holo = build_holo(img, COLOR_MIX.get(name, 0.3))
        self.aspect = img.shape[1] / img.shape[0]

    def render(self, size, t, style="holo"):
        """Return (BGRA image, alpha multiplier, dx, dy) for animation time t."""
        base = self.holo if style == "holo" else self.img
        if self.aspect >= 1:  # landscape sprite: `size` is the width
            w, h = size, size / self.aspect
        else:                 # portrait sprite: `size` is the height
            w, h = size * self.aspect, size
        sx = sy = 1.0
        dx = dy = rot = 0.0
        a = 1.0
        n = self.name
        if n == "heart":
            beat = abs(math.sin(t * 3.2)) ** 8
            sx = sy = 1.0 + 0.10 * beat + 0.015 * math.sin(t * 1.3)
            dy = 6 * math.sin(t * 1.5)
        elif n == "car":
            dx = 16 * math.sin(t * 1.6)
            dy = 2 * math.sin(t * 38)
            rot = 1.2 * math.sin(t * 1.6)
        elif n == "mantis":
            sx = sy = 1.0 + 0.02 * math.sin(t * 2.0)
            dy = 5 * math.sin(t * 1.1)
        elif n == "butterfly":
            sx = 0.42 + 0.58 * abs(math.cos(t * 6.0))
            dx = 14 * math.sin(t * 1.3)
            dy = 14 * math.sin(t * 2.2)
            rot = 8 * math.sin(t * 1.3)
        if style == "holo" and n != "butterfly":
            sx *= math.cos(0.75 * math.sin(t * 0.7))   # gentle turntable swivel (fake Y-rotation)
        wi, hi = max(2, int(w * sx)), max(2, int(h * sy))
        interp = cv2.INTER_AREA if wi < base.shape[1] else cv2.INTER_LINEAR
        out = cv2.resize(base, (wi, hi), interpolation=interp)
        if n == "mantis" and (t % 1.9) < 0.14:  # glitch burst
            rnd = random.Random(int(t * 40))
            a = 0.75
            for _ in range(4):
                y0 = rnd.randrange(0, max(1, hi - 20))
                y1 = min(hi, y0 + rnd.randrange(8, 40))
                out[y0:y1] = np.roll(out[y0:y1], rnd.randrange(-24, 25), axis=1)
        out = rotate_bgra(out, rot)
        return out, a, dx, dy


class HandState:
    def __init__(self):
        self.smoother = G.GestureSmoother()
        self.shown = None       # sprite name being displayed
        self.alpha = 0.0
        self.cx = self.cy = self.size = None
        self.palm = (0, 0)
        self.last_seen = time.time()
        self.last_gesture = G.UNKNOWN


class Holo:
    def __init__(self, assets_dir):
        self.sprites = {}
        for name in set(SPRITE_FOR.values()):
            path = os.path.join(assets_dir, name + ".png")
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{path} is missing - run:  python tools/generate_assets.py")
            self.sprites[name] = Sprite(name, path)
        self.states = {}
        self.style = "holo"   # "holo" (Jarvis hologram) or "color" (plain images)

    def toggle_style(self):
        self.style = "color" if self.style == "holo" else "holo"
        return self.style

    def _blit(self, frame, spr, cx, cy, k, t):
        if self.style == "holo":
            blit_holo(frame, spr, cx, cy, k, t)
        else:
            blit(frame, spr, cx, cy, k)

    def state(self, key):
        if key not in self.states:
            self.states[key] = HandState()
        return self.states[key]

    # -------------------------------------------------------------- drawing
    def _place(self, st, lm, W, H):
        xs = [p.x * W for p in lm]
        ys = [p.y * H for p in lm]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        span = max(x1 - x0, y1 - y0)
        sp = self.sprites[st.shown]
        size = float(np.clip(span * 1.25, 170, 0.55 * H))
        if sp.aspect >= 1:  # wide models (the car) get more width so they look as big as the tall ones
            size = min(size * 1.5, 0.62 * W)
        cx = (x0 + x1) / 2
        palm = (sum(lm[i].x for i in (0, 5, 9, 13, 17)) / 5 * W,
                sum(lm[i].y for i in (0, 5, 9, 13, 17)) / 5 * H)
        hh = size / sp.aspect if sp.aspect >= 1 else size
        ww = size if sp.aspect >= 1 else size * sp.aspect
        cy = y0 - hh * 0.55 - 12  # float above the hand
        cy = max(cy, hh / 2 + 10)  # keep on screen
        cx = float(np.clip(cx, ww / 2 + 6, W - ww / 2 - 6))
        if st.cx is None:
            st.cx, st.cy, st.size = cx, cy, size
        else:  # smooth follow
            st.cx += (cx - st.cx) * 0.35
            st.cy += (cy - st.cy) * 0.35
            st.size += (size - st.size) * 0.25
        st.palm = palm

    def draw(self, frame, st, lm, stable, seen, t):
        """Update fade state and draw. Returns hologram info for the HUD or None."""
        H, W = frame.shape[:2]
        target = SPRITE_FOR.get(stable) if seen else None

        # fade logic: swap sprite only after the old one has faded out
        if target is not None and target == st.shown:
            st.alpha = min(1.0, st.alpha + 0.14)
        else:
            st.alpha = max(0.0, st.alpha - 0.16)
            if st.alpha <= 0.0:
                st.shown = target
                if target is not None:
                    st.cx = None  # snap to new position when appearing
        if st.shown is None or st.alpha <= 0.0:
            return None

        if seen:
            self._place(st, lm, W, H)
        spr, amul, dx, dy = self.sprites[st.shown].render(st.size, t, self.style)
        eased = st.alpha * st.alpha * (3 - 2 * st.alpha)
        grow = 0.85 + 0.15 * eased  # slight zoom-in while appearing
        if abs(grow - 1.0) > 0.01:
            spr = cv2.resize(spr, (max(2, int(spr.shape[1] * grow)), max(2, int(spr.shape[0] * grow))))
        self._blit(frame, spr, st.cx + dx, st.cy + dy, eased * amul, t)
        return {
            "name": st.shown, "cx": st.cx + dx, "cy": st.cy + dy,
            "w": spr.shape[1], "h": spr.shape[0],
            "alpha": eased, "palm": st.palm, "holo": self.style == "holo",
        }

    def draw_demo(self, frame, name, t):
        """Preview a model in the middle of the screen (keys 1-4)."""
        H, W = frame.shape[:2]
        spr, amul, dx, dy = self.sprites[name].render(0.55 * H, t, self.style)
        self._blit(frame, spr, W / 2 + dx, H / 2 + dy, amul, t)
        return {"name": name, "cx": W / 2 + dx, "cy": H / 2 + dy,
                "w": spr.shape[1], "h": spr.shape[0], "alpha": 1.0, "palm": None,
                "holo": self.style == "holo"}

    def prune(self, now):
        for k in [k for k, s in self.states.items() if now - s.last_seen > 1.0 and s.alpha <= 0.0]:
            del self.states[k]
