"""
Generates the four hologram sprites used by the gesture app.

    python tools/generate_assets.py

Output (transparent PNGs) goes to ./assets :
    heart.png      - anatomical human heart (open palm)
    car.png        - sports car (closed fist / fist)
    mantis.png     - original dark "shadow mantis" creature face (pinch)
    butterfly.png  - butterfly (one finger)

Want your own artwork? Just drop a PNG with the same file name into ./assets
and the app will use yours instead (see README).
"""
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

S = 1024
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")


# ----------------------------------------------------------------- helpers
def hexc(h, a=255):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


def bez(p0, p1, p2, p3, n=48):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.array(p, dtype=float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def path(*segs):
    pts = []
    for s in segs:
        pts.extend(map(tuple, bez(*s)))
    return pts


def canvas(w=S, h=S):
    return Image.new("RGBA", (w, h), (0, 0, 0, 0))


def poly_mask(size, pts):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in pts], fill=255)
    return m


def gradient(size, box, c1, c2, radial=False):
    """Gradient image (RGBA) covering `box` (x0,y0,x1,y1); outside uses end colours."""
    w, h = size
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if radial:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        r = max((x1 - x0), (y1 - y0)) / 2
        t = np.clip(np.hypot(xx - cx, yy - cy) / max(r, 1), 0, 1)
    else:
        t = np.clip((yy - y0) / max(y1 - y0, 1), 0, 1)
    c1 = np.array(c1, dtype=np.float32)
    c2 = np.array(c2, dtype=np.float32)
    arr = c1[None, None, :] * (1 - t[..., None]) + c2[None, None, :] * t[..., None]
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def fill(img, mask, c1, c2=None, radial=False, box=None):
    if box is None:
        box = mask.getbbox() or (0, 0, img.size[0], img.size[1])
    g = gradient(img.size, box, c1, c2 or c1, radial)
    img.paste(g, (0, 0), mask)


def edge_shade(img, mask, radius=28, dark=0.55, light=None):
    """Darken (and optionally add a rim light to) pixels near the mask edge."""
    inner = np.asarray(mask.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0
    m = np.asarray(mask, dtype=np.float32) / 255.0
    arr = np.asarray(img).astype(np.float32)
    shade = (dark + (1 - dark) * inner)[..., None]
    arr[..., :3] = arr[..., :3] * (1 - m[..., None]) + arr[..., :3] * shade * m[..., None]
    if light is not None:
        rim = np.clip(1.0 - inner, 0, 1) * m
        rim = rim[..., None] * np.array(light[:3], dtype=np.float32)[None, None, :] * 0.35
        arr[..., :3] = np.clip(arr[..., :3] + rim, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def put(base, layer, mask=None):
    """Alpha-composite `layer` onto `base`, optionally limited by an L mask."""
    if mask is not None:
        layer = layer.copy()
        a = np.minimum(np.asarray(layer.split()[-1]), np.asarray(mask))
        layer.putalpha(Image.fromarray(a, "L"))
    return Image.alpha_composite(base, layer)


def tube(img, pts, w0, w1, color, color2=None):
    """Tapered tube drawn as overlapping discs along a poly-line."""
    d = ImageDraw.Draw(img)
    pts = [tuple(p) for p in pts]
    n = len(pts)
    for i, (x, y) in enumerate(pts):
        t = i / max(n - 1, 1)
        w = w0 + (w1 - w0) * t
        c = color if color2 is None else tuple(int(color[k] * (1 - t) + color2[k] * t) for k in range(4))
        d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=c)


def glow(img, mask_or_layer, color, radius=30, strength=0.8):
    """Add a coloured glow behind `img` from the alpha of `mask_or_layer`."""
    a = mask_or_layer.split()[-1] if mask_or_layer.mode == "RGBA" else mask_or_layer
    a = a.filter(ImageFilter.GaussianBlur(radius)).point(lambda v: int(v * strength))
    g = Image.new("RGBA", img.size, color[:3] + (0,))
    g.putalpha(a)
    return Image.alpha_composite(g, img)


def trim_and_save(img, name, max_side=512, pad=24):
    bbox = img.getbbox()
    img = img.crop(bbox)
    w, h = img.size
    scale = max_side / max(w, h)
    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    out = Image.new("RGBA", (img.size[0] + pad * 2, img.size[1] + pad * 2), (0, 0, 0, 0))
    out.paste(img, (pad, pad))
    os.makedirs(OUT, exist_ok=True)
    out.save(os.path.join(OUT, name))
    print("wrote", name, out.size)


# ------------------------------------------------------------------- HEART
def make_heart():
    img = canvas()
    d = ImageDraw.Draw(img)
    cx, cy, th = 500, 560, math.radians(24)

    def heart_pt(t):
        x = 15.5 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        x, y = x * 22, y * 22
        return (cx + x * math.cos(th) - y * math.sin(th), cy + x * math.sin(th) + y * math.cos(th))

    outline = [heart_pt(t) for t in np.linspace(0, 2 * math.pi, 400)]

    # --- great vessels (drawn first so the heart body overlaps their bases)
    blue_dark, blue = hexc("#34498c"), hexc("#5b73bd")
    tube(img, path(((400, 400), (385, 330), (390, 220), (395, 110))), 62, 56, blue_dark, blue)   # superior vena cava
    tube(img, path(((340, 650), (330, 720), (335, 790), (330, 880))), 56, 50, blue_dark, blue)   # inferior vena cava
    # pulmonary trunk + branches
    tube(img, path(((590, 390), (610, 300), (660, 250), (740, 235))), 58, 48, hexc("#4b62a8"), hexc("#6d83cc"))
    tube(img, path(((740, 235), (790, 225), (830, 200), (880, 190))), 40, 30, hexc("#6d83cc"), hexc("#8ea0dd"))
    tube(img, path(((690, 250), (700, 330), (730, 380), (790, 410))), 34, 26, hexc("#6d83cc"), hexc("#8ea0dd"))
    # aorta + arch + branches
    aorta_c1, aorta_c2 = hexc("#a8242a"), hexc("#d4504c")
    tube(img, path(((500, 420), (490, 300), (500, 170), (590, 130)), ((590, 130), (660, 105), (720, 170), (735, 300))), 78, 62, aorta_c1, aorta_c2)
    for (a, b) in [((520, 150), (505, 40)), ((575, 120), (580, 30)), ((640, 125), (665, 50))]:
        tube(img, path((a, ((a[0] + b[0]) / 2, a[1] - 20), ((a[0] + b[0]) / 2, b[1] + 20), b)), 36, 28, hexc("#b83a3a"), hexc("#d9605a"))
    # pulmonary veins (entering from the right)
    for (a, b) in [((720, 480), (880, 450)), ((710, 540), (890, 560))]:
        tube(img, path((a, (a[0] + 60, a[1] - 10), (b[0] - 40, b[1] + 5), b)), 34, 30, hexc("#b43b3f"), hexc("#d8645f"))

    # --- heart body
    mask = poly_mask(img.size, outline)
    body = canvas()
    fill(body, mask, hexc("#d23a3a"), hexc("#6b0d17"), box=(200, 260, 800, 880))

    # right ventricle / atrium tint (viewer's left side of the groove)
    groove = path(((560, 330), (540, 450), (470, 600), (330, 800)))
    left_region = [(0, 0), (560, 0)] + groove + [(330, 1024), (0, 1024)]
    rv = ImageChops_multiply_mask(mask, poly_mask(img.size, left_region))
    tint = Image.new("RGBA", img.size, hexc("#a8385a", 95))
    body = put(body, tint, rv)

    # fat / muscle texture: faint striations following the ventricle curvature
    tex = canvas()
    td = ImageDraw.Draw(tex)
    rng = np.random.default_rng(7)
    for _ in range(70):
        x0 = rng.uniform(300, 760)
        y0 = rng.uniform(400, 820)
        l = rng.uniform(40, 110)
        a = th + rng.uniform(-0.35, 0.35) + math.pi / 2
        td.line([(x0, y0), (x0 + math.cos(a) * l, y0 + math.sin(a) * l)], fill=(70, 5, 12, 38), width=int(rng.uniform(3, 8)))
    tex = tex.filter(ImageFilter.GaussianBlur(2))
    body = put(body, tex, mask)

    body = edge_shade(body, mask, radius=34, dark=0.45, light=hexc("#ff7a6a"))

    # coronary vessels (arteries red, veins blue) over the surface
    bd = ImageDraw.Draw(body)
    for pts, col, w in [
        (path(((545, 380), (520, 470), (450, 600), (350, 760))), hexc("#f26b4c"), 11),
        (path(((545, 380), (610, 450), (640, 560), (610, 700))), hexc("#f26b4c"), 9),
        (path(((530, 460), (600, 520), (640, 620), (600, 720))), hexc("#8d2a35"), 7),
        (path(((500, 520), (440, 540), (400, 610), (385, 690))), hexc("#6074b8"), 7),
        (path(((470, 410), (420, 440), (380, 480), (355, 540))), hexc("#6074b8"), 8),
        (path(((600, 620), (680, 600), (720, 540), (730, 470))), hexc("#f26b4c"), 7),
    ]:
        bd.line([tuple(p) for p in pts], fill=col, width=w, joint="curve")
    # coronary groove (darker)
    bd.line([tuple(p) for p in groove], fill=hexc("#4a0710", 170), width=14, joint="curve")
    body = Image.composite(body, canvas(), mask)

    img = Image.alpha_composite(img, body)

    # specular highlights
    hl = canvas()
    hd = ImageDraw.Draw(hl)
    hd.ellipse([560, 480, 680, 640], fill=(255, 255, 255, 90))
    hd.ellipse([330, 400, 420, 470], fill=(255, 255, 255, 60))
    hl = hl.filter(ImageFilter.GaussianBlur(22))
    img = put(img, hl, mask)

    # dark outline
    ImageDraw.Draw(img).line(outline + [outline[0]], fill=hexc("#2a050a", 230), width=6, joint="curve")
    img = glow(img, img, hexc("#ff3b3b"), radius=26, strength=0.55)
    trim_and_save(img, "heart.png")


def ImageChops_multiply_mask(a, b):
    return Image.fromarray(np.minimum(np.asarray(a), np.asarray(b)), "L")


# --------------------------------------------------------------------- CAR
def make_car():
    W, H = 1024, 600
    img = canvas(W, H)

    # ground shadow
    sh = canvas(W, H)
    ImageDraw.Draw(sh).ellipse([90, 470, 950, 540], fill=(0, 0, 0, 150))
    img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(14)))

    body_pts = path(
        ((90, 400), (85, 370), (80, 340), (82, 315)),
        ((82, 315), (120, 300), (200, 292), (280, 288)),
        ((280, 288), (330, 225), (390, 178), (470, 170)),
        ((470, 170), (560, 165), (640, 200), (715, 280)),
        ((715, 280), (800, 290), (900, 300), (950, 335)),
        ((950, 335), (975, 350), (982, 378), (970, 408)),
    ) + [(930, 428), (120, 428)]
    mask = poly_mask((W, H), body_pts)
    md = ImageDraw.Draw(mask)
    for wx in (260, 780):  # wheel arches cut out of the body
        md.ellipse([wx - 96, 428 - 96, wx + 96, 428 + 96], fill=0)

    car = canvas(W, H)
    fill(car, mask, hexc("#3d8bff"), hexc("#0b2a80"), box=(80, 165, 980, 430))
    # beltline highlight + lower shade
    cd = ImageDraw.Draw(car)
    cd.line([(110, 335), (900, 322)], fill=(255, 255, 255, 70), width=5)
    cd.line([(120, 395), (930, 400)], fill=(5, 15, 60, 120), width=10)
    car = edge_shade(car, mask, radius=18, dark=0.6, light=hexc("#9fd0ff"))
    car = Image.composite(car, canvas(W, H), mask)

    cd = ImageDraw.Draw(car)
    # windows
    win1 = [(305, 285), (372, 205), (455, 196), (455, 285)]
    win2 = [(478, 196), (560, 198), (640, 240), (692, 285), (478, 285)]
    for wp in (win1, win2):
        wm = poly_mask((W, H), wp)
        fill(car, wm, hexc("#b9e4ff"), hexc("#2c5d8f"), box=(300, 190, 700, 290))
        cd.polygon(wp, outline=hexc("#06142e"), width=4)
    cd.line([(466, 196), (466, 405)], fill=hexc("#051233", 200), width=4)  # door seam
    cd.rounded_rectangle([505, 312, 560, 326], radius=6, fill=hexc("#cfe6ff"))  # handle
    # lights
    cd.ellipse([928, 335, 972, 366], fill=hexc("#fff6c4"))
    cd.polygon([(88, 322), (110, 318), (112, 348), (86, 350)], fill=hexc("#ff2a3a"))
    cd.polygon([(940, 392), (975, 390), (972, 408), (935, 410)], fill=hexc("#0a1740"))  # front splitter

    img = Image.alpha_composite(img, car)

    # wheels
    wheels = canvas(W, H)
    wd = ImageDraw.Draw(wheels)
    for wx in (260, 780):
        wy = 428
        wd.ellipse([wx - 84, wy - 84, wx + 84, wy + 84], fill=hexc("#0c0f16"))
        wd.ellipse([wx - 60, wy - 60, wx + 60, wy + 60], fill=hexc("#aab6c8"))
        rim = poly_mask((W, H), [(wx + 60 * math.cos(a), wy + 60 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 60)])
        fill(wheels, rim, hexc("#e9f1ff"), hexc("#6b7890"), box=(wx - 60, wy - 60, wx + 60, wy + 60))
        for k in range(5):
            a = k * 2 * math.pi / 5 + 0.3
            wd.line([(wx, wy), (wx + 58 * math.cos(a), wy + 58 * math.sin(a))], fill=hexc("#1b2233"), width=11)
        wd.ellipse([wx - 14, wy - 14, wx + 14, wy + 14], fill=hexc("#2a3347"))
        wd.ellipse([wx - 5, wy - 5, wx + 5, wy + 5], fill=hexc("#cfd9ea"))
    img = Image.alpha_composite(img, wheels)
    img = glow(img, img, hexc("#33a0ff"), radius=24, strength=0.5)
    trim_and_save(img, "car.png")


# ------------------------------------------------------------------ MANTIS
def make_mantis():
    img = canvas()

    right = path(
        ((512, 100), (600, 100), (720, 170), (760, 300)),
        ((760, 300), (790, 400), (760, 520), (690, 620)),
        ((690, 620), (650, 680), (580, 730), (512, 742)),
    )
    left = [(S - x, y) for (x, y) in reversed(right)]
    head = right + left
    mask = poly_mask(img.size, head)

    # antennae / horns (tapered, curved)
    horns = canvas()
    for sgn in (1, -1):
        pts = path(((512 + sgn * 130, 150), (512 + sgn * 210, 50), (512 + sgn * 300, 20), (512 + sgn * 380, 50)))
        tube(horns, pts, 46, 4, hexc("#252a4d"), hexc("#7a82c8"))
    img = Image.alpha_composite(img, horns)

    skull = canvas()
    fill(skull, mask, hexc("#2b3160"), hexc("#06070f"), box=(250, 100, 780, 745))
    # armour plates: faint panel lines
    sd = ImageDraw.Draw(skull)
    for pts in [
        path(((330, 250), (300, 330), (320, 430), (380, 520))),
        path(((S - 330, 250), (S - 300, 330), (S - 320, 430), (S - 380, 520))),
        path(((420, 560), (450, 620), (480, 660), (512, 690))),
        path(((S - 420, 560), (S - 450, 620), (S - 480, 660), (S - 512, 690))),
    ]:
        sd.line([tuple(p) for p in pts], fill=hexc("#8d96e6", 90), width=4, joint="curve")
    skull = edge_shade(skull, mask, radius=30, dark=0.5, light=hexc("#7f8cff"))
    skull = Image.composite(skull, canvas(), mask)
    img = Image.alpha_composite(img, skull)
    d = ImageDraw.Draw(img)
    d.line(head + [head[0]], fill=hexc("#05060d"), width=8, joint="curve")

    # central ridge plate
    ridge = [(512, 120), (548, 300), (540, 520), (512, 560), (484, 520), (476, 300)]
    rm = poly_mask(img.size, ridge)
    fill(img, rm, hexc("#6a73c2"), hexc("#1a1e44"), box=(470, 120, 555, 560))
    d.line(ridge + [ridge[0]], fill=hexc("#05060d"), width=4)

    # compound eyes with glow
    eyes = canvas()
    for ex in (385, S - 385):
        ey, er = 365, 108
        em = Image.new("L", img.size, 0)
        ImageDraw.Draw(em).ellipse([ex - er, ey - er, ex + er, ey + er], fill=255)
        fill(eyes, em, hexc("#fff0a8"), hexc("#7a1c00"), radial=True, box=(ex - er, ey - er, ex + er, ey + er))
        ed = ImageDraw.Draw(eyes)
        for r in range(24, er, 24):  # facet rings
            ed.ellipse([ex - r, ey - r, ex + r, ey + r], outline=hexc("#5a1400", 150), width=2)
        for k in range(12):  # facet spokes
            a = k * math.pi / 6
            ed.line([(ex, ey), (ex + er * math.cos(a), ey + er * math.sin(a))], fill=hexc("#5a1400", 120), width=2)
        ed.ellipse([ex - er, ey - er, ex + er, ey + er], outline=hexc("#05060d"), width=7)
        ed.ellipse([ex - 60, ey - 62, ex - 28, ey - 34], fill=(255, 255, 255, 190))  # highlight
    img = glow(img, eyes, hexc("#ffa31a"), radius=34, strength=0.9)
    img = Image.alpha_composite(img, eyes)

    # mouth slit + mandibles
    d = ImageDraw.Draw(img)
    d.ellipse([462, 630, 562, 745], fill=hexc("#07040c"))
    d.ellipse([462, 630, 562, 745], outline=hexc("#05060d"), width=5)
    man = canvas()
    for sgn in (1, -1):
        pts = path(((512 + sgn * 90, 640), (512 + sgn * 190, 720), (512 + sgn * 150, 840), (512 + sgn * 40, 930)))
        tube(man, pts, 62, 4, hexc("#f2eedc"), hexc("#8a8672"))
        pts2 = path(((512 + sgn * 40, 700), (512 + sgn * 70, 770), (512 + sgn * 50, 820), (512 + sgn * 20, 850)))
        tube(man, pts2, 24, 3, hexc("#cfcbb6"), hexc("#7a7662"))
    # outline the mandibles by drawing a dark blurred copy underneath
    dark = Image.new("RGBA", img.size, (5, 6, 13, 0))
    dark.putalpha(man.split()[-1].filter(ImageFilter.MaxFilter(9)))
    img = Image.alpha_composite(img, dark)
    img = Image.alpha_composite(img, man)

    # little cyan "scanner" nodes on the cheeks (Jarvis flavour)
    nodes = canvas()
    nd = ImageDraw.Draw(nodes)
    for (x, y) in [(318, 520), (S - 318, 520), (345, 560), (S - 345, 560)]:
        nd.ellipse([x - 7, y - 7, x + 7, y + 7], fill=hexc("#39e6ff"))
    img = glow(img, nodes, hexc("#39e6ff"), radius=10, strength=1.0)
    img = Image.alpha_composite(img, nodes)

    img = glow(img, img, hexc("#6b5bff"), radius=30, strength=0.45)
    trim_and_save(img, "mantis.png")


# --------------------------------------------------------------- BUTTERFLY
def make_butterfly():
    left = canvas()

    fore = path(
        ((505, 490), (430, 300), (280, 130), (110, 150)),
        ((110, 150), (60, 160), (70, 260), (110, 360)),
        ((110, 360), (170, 470), (330, 520), (505, 520)),
    )
    hind = path(
        ((505, 520), (330, 520), (180, 560), (190, 690)),
        ((190, 690), (200, 800), (330, 830), (400, 760)),
        ((400, 760), (450, 720), (490, 650), (505, 600)),
    )
    dark = hexc("#14090a")
    dots = []
    for pts, c1, c2, box in [
        (fore, hexc("#ffb02e"), hexc("#e8480c"), (60, 130, 505, 520)),
        (hind, hexc("#ffd23f"), hexc("#f0700c"), (180, 520, 505, 830)),
    ]:
        m = poly_mask(left.size, pts)
        fill(left, m, c1, c2, box=box)
        arr = np.asarray(m)
        eroded = cv2.erode(arr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (47, 47)))
        border = Image.fromarray(np.clip(arr.astype(int) - eroded.astype(int), 0, 255).astype(np.uint8), "L")
        left.paste(Image.new("RGBA", left.size, dark), (0, 0), border)
        # veins
        vd = ImageDraw.Draw(left)
        ox, oy = 505, (500 if pts is fore else 540)
        for k in range(7):
            idx = int(len(pts) * (0.12 + 0.76 * k / 6))
            tx, ty = pts[idx]
            mx, my = (ox + tx) / 2, (oy + ty) / 2 - 10
            vd.line([(ox, oy), (mx, my), (tx * 0.9 + ox * 0.1, ty * 0.9 + oy * 0.1)], fill=hexc("#14090a", 190), width=5, joint="curve")
        # white spots along the outer border
        cxm = sum(p[0] for p in pts) / len(pts)
        cym = sum(p[1] for p in pts) / len(pts)
        for k in range(10):
            idx = int(len(pts) * (0.22 + 0.5 * k / 9)) if pts is fore else int(len(pts) * (0.2 + 0.5 * k / 9))
            px, py = pts[idx]
            px, py = px + 0.055 * (cxm - px), py + 0.055 * (cym - py)
            dots.append((px, py, 9 if k % 2 == 0 else 6))
        ImageDraw.Draw(left).line(pts + [pts[0]], fill=dark, width=7, joint="curve")
    dd = ImageDraw.Draw(left)
    for (x, y, r) in dots:
        dd.ellipse([x - r, y - r, x + r, y + r], fill=hexc("#fff8e8"))

    img = Image.alpha_composite(left, ImageOps.mirror(left))

    # body, head, antennae
    d = ImageDraw.Draw(img)
    body = canvas()
    bm = Image.new("L", img.size, 0)
    ImageDraw.Draw(bm).ellipse([497, 430, 527, 760], fill=255)
    fill(body, bm, hexc("#3a2418"), hexc("#0c0605"), box=(497, 430, 527, 760))
    body = edge_shade(body, bm, radius=5, dark=0.6)
    body = Image.composite(body, canvas(), bm)
    img = Image.alpha_composite(img, body)
    d = ImageDraw.Draw(img)
    d.ellipse([490, 392, 534, 436], fill=hexc("#1a0d08"))
    for sgn in (1, -1):
        pts = path(((512 + sgn * 6, 400), (512 + sgn * 40, 340), (512 + sgn * 80, 290), (512 + sgn * 120, 255)))
        d.line([tuple(p) for p in pts], fill=hexc("#1a0d08"), width=6, joint="curve")
        ex, ey = pts[-1]
        d.ellipse([ex - 9, ey - 9, ex + 9, ey + 9], fill=hexc("#1a0d08"))

    img = glow(img, img, hexc("#ffb02e"), radius=26, strength=0.5)
    trim_and_save(img, "butterfly.png")


if __name__ == "__main__":
    which = set(sys.argv[1:]) or {"heart", "car", "mantis", "butterfly"}
    if "heart" in which:
        make_heart()
    if "car" in which:
        make_car()
    if "mantis" in which:
        make_mantis()
    if "butterfly" in which:
        make_butterfly()
