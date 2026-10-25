"""
J.A.R.V.I.S. gesture interface
==============================
Open palm   -> human heart
Closed fist -> car
Pinch       -> shadow mantis face
One finger  -> butterfly

Run:  python main.py            (python main.py --help for options)
Keys: Q/Esc quit | H toggle HUD | V voice | S screenshot | F fullscreen | D debug | M hologram/plain | 1-4 demo models
"""
import argparse
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

from jarvis import gestures as G
from jarvis import hud
from jarvis.sprites import SPEECH, SPRITE_FOR, Holo
from jarvis.tracker import HandTracker, ensure_model
from jarvis.voice import Voice

ROOT = os.path.dirname(os.path.abspath(__file__))
WINDOW = "J.A.R.V.I.S. - Gesture Interface"
DEMO_KEYS = {ord("1"): "heart", ord("2"): "car", ord("3"): "mantis", ord("4"): "butterfly"}
DETECT_WIDTH = 480  # the hand model works on a tiny image anyway; smaller = faster


def open_camera(index, width, height, mjpg=True):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    if mjpg:  # compressed stream: webcams deliver 30 fps at 720p only in MJPG mode
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the newest frame = less lag
    return cap


class CameraReader:
    """Reads the camera in a background thread so processing never waits on the webcam."""

    def __init__(self, cap):
        self.cap = cap
        self.cond = threading.Condition()
        self.frame = None
        self.count = 0
        self.alive = True
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            ok, frame = self.cap.read()
            with self.cond:
                if not ok:
                    self.alive = False
                    self.cond.notify_all()
                    return
                self.frame = frame
                self.count += 1
                self.cond.notify_all()

    def read(self, last_count, timeout=2.0):
        """Block until a frame newer than `last_count` exists. Returns (frame, count) or (None, count)."""
        with self.cond:
            self.cond.wait_for(lambda: self.count != last_count or not self.alive, timeout)
            if self.count == last_count:
                return None, last_count
            return self.frame, self.count

    def stop(self):
        self.running = False


def main():
    ap = argparse.ArgumentParser(description="Jarvis hand-gesture hologram interface")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--width", type=int, default=1280, help="camera capture width")
    ap.add_argument("--height", type=int, default=720, help="camera capture height")
    ap.add_argument("--render-width", type=int, default=1024,
                    help="width the picture is processed/drawn at (default 1024; lower = faster, 0 = camera size)")
    ap.add_argument("--no-mjpg", action="store_true", help="don't request MJPG from the camera")
    ap.add_argument("--no-voice", action="store_true", help="start with voice off")
    ap.add_argument("--no-hud", action="store_true", help="start with HUD hidden")
    ap.add_argument("--plain-models", action="store_true", help="show the plain coloured images instead of holograms")
    ap.add_argument("--debug", action="store_true", help="show live finger ratios (for tuning)")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--check", action="store_true", help="test camera + model, then exit")
    ap.add_argument("--hands", type=int, default=2, choices=[1, 2], help="max hands to track")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "hand_landmarker.task"))
    ap.add_argument("--assets", default=os.path.join(ROOT, "assets"))
    args = ap.parse_args()

    try:
        holo = Holo(args.assets)
    except FileNotFoundError as e:
        print("[jarvis]", e)
        sys.exit(1)
    if args.plain_models:
        holo.style = "color"

    model_path = ensure_model(args.model)
    try:
        tracker = HandTracker(model_path, max_hands=args.hands)
    except Exception as e:  # noqa: BLE001
        print(f"[jarvis] could not load the hand model: {e}")
        print("         Deleting it so it is downloaded again on the next run.")
        try:
            os.remove(model_path)
        except OSError:
            pass
        sys.exit(1)

    cap = open_camera(args.camera, args.width, args.height, mjpg=not args.no_mjpg)
    if cap is None:
        print(f"[jarvis] cannot open camera {args.camera}. Close other apps using it "
              "(Zoom, Teams, browser tabs) or try:  python main.py --camera 1")
        sys.exit(1)

    if args.check:
        ok, frame = cap.read()
        print(f"[jarvis] opencv {cv2.__version__}")
        print(f"[jarvis] model loaded: {model_path}")
        print(f"[jarvis] camera {args.camera}: {'frame ' + str(frame.shape[1]) + 'x' + str(frame.shape[0]) if ok else 'opened but NO FRAMES'}")
        print("[jarvis] CHECK PASSED" if ok else "[jarvis] CHECK FAILED")
        cap.release()
        tracker.close()
        sys.exit(0 if ok else 1)

    reader = CameraReader(cap)

    voice = Voice(enabled=not args.no_voice)
    if not voice.available and not args.no_voice:
        print("[jarvis] voice disabled (optional: pip install pyttsx3)")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fullscreen = args.fullscreen

    show_hud = not args.no_hud
    debug = args.debug
    demo = None            # forced model name from keys 1-4
    log = deque(maxlen=20)
    log.append((time.strftime("%H:%M:%S"), "SYSTEM ONLINE"))
    voice.say("Jarvis online. Show me your hand.")

    t0 = time.time()
    frames = 0
    fps = 0.0
    last = time.time()
    last_count = 0
    shot_dir = os.path.join(ROOT, "screenshots")

    while True:
        frame, last_count = reader.read(last_count)
        if frame is None:
            if not reader.alive:
                print("[jarvis] camera stopped delivering frames")
                break
            continue  # no new frame yet - keep the window responsive
        frames += 1

        # shrink to the render size (the biggest speed-up: every later step touches fewer pixels)
        if args.render_width and frame.shape[1] > args.render_width:
            h = int(frame.shape[0] * args.render_width / frame.shape[1])
            frame = cv2.resize(frame, (args.render_width, h), interpolation=cv2.INTER_AREA)
        frame = cv2.flip(frame, 1)  # mirror, so it behaves like a mirror
        H, W = frame.shape[:2]
        now = time.time()
        t = now - t0
        inst = 1.0 / max(now - last, 1e-6)
        fps = inst if fps == 0.0 else 0.92 * fps + 0.08 * inst
        last = now

        # detection runs on a small copy; landmarks are normalised so they map straight back
        if W > DETECT_WIDTH:
            small = cv2.resize(frame, (DETECT_WIDTH, int(H * DETECT_WIDTH / W)), interpolation=cv2.INTER_LINEAR)
        else:
            small = frame
        hands = tracker.detect(small)
        frame = hud.filter_frame(frame)
        fx = np.zeros_like(frame)

        seen_keys = set()
        hands_info = []
        active_name = None
        holo_infos = []

        for label, score, lm in hands:
            key = label
            if key in seen_keys:
                key += "2"
            seen_keys.add(key)
            st = holo.state(key)
            st.last_seen = now

            g, info = G.classify(lm, W / H)
            stable = st.smoother.update(g)
            if stable != st.last_gesture:
                if stable != G.UNKNOWN:
                    log.append((time.strftime("%H:%M:%S"), f"{key.upper()}: {G.LABELS[stable]}"))
                    name = SPRITE_FOR.get(stable)
                    if name:
                        voice.say(SPEECH[name])
                st.last_gesture = stable

            hi = holo.draw(frame, st, lm, stable, True, t)
            if hi:
                holo_infos.append(hi)
                active_name = hi["name"]
            n_hand = len(hands_info) + 1
            lab = f"HAND {n_hand} // {key.upper()} // {G.LABELS[stable]}"
            hud.draw_hand(fx, lm, W, H, stable, info["extended"], t, lab, info if debug else None)
            hands_info.append({"gesture": stable, "count": info["count"]})

        # hands that vanished: fade their hologram out
        for key, st in list(holo.states.items()):
            if key not in seen_keys:
                hi = holo.draw(frame, st, None, G.UNKNOWN, False, t)
                if hi:
                    holo_infos.append(hi)
        holo.prune(now)

        if demo:
            holo_infos.append(holo.draw_demo(frame, demo, t))
            active_name = demo

        for hi in holo_infos:
            hud.draw_hologram_frame(fx, hi, t)

        if show_hud:
            hud.draw_hud(frame, fx, t, fps, hands_info, log,
                         {"voice": voice.enabled and voice.available}, active_name)
        else:
            hud.txt(fx, f"{G.LABELS[hands_info[0]['gesture']] if hands_info else 'NO HAND'}", (20, 40), 0.8, hud.CYAN, 2)

        out = hud.compose_fx(frame, fx)
        cv2.imshow(WINDOW, out)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("h"):
            show_hud = not show_hud
        elif k == ord("d"):
            debug = not debug
        elif k == ord("m"):
            style = holo.toggle_style()
            log.append((time.strftime("%H:%M:%S"), f"STYLE: {'HOLOGRAM' if style == 'holo' else 'PLAIN'}"))
        elif k == ord("v"):
            on = voice.toggle()
            log.append((time.strftime("%H:%M:%S"), f"VOICE {'ON' if on else 'OFF'}"))
        elif k == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif k == ord("s"):
            os.makedirs(shot_dir, exist_ok=True)
            path = os.path.join(shot_dir, time.strftime("jarvis_%Y%m%d_%H%M%S.png"))
            cv2.imwrite(path, out)
            log.append((time.strftime("%H:%M:%S"), "SCREENSHOT SAVED"))
            print("[jarvis] saved", path)
        elif k in DEMO_KEYS:
            demo = None if demo == DEMO_KEYS[k] else DEMO_KEYS[k]
        elif k == ord("0"):
            demo = None

        try:  # window closed with the X button (ignore the first frames while it opens)
            if frames > 15 and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    reader.stop()
    time.sleep(0.05)
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
