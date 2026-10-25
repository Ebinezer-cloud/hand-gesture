# J.A.R.V.I.S. Gesture Interface

Webcam hand tracking (all 5 fingers) with an Iron-Man style HUD. Each gesture summons a
floating hologram that follows your hand.

| Gesture | Hologram |
|---|---|
| Open palm (all fingers out) | Human heart (beating) |
| Closed fist | Car |
| Pinch (thumb + index touching) | Shadow mantis face (glitching) |
| One finger (index up) | Butterfly (flapping) |

The models are shown as **Jarvis-style holograms**: translucent cyan wireframe with a glowing
outline and 3D-mesh grid, scanlines and a moving highlight, flicker, a bottom-up "materialise"
effect, a rotating projector base with rising particles, a scan ring, and live telemetry
call-outs (heart rate, power, threat level, wingspan ...). Press `M` for the plain coloured
images, or start with `--plain-models`.

Also: neon hand skeleton, live finger count (0-5), two-hand tracking, event log, spinning
arc reactor, optional voice ("Cardiac model online."), screenshots.

## Quick start

Needs Python 3.10 - 3.13 (3.12 recommended) and a webcam. If you don't have Python: install 3.12 from python.org and tick "Add python.exe to PATH".

**Windows:** double-click `run.bat` (installs everything, then starts the app; the window stays open if something fails so you can read the message)
**macOS / Linux:** `./run.sh`

First launch creates a virtual environment, installs the packages, and downloads the
~8 MB MediaPipe hand model into `models/` (internet needed once). The download shows progress,
retries, and rejects incomplete files. If it still fails, `run.bat` opens the download link in
your browser: let it finish, press any key in the window, and the file is picked up from your
Downloads folder automatically.

Test the camera and model without starting the app: `run.bat --check` (or `python main.py --check`).

Manual way:

```bash
pip install -r requirements.txt
python tools/generate_assets.py     # only if assets/ is empty
python main.py
```

If package install fails, install Python 3.12, delete the `.venv` folder and run again.

## Keys

| Key | Action |
|---|---|
| `Q` / `Esc` | Quit |
| `H` | Show / hide HUD |
| `V` | Voice on / off (needs `pip install pyttsx3`) |
| `S` | Save screenshot to `screenshots/` |
| `F` | Fullscreen |
| `M` | Switch models between hologram and plain colour |
| `D` | Debug: show live finger ratios beside your hand |
| `1` `2` `3` `4` | Preview heart / car / mantis / butterfly without a hand (press again to hide) |

Options: `python main.py --camera 1 --width 1920 --height 1080 --hands 1 --no-voice --fullscreen`

## Use your own images

Every model is a transparent PNG in `assets/`. Replace a file (keep the name) and restart:

- `heart.png`, `car.png`, `mantis.png`, `butterfly.png`

A real heart photo, a render of your own car, your own character art: all work.
Use PNGs with a transparent background, about 500 px on the long side.

## Tuning the gestures

If a gesture triggers too easily or not enough, run `python main.py --debug` and watch
the numbers beside your hand, then edit the constants at the top of `jarvis/gestures.py`:

| Constant | Meaning |
|---|---|
| `FINGER_EXTENDED_RATIO` | Higher = finger must be straighter to count as "out" |
| `THUMB_OUT_RATIO` | Higher = thumb must stick out further |
| `PINCH_RATIO` | Higher = thumb and index may be farther apart and still pinch |
| `PINCH_INDEX_MIN_RATIO` | Stops a closed fist being read as a pinch |

Tips: good light on your hand, plain background, keep the whole hand in frame, hold a
gesture for ~0.2 s (the smoother ignores flicker).

## Speed

Defaults are already tuned: the camera is read in a background thread, requested in MJPG mode
(many webcams are stuck at 10-15 fps at 720p without it), and hand detection runs on a small copy
of the image. If it is still slow:

- `python main.py --render-width 800` (lower = faster) or `--width 640 --height 480`
- `--hands 1` tracks one hand only
- `H` hides the HUD, `M` switches to plain images (slightly cheaper)
- if the picture looks wrong or the camera misbehaves, try `--no-mjpg`
- close other apps that use the camera or the GPU (Zoom, Teams, browser video calls)

## Troubleshooting

- **Camera won't open:** close Zoom/Teams/browser tabs using it, or try `--camera 1`.
- **Stuck or failing model download:** `run.bat` retries and then opens your browser. Or download `hand_landmarker.task` from
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
  and save it as `models/hand_landmarker.task`.
- **Fist shows as pinch (or reverse):** adjust `PINCH_INDEX_MIN_RATIO` / `PINCH_RATIO`.

## Project layout

```
main.py                 app loop (camera -> tracking -> gesture -> hologram -> HUD)
jarvis/gestures.py      finger + gesture classifier, smoothing
jarvis/tracker.py       MediaPipe Hand Landmarker wrapper + model download
jarvis/sprites.py       hologram loading, animation, fade/follow logic
jarvis/hud.py           HUD drawing (neon glow, reticles, reactor, panels)
jarvis/voice.py         optional text-to-speech
tools/generate_assets.py  draws the four PNGs in assets/
tests/                  gesture tests + offscreen render preview (no camera needed)
```

Run the tests: `python tests/test_gestures.py`, `python tests/smoke_main.py` (full loop, fake camera) and `python tests/render_preview.py`.

## Adding a new gesture

1. Add a constant and a rule in `classify()` (`jarvis/gestures.py`).
2. Map it in `SPRITE_FOR` / `TITLE` / `SPEECH` (`jarvis/sprites.py`) and add `assets/<name>.png`.
3. Add its animation in `Sprite.render()` and an entry in the legend (`jarvis/hud.py`).
