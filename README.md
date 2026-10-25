# ⚡ J.A.R.V.I.S. Gesture Interface

### AI-Powered Hand Tracking & Holographic Visualization

J.A.R.V.I.S. Gesture Interface is a real-time computer vision project that uses a webcam to recognize hand gestures and display animated holographic objects. Inspired by the futuristic J.A.R.V.I.S. interface from Iron Man, the application combines hand tracking, gesture classification, animated visual effects, and a sci-fi heads-up display (HUD).

Control the holographic interface using simple hand movements — no keyboard interaction is required for the main gestures.

---

## 🚀 Features

- **Real-Time Hand Tracking:** Detects and tracks hand landmarks using MediaPipe.
- **Five-Finger Analysis:** Estimates which fingers are extended and displays the finger count.
- **Four Gesture Controls:** Recognizes an open palm, closed fist, pinch, and one-finger gesture.
- **Holographic Visual Effects:** Cyan wireframes, glowing outlines, scanlines, mesh grids, flicker, and materialization animations.
- **Animated Objects:** Displays a beating heart, moving car, glitching Shadow Mantis, and animated butterfly.
- **J.A.R.V.I.S.-Style HUD:** Futuristic interface with neon hand skeletons, tracking reticles, an arc reactor, and telemetry panels.
- **Two-Hand Tracking:** Supports tracking up to two hands simultaneously.
- **Optional Voice Feedback:** Uses text-to-speech when the optional voice dependency is installed.
- **Screenshot Capture:** Save screenshots of the live interface.
- **Interactive Controls:** Toggle the HUD, switch rendering styles, enable debugging, and preview models.
- **Performance Options:** Adjustable camera resolution, render width, and hand-tracking count.
- **Automatic Setup:** The Windows launcher creates a virtual environment and installs the required packages.
- **Gesture Smoothing:** Reduces flickering between gesture classifications.

## ✋ Gesture Controls

| Hand Gesture | Activated Object |
|---|---|
| 🖐️ Open Palm | ❤️ Human Heart |
| ✊ Closed Fist | 🚗 Car |
| 🤏 Pinch (Thumb + Index) | 👾 Shadow Mantis |
| ☝️ One Finger (Index Up) | 🦋 Butterfly |

The application uses hand landmarks and relative distances to classify gestures. A smoothing algorithm helps maintain stable recognition across consecutive frames.

## 🛠️ Technology Stack

- **Python** — Application logic and gesture processing
- **MediaPipe** — Hand landmark detection
- **OpenCV** — Webcam capture, image processing, rendering, and visual effects
- **NumPy** — Numerical operations and image manipulation
- **Pillow** — Image-processing dependency
- **pyttsx3 (Optional)** — Text-to-speech feedback
- **Threading** — Background camera capture for improved responsiveness

## 📋 Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10–3.13 (Python 3.12 recommended)
- A working webcam
- Internet access for initial package installation and model download
- Sufficient processing power for real-time video processing

## ⚙️ Installation & Setup

### Windows — Quick Start

**Step 1:** Download or clone this repository.

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

Replace the repository URL with your actual GitHub repository URL.

**Step 2:** Open the project folder.

**Step 3:** Double-click `run.bat`.

The launcher automatically:
1. Detects a supported Python installation.
2. Creates a virtual environment.
3. Installs the required Python packages.
4. Checks the hologram assets.
5. Downloads the MediaPipe hand-tracking model if needed.
6. Starts the application.

**Step 4:** Allow webcam access and show your hand to the camera.

**Step 5:** Perform one of the supported gestures to activate its hologram.

### macOS / Linux

Open a terminal in the project directory and run:

```bash
chmod +x run.sh
./run.sh
```

### Manual Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install the packages and start the application:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

The application downloads the hand-tracking model when it is not already available.

## 🎮 Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` / `Esc` | Exit the application |
| `H` | Show or hide the HUD |
| `V` | Toggle voice feedback |
| `S` | Save a screenshot |
| `F` | Toggle fullscreen |
| `M` | Switch between holographic and plain-color models |
| `D` | Display gesture debugging information |
| `1` | Preview the heart |
| `2` | Preview the car |
| `3` | Preview the Shadow Mantis |
| `4` | Preview the butterfly |

Press the same model preview key again to hide its preview.

## 🧠 How It Works

The application follows a real-time computer vision pipeline:

1. **Webcam Capture:** Reads live frames from the camera.
2. **Hand Detection:** MediaPipe identifies hand landmarks, including fingertips and joints.
3. **Gesture Classification:** The gesture module calculates relative distances and finger-extension ratios.
4. **Gesture Smoothing:** A majority-vote algorithm reduces unstable classifications.
5. **Object Selection:** The recognized gesture is mapped to its corresponding holographic object.
6. **Hologram Rendering:** OpenCV processes the object with glowing edges, scanlines, animations, and transparency.
7. **HUD Composition:** The final frame combines the camera feed, hand skeleton, holograms, interface elements, and telemetry.

## 📂 Project Structure

```text
jarvis_gesture/
│
├── main.py
├── run.bat
├── run.sh
├── requirements.txt
├── README.md
│
├── assets/
│   ├── heart.png
│   ├── car.png
│   ├── mantis.png
│   └── butterfly.png
│
├── jarvis/
│   ├── __init__.py
│   ├── gestures.py
│   ├── tracker.py
│   ├── sprites.py
│   ├── hud.py
│   ├── voice.py
│   └── modelfetch.py
│
├── models/
│   ├── .gitkeep
│   └── hand_landmarker.task
│
├── tools/
│   ├── fetch_model.py
│   └── generate_assets.py
│
└── tests/
    ├── fake_hand.py
    ├── test_gestures.py
    ├── smoke_main.py
    └── render_preview.py
```

The `models/hand_landmarker.task` file is downloaded automatically when needed and does not have to be committed to GitHub.

## 🧪 Testing

Run the gesture-classification tests:

```bash
python tests/test_gestures.py
```

Run the application smoke test:

```bash
python tests/smoke_main.py
```

Generate the rendering preview:

```bash
python tests/render_preview.py
```

These scripts are provided for checking gesture classification and application rendering. Run them in an environment with the required dependencies installed.

## 🎨 Customize the Holograms

You can replace the images in the `assets/` folder with your own transparent PNG images.

Keep the existing filenames:

- `heart.png`
- `car.png`
- `mantis.png`
- `butterfly.png`

The application loads these assets and applies its holographic rendering and animation effects.

## ⚡ Performance Optimization

If the application runs slowly, try reducing the rendering resolution:

```bash
python main.py --render-width 800
```

Use a lower camera resolution:

```bash
python main.py --width 640 --height 480
```

Track only one hand:

```bash
python main.py --hands 1
```

Other options include disabling the HUD, switching to plain-color images, or disabling the camera's MJPG setting if the webcam behaves incorrectly.

## 🔧 Troubleshooting

**Camera not opening**
- Close other applications using the webcam.
- Try another camera index with `--camera 1`.

**Hand-tracking model download fails**
- Check your internet connection.
- Run the launcher again or download the official MediaPipe hand-landmarker model and place it at the configured model path.

**Gesture recognition is unstable**
- Improve lighting.
- Keep the hand fully visible.
- Use a plain background.
- Adjust the thresholds in `jarvis/gestures.py`.

**Voice feedback is unavailable**

Install the optional dependency:

```bash
pip install pyttsx3
```

**Package installation errors**
- Verify that your Python version is supported.
- Recreate the virtual environment if necessary.
- Check that the required packages installed successfully.

## 🔒 Privacy

The application processes webcam frames locally for hand tracking and visual rendering. It does not require a cloud-based gesture recognition service. The initial dependency and model downloads require internet access.

## ⚠️ Disclaimer

The heart-rate, SpO₂, engine-power, and other telemetry displayed in the interface are simulated visual elements for demonstration purposes. They are not actual sensor readings, medical measurements, or verified vehicle diagnostics.

## 🌟 Future Improvements

Potential extensions include:

- Additional custom hand gestures
- More interactive holographic objects
- Customizable gesture-to-object mappings
- Improved gesture recognition for different hand orientations
- Additional voice commands and interface controls
- More advanced 3D visualization and interaction

## 👨‍💻 Author

**Ebinezer Jeba Samuel**

Aspiring Software Engineer | AI & ML Enthusiast | Computer Vision Developer

GitHub: [@Ebinezer-cloud](https://github.com/Ebinezer-cloud)

---

⭐ If you find this project interesting, consider giving the repository a star!

**Built with Python, Computer Vision, and a J.A.R.V.I.S.-inspired vision.**
