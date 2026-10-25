"""
Hand-model download that cannot hang silently.

 * shows progress, times out when the connection stalls, retries
 * downloads to a .part file and validates the size, so a half-finished download is never used
 * falls back to curl.exe (built into Windows 10/11), then to a copy in your Downloads folder
Uses only the standard library (no mediapipe import) so it is fast.
"""
import glob
import os
import shutil
import subprocess
import sys
import time
import urllib.request

MODEL_URL = os.environ.get(
    "JARVIS_MODEL_URL",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task",
)
MIN_BYTES = 3_000_000  # the real file is about 7.8 MB


def is_valid(path):
    return os.path.isfile(path) and os.path.getsize(path) > MIN_BYTES


def _urllib_download(url, tmp, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 jarvis-gesture"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        last = 0.0
        while True:
            chunk = r.read(65536)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last > 0.2:
                last = now
                if total:
                    sys.stdout.write(f"\r         {done / 1e6:5.1f} / {total / 1e6:.1f} MB  ({done * 100 // total}%)   ")
                else:
                    sys.stdout.write(f"\r         {done / 1e6:5.1f} MB   ")
                sys.stdout.flush()
    sys.stdout.write("\n")
    if total and done != total:
        raise IOError(f"incomplete download ({done} of {total} bytes)")


def _curl_download(url, tmp):
    exe = shutil.which("curl.exe") or shutil.which("curl")
    if not exe:
        return False
    cmd = [exe, "-L", "--fail", "--connect-timeout", "15", "--max-time", "240", "-o", tmp, url]
    if sys.platform.startswith("win"):
        cmd.insert(1, "--ssl-no-revoke")  # avoids certificate-revocation hangs behind some AV/proxies
    return subprocess.call(cmd) == 0


def _from_downloads(dest):
    """If you saved the file from the browser, pick it up from the Downloads folder."""
    home = os.path.expanduser("~")
    for pat in ("Downloads/hand_landmarker*.task", "Desktop/hand_landmarker*.task"):
        for p in glob.glob(os.path.join(home, pat)):
            if is_valid(p):
                print(f"[jarvis] found {p} - copying it into place")
                shutil.copyfile(p, dest)
                return True
    return False


def fetch(dest, retries=3, timeout=20):
    """Make sure `dest` is a valid model file. Returns True on success."""
    if is_valid(dest):
        return True
    if os.path.isfile(dest):  # broken / partial file from an earlier attempt
        os.remove(dest)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    if _from_downloads(dest) and is_valid(dest):
        return True

    tmp = dest + ".part"
    print(f"[jarvis] downloading hand model (one time, ~8 MB)\n         {MODEL_URL}")
    for attempt in range(1, retries + 1):
        try:
            _urllib_download(MODEL_URL, tmp, timeout)
        except Exception as e:  # noqa: BLE001
            print(f"\n         attempt {attempt}/{retries} failed: {e}")
            if os.path.isfile(tmp):
                os.remove(tmp)  # never keep a partial file
        if os.path.isfile(tmp) and os.path.getsize(tmp) > MIN_BYTES:
            os.replace(tmp, dest)
            return True
        if attempt == 2 and not os.path.isfile(tmp):  # second try: use curl instead of Python's downloader
            print("         trying curl ...")
            if _curl_download(MODEL_URL, tmp) and os.path.isfile(tmp) and os.path.getsize(tmp) > MIN_BYTES:
                os.replace(tmp, dest)
                return True
            if os.path.isfile(tmp):
                os.remove(tmp)
        time.sleep(1)
    if os.path.isfile(tmp):
        os.remove(tmp)
    return False
