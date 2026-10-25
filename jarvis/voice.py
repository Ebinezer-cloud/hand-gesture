"""Optional Jarvis voice. Works if `pyttsx3` is installed; silently disabled otherwise."""
import queue
import threading


class Voice:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.available = False
        self._q = queue.Queue()
        try:
            import pyttsx3  # noqa: F401
            self.available = True
        except Exception:  # noqa: BLE001
            return
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        import pyttsx3
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
        except Exception:  # noqa: BLE001
            self.available = False
            return
        while True:
            text = self._q.get()
            if text is None:
                return
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:  # noqa: BLE001
                pass

    def say(self, text):
        if self.enabled and self.available and self._q.qsize() < 2:
            self._q.put(text)

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled
