"""
Capture thread: reads raw BGR frames from the GStreamer subprocess
pipe and pushes them into frame_queue.
"""

import queue
import subprocess
import time

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from gstreamer_utils import start_gstreamer


class CaptureThread(QThread):
    """
    Pure I/O thread.  Reads raw BGR frames from the GStreamer subprocess
    pipe and pushes them into frame_queue.  Drops old frames if the
    detection thread can't keep up (keeps UI smooth).
    """
    status  = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, source, frame_queue, width, height):
        super().__init__()
        self.source = source
        self.frame_queue = frame_queue
        self.width = width
        self.height = height
        self._running = False

    def run(self):
        self._running = True
        frame_size = self.width * self.height * 3

        self.status.emit("[Capture] Connecting via GStreamer...")
        proc = start_gstreamer(self.source, self.width, self.height)

        # Give GStreamer time to connect & preroll
        time.sleep(2.0)

        # Check if process died immediately
        if proc.poll() is not None:
            err = proc.stderr.read().decode(errors='replace') if proc.stderr else ""
            print(f"[GStreamer] FAILED (exit {proc.returncode}):\n{err}")
            self.status.emit(f"[Capture] GStreamer failed: {err[:80]}")
            self.stopped.emit()
            return

        self.status.emit("[Capture] GStreamer connected ✓")
        print("[CaptureThread] Started.")

        while self._running:
            raw = proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                err = ""
                if proc.stderr:
                    try:
                        err = proc.stderr.read(2000).decode(errors='replace')
                    except Exception:
                        pass
                print(f"[GStreamer] Stream ended. Got {len(raw)}/{frame_size} bytes")
                if err:
                    print(f"[GStreamer] stderr: {err[:500]}")
                self.status.emit("[Capture] Stream ended or broken.")
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)).copy()

            # Non-blocking put — drop oldest frame if queue is full
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()  # discard oldest
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait(frame)

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        self.stopped.emit()
        print("[CaptureThread] Stopped.")

    def stop(self):
        self._running = False
        self.wait(5000)
