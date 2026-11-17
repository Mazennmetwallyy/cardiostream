"""ONNX Runtime inference and the online beat-classification pipeline.

The serving image needs only numpy, scipy and onnxruntime: no PyTorch.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .signal import (
    CLASSES,
    FS,
    POST,
    PRE,
    CausalFilter,
    StreamingQRSDetector,
    beat_window,
    rr_features,
)


class Classifier:
    def __init__(self, model_path: str | Path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])

    def predict_proba(self, beats: np.ndarray, rr: np.ndarray) -> np.ndarray:
        beats = np.asarray(beats, np.float32).reshape(len(beats), 1, -1)
        logits = self.session.run(["logits"], {"beat": beats, "rr": np.asarray(rr, np.float32)})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


@dataclass
class BeatEvent:
    sample: int  # global R-peak index
    label: str
    probs: dict[str, float]
    rr: float  # pre-RR in seconds
    heart_rate: float  # bpm from the local average RR


class OnlinePipeline:
    """Filter -> detect QRS -> build features -> classify, one chunk at a time.

    A beat is classified once the *next* beat has been detected (its post-RR
    is a feature), so decisions arrive about one heartbeat after the beat.
    """

    def __init__(self, classifier: Classifier, fs: int = FS, history: int = 10, start: int = 0):
        """``start`` is the global sample index of the first sample pushed,
        so beat positions line up with the source record when a stream
        begins part-way through it."""
        self.clf = classifier
        self.fs = fs
        self.filter = CausalFilter(fs)
        self.detector = StreamingQRSDetector(fs=fs, start=start)
        self.peaks: list[int] = []
        self.rr_hist: deque[float] = deque(maxlen=history)
        self.buf = np.zeros(0)  # recent filtered samples
        self.buf_start = start  # global index of buf[0]

    def push(self, raw_chunk: np.ndarray) -> list[BeatEvent]:
        f = self.filter(raw_chunk)
        self.buf = np.concatenate((self.buf, f))
        events: list[BeatEvent] = []
        for r in self.detector.push(f):
            self.peaks.append(r)
            if len(self.peaks) >= 3:
                ev = self._classify(len(self.peaks) - 2)
                if ev is not None:
                    events.append(ev)
        # Keep ~4 s of filtered signal: enough to cut any pending beat window.
        keep = 4 * self.fs
        if len(self.buf) > keep:
            cut = len(self.buf) - keep
            self.buf = self.buf[cut:]
            self.buf_start += cut
        return events

    def _classify(self, i: int) -> BeatEvent | None:
        prev, r, nxt = self.peaks[i - 1], self.peaks[i], self.peaks[i + 1]
        pre, post = (r - prev) / self.fs, (nxt - r) / self.fs
        self.rr_hist.append(pre)
        local = float(np.mean(self.rr_hist))
        lo = r - PRE - self.buf_start
        if lo < 0 or r + POST - self.buf_start > len(self.buf):
            return None
        w = beat_window(self.buf, r - self.buf_start)
        if w is None:
            return None
        p = self.clf.predict_proba(w[None], rr_features(pre, post, local)[None])[0]
        k = int(p.argmax())
        return BeatEvent(
            sample=r,
            label=CLASSES[k],
            probs={c: round(float(v), 4) for c, v in zip(CLASSES, p)},
            rr=round(pre, 3),
            heart_rate=round(60.0 / local, 1),
        )
