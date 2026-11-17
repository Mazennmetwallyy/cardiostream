"""Signal processing shared by training and live serving.

Everything here is causal (it only looks at past samples, plus a short,
bounded lookahead in the QRS detector), so the exact same code runs
offline over a whole record and online over a live stream. Keeping one
implementation for both paths is what prevents train/serve skew.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

FS = 360  # MIT-BIH sampling rate (Hz)

# Beat window around the R-peak: 100 samples before (~278 ms) and
# 156 after (~433 ms) gives a 256-sample window covering P-QRS-T.
PRE = 100
POST = 156
WINDOW = PRE + POST

CLASSES = ("N", "S", "V", "F")
CLASS_NAMES = {
    "N": "Normal / bundle branch block",
    "S": "Supraventricular ectopic",
    "V": "Ventricular ectopic",
    "F": "Fusion",
}

# Number of RR-interval features fed to the model alongside the waveform.
N_RR = 5
_RR_HISTORY = 10


def bandpass_sos(fs: int = FS) -> np.ndarray:
    """0.5-40 Hz Butterworth band-pass: removes baseline wander and mains noise."""
    return sps.butter(2, [0.5, 40.0], btype="bandpass", fs=fs, output="sos")


class CausalFilter:
    """Stateful band-pass filter. Filtering a record in chunks gives exactly
    the same output as filtering it in one go."""

    def __init__(self, fs: int = FS):
        self.sos = bandpass_sos(fs)
        self.zi = np.zeros((self.sos.shape[0], 2))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = sps.sosfilt(self.sos, np.asarray(x, dtype=np.float64), zi=self.zi)
        return y


def filter_record(x: np.ndarray, fs: int = FS) -> np.ndarray:
    return CausalFilter(fs)(x)


def refine_peak(filtered: np.ndarray, approx: int, radius: int = int(0.05 * FS)) -> int:
    """Snap an approximate R location to the largest absolute deflection within
    +/-50 ms. Training (reference annotations) and serving (detector output)
    both centre beats with this rule, so windows are aligned identically."""
    lo = max(approx - radius, 0)
    hi = min(approx + radius + 1, len(filtered))
    if hi <= lo:
        return approx
    return lo + int(np.argmax(np.abs(filtered[lo:hi])))


def beat_window(filtered: np.ndarray, r: int) -> np.ndarray | None:
    """Return the z-normalised window around R, or None if it runs off the edge."""
    lo, hi = r - PRE, r + POST
    if lo < 0 or hi > len(filtered):
        return None
    w = filtered[lo:hi].astype(np.float32)
    w = w - np.median(w)
    std = w.std()
    return w / (std + 1e-6)


def rr_features(pre_rr: float, post_rr: float, local_rr: float) -> np.ndarray:
    """RR features in seconds plus ratios to the patient's recent rhythm.
    Premature beats (S, V) show up as a short pre-RR and long post-RR
    relative to the local average, which morphology alone can miss."""
    return np.array(
        [pre_rr, post_rr, local_rr, pre_rr / local_rr, post_rr / local_rr],
        dtype=np.float32,
    )


def rr_feature_matrix(r_peaks: np.ndarray, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """RR features for every beat that has a previous and a next beat.

    Returns (features, valid_index) where valid_index indexes into r_peaks.
    The local average uses only *previous* RR intervals, matching what is
    available online.
    """
    r = np.asarray(r_peaks, dtype=np.int64)
    rr = np.diff(r) / fs
    feats, idx = [], []
    for i in range(1, len(r) - 1):
        pre, post = rr[i - 1], rr[i]
        hist = rr[max(0, i - _RR_HISTORY) : i]
        local = float(np.mean(hist))
        feats.append(rr_features(pre, post, local))
        idx.append(i)
    if not feats:
        return np.zeros((0, N_RR), np.float32), np.zeros(0, np.int64)
    return np.stack(feats), np.asarray(idx)


@dataclass
class StreamingQRSDetector:
    """Online QRS detector in the style of Pan & Tompkins (1985).

    Input is the band-passed signal, pushed in arbitrary-sized chunks.
    Output is the global sample index of each R-peak, emitted with a
    bounded delay (``lookahead`` seconds) once it can no longer change.
    """

    fs: int = FS
    lookahead: float = 0.35
    refractory: float = 0.25
    threshold_ratio: float = 0.3
    min_amplitude: float = 0.1  # mV, band-passed R-wave

    _filt: list = field(default_factory=list, repr=False)
    _mwi: list = field(default_factory=list, repr=False)
    _offset: int = 0  # global index of _filt[0]
    _prev: float = 0.0
    _scanned: int = 0  # global index up to which peaks have been searched
    _last_r: int = -(10**9)
    _spk: float | None = None
    _npk: float = 0.0
    _seen: int = 0
    _rr_avg: float | None = None
    _pending: list = field(default_factory=list, repr=False)  # sub-threshold candidates

    start: int = 0  # global index of the first sample that will be pushed

    def __post_init__(self):
        self._offset = self._scanned = self.start
        self._win = max(1, int(0.15 * self.fs))
        self._sq_hist = np.zeros(0)

    def push(self, filtered_chunk: np.ndarray) -> list[int]:
        x = np.asarray(filtered_chunk, dtype=np.float64)
        if x.size == 0:
            return []
        # derivative -> square -> moving-window integration (150 ms)
        d = np.diff(np.concatenate(([self._prev], x)))
        self._prev = float(x[-1])
        sq = d * d
        sq_all = np.concatenate((self._sq_hist, sq))
        csum = np.cumsum(np.concatenate(([0.0], sq_all)))
        n_new = len(sq)
        start = len(sq_all) - n_new
        j = np.arange(start + 1, start + n_new + 1)
        mwi = (csum[j] - csum[np.maximum(0, j - self._win)]) / self._win
        self._sq_hist = sq_all[-self._win :]

        self._filt.extend(x.tolist())
        self._mwi.extend(mwi.tolist())
        self._seen += n_new

        # Learn initial signal level from the first 2 s.
        if self._spk is None:
            if self._seen < 2 * self.fs:
                return []
            arr = np.asarray(self._mwi)
            self._spk = float(np.percentile(arr, 99)) * 0.6
            self._npk = float(np.median(arr))
            self._scanned = self._offset

        return self._scan()

    def _emit(self, g_mwi: int, filt: np.ndarray, out: list[int]) -> None:
        # MWI peaks lag the QRS; look back up to 150 ms in the filtered
        # signal for the actual R wave.
        li = g_mwi - self._offset
        a = max(0, li - self._win)
        r_local = a + int(np.argmax(np.abs(filt[a : li + 1])))
        r_local = refine_peak(filt, r_local)
        r = self._offset + r_local
        if r - self._last_r <= self.refractory * self.fs:
            return
        # Absolute floor: thresholds above are relative, so without this a
        # disconnected lead (pure noise) would still produce "beats".
        if abs(filt[r_local]) < self.min_amplitude:
            return
        if self._last_r > 0:
            rr = r - self._last_r
            if rr < 3 * self.fs:
                self._rr_avg = rr if self._rr_avg is None else 0.125 * rr + 0.875 * self._rr_avg
        out.append(int(r))
        self._last_r = r
        self._pending = [(pg, ph) for pg, ph in self._pending if pg > g_mwi]

    def _scan(self) -> list[int]:
        out: list[int] = []
        mwi = np.asarray(self._mwi)
        filt = np.asarray(self._filt)
        end_global = self._offset + len(mwi)
        safe_end = end_global - int(self.lookahead * self.fs)
        lo = self._scanned - self._offset
        hi = safe_end - self._offset
        if hi <= lo + 2:
            return out
        seg = mwi[max(lo - 1, 0) : hi + 1]
        base = max(lo - 1, 0)
        cand, _ = sps.find_peaks(seg, distance=int(0.2 * self.fs))
        for c in cand:
            li = base + c
            g = self._offset + li
            if g < self._scanned:
                continue
            h = float(mwi[li])
            thr = self._npk + self.threshold_ratio * (self._spk - self._npk)
            # Searchback: if a beat is overdue (> 1.66x the average RR), accept
            # the strongest sub-threshold candidate since the last beat, at half
            # the threshold. Handles records where tall ectopic beats inflate the
            # threshold above the normal beats.
            if self._rr_avg is not None and g - self._last_r > 1.66 * self._rr_avg:
                best = None
                for pg, ph in self._pending:
                    if (
                        ph > 0.5 * thr
                        and pg - self._last_r > self.refractory * self.fs
                        and (best is None or ph > best[1])
                    ):
                        best = (pg, ph)
                if best is not None:
                    self._emit(best[0], filt, out)
                    self._spk = 0.25 * best[1] + 0.75 * self._spk
            if h > thr and g - self._last_r > self.refractory * self.fs:
                self._emit(g, filt, out)
                self._spk = 0.125 * h + 0.875 * self._spk
            else:
                self._npk = 0.125 * h + 0.875 * self._npk
                self._pending.append((g, h))
        self._scanned = safe_end

        # Keep a bounded buffer (~3 s) so memory stays flat on long streams.
        keep = 3 * self.fs
        drop = len(self._filt) - keep
        if drop > 0 and self._offset + drop < self._scanned - self.fs:
            del self._filt[:drop]
            del self._mwi[:drop]
            self._offset += drop
        # Candidates that fell out of the buffer can no longer be searched back.
        self._pending = [(pg, ph) for pg, ph in self._pending if pg - self._offset >= self._win]
        return out


def detect_qrs(filtered: np.ndarray, fs: int = FS, chunk: int = 360) -> np.ndarray:
    """Run the streaming detector over a whole record (for offline evaluation)."""
    det = StreamingQRSDetector(fs=fs)
    peaks: list[int] = []
    for i in range(0, len(filtered), chunk):
        peaks.extend(det.push(filtered[i : i + chunk]))
    return np.asarray(peaks, dtype=np.int64)


def match_peaks(detected: np.ndarray, reference: np.ndarray, tol: int = int(0.15 * FS)):
    """Greedy one-to-one matching within +/-150 ms (the standard ANSI/AAMI EC57
    tolerance). Returns (pairs, unmatched_detected, unmatched_reference)."""
    detected = np.sort(np.asarray(detected))
    reference = np.sort(np.asarray(reference))
    pairs, i, j = [], 0, 0
    used_d = np.zeros(len(detected), bool)
    used_r = np.zeros(len(reference), bool)
    while i < len(detected) and j < len(reference):
        diff = detected[i] - reference[j]
        if abs(diff) <= tol:
            pairs.append((i, j))
            used_d[i] = used_r[j] = True
            i += 1
            j += 1
        elif diff < 0:
            i += 1
        else:
            j += 1
    return pairs, np.where(~used_d)[0], np.where(~used_r)[0]
