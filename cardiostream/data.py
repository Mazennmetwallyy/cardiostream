"""MIT-BIH Arrhythmia Database loading with the AAMI EC57 class mapping and the
inter-patient DS1/DS2 split of de Chazal et al. (2004).

Most published MIT-BIH results shuffle beats from all patients into train and
test sets, so the model is tested on patients it has already seen. That
inflates accuracy to ~99%. Here no test patient is ever seen in training,
which is the setting a deployed monitor actually faces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .signal import CLASSES, FS, beat_window, filter_record, refine_peak, rr_feature_matrix

# Paced records (102, 104, 107, 217) are excluded, as recommended by AAMI.
DS1 = [
    "101",
    "106",
    "108",
    "109",
    "112",
    "114",
    "115",
    "116",
    "118",
    "119",
    "122",
    "124",
    "201",
    "203",
    "205",
    "207",
    "208",
    "209",
    "215",
    "220",
    "223",
    "230",
]
DS2 = [
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "202",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
]

TRAIN = DS1

AAMI = {
    **dict.fromkeys(["N", "L", "R", "e", "j"], "N"),
    **dict.fromkeys(["A", "a", "J", "S"], "S"),
    **dict.fromkeys(["V", "E"], "V"),
    "F": "F",
}
# Every symbol that marks a heartbeat (used for RR intervals), including
# unclassifiable (Q) beats that we do not train on.
BEAT_SYMBOLS = set(AAMI) | {"/", "f", "Q"}


@dataclass
class Record:
    name: str
    signal: np.ndarray  # raw MLII lead, millivolts
    beat_samples: np.ndarray  # reference R locations (all beats)
    beat_symbols: np.ndarray  # original MIT-BIH symbols


def load_record(name: str, root: str | Path) -> Record:
    import wfdb  # training-only dependency

    path = str(Path(root) / name)
    rec = wfdb.rdrecord(path)
    ch = rec.sig_name.index("MLII")
    ann = wfdb.rdann(path, "atr")
    sym = np.asarray(ann.symbol)
    smp = np.asarray(ann.sample)
    keep = np.isin(sym, list(BEAT_SYMBOLS))
    return Record(name, rec.p_signal[:, ch].astype(np.float32), smp[keep], sym[keep])


def record_beats(rec: Record):
    """Extract (windows, rr_features, labels) for every AAMI N/S/V/F beat."""
    filtered = filter_record(rec.signal)
    r_peaks = np.array([refine_peak(filtered, int(s)) for s in rec.beat_samples])
    rr, idx = rr_feature_matrix(r_peaks)
    xs, rs, ys = [], [], []
    for f, i in zip(rr, idx):
        label = AAMI.get(rec.beat_symbols[i])
        if label is None:
            continue
        w = beat_window(filtered, int(r_peaks[i]))
        if w is None:
            continue
        xs.append(w)
        rs.append(f)
        ys.append(CLASSES.index(label))
    return np.stack(xs), np.stack(rs), np.asarray(ys, dtype=np.int64)


def build_split(names: list[str], root: str | Path):
    """Concatenate beats from several records. Returns (X, RR, y, groups),
    where groups holds each beat's record name for patient-grouped CV."""
    parts = [record_beats(load_record(n, root)) for n in names]
    X, R, y = (np.concatenate(p) for p in zip(*parts))
    groups = np.concatenate([np.full(len(p[2]), n) for n, p in zip(names, parts)])
    return X, R, y, groups


def class_counts(y: np.ndarray) -> dict[str, int]:
    return {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}


__all__ = ["DS1", "DS2", "TRAIN", "AAMI", "FS", "load_record", "build_split", "class_counts"]
