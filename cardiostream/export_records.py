"""Package the held-out DS2 test records for the live demo.

    python -m cardiostream.export_records --data data/raw --out artifacts/records

Only test-set patients are streamed, so everything the dashboard shows is
on patients the model never trained on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import AAMI, DS2, load_record
from .signal import FS

# Rhythm annotation codes used in MIT-BIH (aux_note on "+" annotations).
RHYTHMS = {
    "(AB": "atrial bigeminy",
    "(AFIB": "atrial fibrillation",
    "(AFL": "atrial flutter",
    "(B": "ventricular bigeminy",
    "(BII": "2nd-degree heart block",
    "(IVR": "idioventricular rhythm",
    "(N": "normal sinus rhythm",
    "(NOD": "nodal rhythm",
    "(SBR": "sinus bradycardia",
    "(SVTA": "supraventricular tachyarrhythmia",
    "(T": "ventricular trigeminy",
    "(VFL": "ventricular flutter",
    "(VT": "ventricular tachycardia",
}


def describe(path: str) -> list[str]:
    """Rhythms present in the record, longest first (from its annotations)."""
    import wfdb

    ann = wfdb.rdann(path, "atr")
    length = wfdb.rdheader(path).sig_len
    marks = [
        (int(s), str(n).strip("\x00").strip())
        for s, n, sym in zip(ann.sample, ann.aux_note, ann.symbol)
        if sym == "+"
    ]
    ends = [m[0] for m in marks[1:]] + [length]
    dur: dict[str, int] = {}
    for (s, n), e in zip(marks, ends):
        if n in RHYTHMS:
            dur[RHYTHMS[n]] = dur.get(RHYTHMS[n], 0) + (e - s)
    return [k for k, _ in sorted(dur.items(), key=lambda kv: -kv[1])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="artifacts/records")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for name in DS2:
        rec = load_record(name, args.data)
        labels = np.array([AAMI.get(s, "Q") for s in rec.beat_symbols])
        # int16 at 1 uV resolution keeps 30 min of ECG at ~1 MB compressed.
        sig = np.round(rec.signal * 1000).astype(np.int16)
        np.savez_compressed(
            out / f"{name}.npz", signal_uv=sig, beat_samples=rec.beat_samples, beat_labels=labels
        )
        counts = {c: int((labels == c).sum()) for c in ("N", "S", "V", "F", "Q")}
        index.append(
            {
                "id": name,
                "rhythms": describe(str(Path(args.data) / name)),
                "duration_s": round(len(sig) / FS, 1),
                "beats": counts,
            }
        )
        print(name, counts, index[-1]["rhythms"])
    (out / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
