"""Patient-grouped cross-validation on the training patients (DS1) only.

    python -m cardiostream.cv

Used to choose the class-weighting strength without touching the test
patients. Every fold holds out whole patients, so the score estimates how
the model does on someone new, which is the question that matters.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from .signal import CLASSES
from .train import fit, load_splits, predict, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="artifacts/cv.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--powers", default="0,0.25,0.5")
    args = ap.parse_args()

    X, R, y, groups, *_ = load_splits(args.data)  # test arrays are ignored
    powers = [float(p) for p in args.powers.split(",")]
    folds = list(GroupKFold(n_splits=args.folds).split(X, y, groups))
    results = {}
    for power in powers:
        t0 = time.time()
        oof = np.zeros(len(y), dtype=np.int64)
        for k, (tr, va) in enumerate(folds):
            model, _ = fit(X[tr], R[tr], y[tr], epochs=args.epochs, weight_power=power, log=False)
            oof[va] = predict(model, X[va], R[va]).argmax(1)
            print(f"power {power}  fold {k + 1}/{args.folds}  patients {sorted(set(groups[va]))}")
        rep = report(y, oof)
        fold_f1 = [round(float(f1_score(y[va], oof[va], average="macro")), 4) for _, va in folds]
        results[str(power)] = {**rep, "fold_macro_f1": fold_f1, "seconds": round(time.time() - t0)}
        pc = {c: rep["per_class"][c]["f1"] for c in CLASSES}
        print(f"power {power}: out-of-fold macro F1 {rep['macro_f1']}  per-class F1 {pc}")

    best = max(results, key=lambda p: results[p]["macro_f1"])
    summary = {
        "protocol": f"{args.folds}-fold GroupKFold over DS1 patients; pooled out-of-fold predictions",
        "selection_metric": "macro F1",
        "chosen_weight_power": float(best),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"chosen class-weight power: {best}")


if __name__ == "__main__":
    main()
