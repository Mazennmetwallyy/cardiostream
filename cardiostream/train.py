"""Train BeatNet on the DS1 patients and report once on the unseen DS2 patients.

    python -m cardiostream.cv      # choose class weighting by patient-grouped CV on DS1
    python -m cardiostream.train   # train on all of DS1, evaluate on DS2, export ONNX

Writes artifacts/model.onnx (served by the API) and artifacts/metrics.json
(the numbers quoted in the README and shown on the dashboard).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn

from . import __version__
from .data import AAMI, DS2, TRAIN, build_split, class_counts, load_record
from .model import BeatNet, example_inputs
from .signal import CLASSES, beat_window, detect_qrs, filter_record, match_peaks, rr_feature_matrix

# Chosen by patient-grouped cross-validation on DS1 (see cardiostream/cv.py and
# artifacts/cv.json), never by looking at test results.
DEFAULT_WEIGHT_POWER = 0.0


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Amplitude scaling, small time shift and Gaussian noise."""
    b = x.shape[0]
    x = x * (1 + 0.15 * torch.randn(b, 1, 1, device=x.device))
    x = torch.roll(x, int(np.random.randint(-5, 6)), dims=2)
    return x + 0.05 * torch.randn_like(x)


def class_weights(y: np.ndarray, power: float) -> torch.Tensor:
    """(N / n_c) ** power, normalised to mean 1. power=0 is the plain loss;
    0.5 is inverse-sqrt-frequency; 1 is full inverse frequency."""
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    w = (counts.sum() / np.maximum(counts, 1)) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, R: np.ndarray, bs: int = 4096) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i : i + bs]).unsqueeze(1).to(dev)
        rb = torch.from_numpy(R[i : i + bs]).to(dev)
        out.append(torch.softmax(model(xb, rb), 1).cpu().numpy())
    return np.concatenate(out)


def fit(
    X: np.ndarray,
    R: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 16,
    weight_power: float = DEFAULT_WEIGHT_POWER,
    batch: int = 256,
    lr: float = 2e-3,
    seed: int = 7,
    monitor: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    log: bool = True,
) -> tuple[nn.Module, list[dict]]:
    """Fixed-schedule training (one-cycle LR, final weights kept)."""
    set_seed(seed)
    dev = device()
    model = BeatNet().to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y, weight_power).to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    steps = epochs * int(np.ceil(len(y) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    Xt = torch.from_numpy(X).unsqueeze(1).to(dev)
    Rt, yt = torch.from_numpy(R).to(dev), torch.from_numpy(y).to(dev)
    history, t0 = [], time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(yt), device=dev)
        total = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch]
            loss = loss_fn(model(augment(Xt[b]), Rt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        h = {"epoch": epoch, "train_loss": round(total / len(yt), 4)}
        if monitor is not None:
            h["holdout_macro_f1"] = round(
                f1_score(monitor[2], predict(model, *monitor[:2]).argmax(1), average="macro"), 4
            )
        history.append(h)
        if log:
            print(f"epoch {epoch:2d}  {h}  ({time.time() - t0:.0f}s)")
    return model, history


def report(y: np.ndarray, p: np.ndarray) -> dict:
    rep = classification_report(
        y, p, labels=range(len(CLASSES)), target_names=CLASSES, output_dict=True, zero_division=0
    )
    per_class = {
        c: {
            "precision": round(rep[c]["precision"], 4),
            "recall": round(rep[c]["recall"], 4),
            "f1": round(rep[c]["f1-score"], 4),
            "support": int(rep[c]["support"]),
        }
        for c in CLASSES
    }
    return {
        "accuracy": round(float((y == p).mean()), 4),
        "macro_f1": round(float(f1_score(y, p, average="macro")), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y, p, labels=range(len(CLASSES))).tolist(),
    }


def streaming_eval(model: nn.Module, root: str) -> dict:
    """End-to-end check of the deployed path: our QRS detector finds beats
    (no reference annotations), features are built from detected peaks, and
    each prediction is scored against the cardiologist label it matches."""
    ys, ps, tp, fp, fn = [], [], 0, 0, 0
    for name in DS2:
        rec = load_record(name, root)
        f = filter_record(rec.signal)
        det = detect_qrs(f)
        pairs, ud, ur = match_peaks(det, rec.beat_samples)
        tp, fp, fn = tp + len(pairs), fp + len(ud), fn + len(ur)
        ref_of = dict(pairs)
        rr, idx = rr_feature_matrix(det)
        X, R, Y = [], [], []
        for feat, i in zip(rr, idx):
            if i not in ref_of:
                continue
            label = AAMI.get(rec.beat_symbols[ref_of[i]])
            w = beat_window(f, int(det[i]))
            if label is None or w is None:
                continue
            X.append(w)
            R.append(feat)
            Y.append(CLASSES.index(label))
        if X:
            ps.append(predict(model, np.stack(X), np.stack(R)).argmax(1))
            ys.append(np.asarray(Y))
    out = report(np.concatenate(ys), np.concatenate(ps))
    out["qrs_detection"] = {
        "sensitivity": round(tp / (tp + fn), 4),
        "ppv": round(tp / (tp + fp), 4),
        "reference_beats": tp + fn,
    }
    return out


def load_splits(root: str):
    cache = Path(root) / "beats_cache.npz"
    if cache.exists():
        z = np.load(cache)
        return tuple(z[k] for k in ("Xtr", "Rtr", "ytr", "gtr", "Xte", "Rte", "yte"))
    Xtr, Rtr, ytr, gtr = build_split(TRAIN, root)
    Xte, Rte, yte, _ = build_split(DS2, root)
    np.savez(cache, Xtr=Xtr, Rtr=Rtr, ytr=ytr, gtr=gtr, Xte=Xte, Rte=Rte, yte=yte)
    return Xtr, Rtr, ytr, gtr, Xte, Rte, yte


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="artifacts")
    # Fixed in advance: there is no model selection on the test patients.
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--weight-power", type=float, default=DEFAULT_WEIGHT_POWER)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    Xtr, Rtr, ytr, _, Xte, Rte, yte = load_splits(args.data)
    print(f"train {class_counts(ytr)}\ntest  {class_counts(yte)}\ndevice {device()}")

    # A random 10% of training beats is held out only to plot the learning
    # curve. It is not used to pick epochs or weights.
    mon = np.random.default_rng(args.seed).random(len(ytr)) < 0.1
    model, history = fit(
        Xtr[~mon],
        Rtr[~mon],
        ytr[~mon],
        epochs=args.epochs,
        weight_power=args.weight_power,
        seed=args.seed,
        monitor=(Xtr[mon], Rtr[mon], ytr[mon]),
    )
    model = model.cpu().eval()
    torch.save(model.state_dict(), out / "beatnet.pt")

    pte = predict(model, Xte, Rte).argmax(1)
    test = report(yte, pte)
    print(classification_report(yte, pte, target_names=CLASSES, digits=4))
    stream = streaming_eval(model, args.data)
    print("streaming", json.dumps({k: stream[k] for k in ("accuracy", "macro_f1", "qrs_detection")}))

    torch.onnx.export(
        model,
        example_inputs(),
        out / "model.onnx",
        input_names=["beat", "rr"],
        output_names=["logits"],
        dynamic_axes={"beat": {0: "batch"}, "rr": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    cv_path = out / "cv.json"
    metrics = {
        "model": "BeatNet (residual 1D-CNN + RR features)",
        "version": __version__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "classes": list(CLASSES),
        "split": {
            "protocol": "Inter-patient (de Chazal et al., 2004): no test patient seen in training",
            "train_records": TRAIN,
            "test_records": list(DS2),
            "train_counts": class_counts(ytr),
            "test_counts": class_counts(yte),
        },
        "training": {
            "epochs": args.epochs,
            "seed": args.seed,
            "class_weight_power": args.weight_power,
            "model_selection": "class weighting chosen by patient-grouped 5-fold CV on DS1; fixed schedule",
            "holdout": "random 10% of training-patient beats, for the learning curve only",
            "history": history,
            "seconds": round(time.time() - t0, 1),
            "hardware": f"{platform.machine()} ({device().type})",
        },
        "cv": json.loads(cv_path.read_text()) if cv_path.exists() else None,
        "test_reference_peaks": test,
        "test_streaming": stream,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out / 'model.onnx'} and {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
