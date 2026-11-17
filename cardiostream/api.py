"""FastAPI service: REST endpoints plus a WebSocket that replays a held-out
patient's ECG in real time and classifies every heartbeat as it arrives.

    uvicorn cardiostream.api:app --reload
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .inference import Classifier, OnlinePipeline
from .signal import CLASSES, FS, N_RR, WINDOW

ARTIFACTS = Path(os.environ.get("CARDIO_ARTIFACTS", "artifacts"))
STATIC = Path(os.environ.get("CARDIO_STATIC", "frontend/dist"))
FRAME_S = 0.04  # 25 frames per second to the browser
MAX_STREAMS = int(os.environ.get("CARDIO_MAX_STREAMS", "50"))
MATCH_TOL = int(0.15 * FS)

app = FastAPI(title="CardioStream", version=__version__)
_active_streams = 0


@lru_cache
def classifier() -> Classifier:
    return Classifier(ARTIFACTS / "model.onnx")


@lru_cache
def record_index() -> list[dict]:
    return json.loads((ARTIFACTS / "records" / "index.json").read_text())


@lru_cache(maxsize=8)
def load_demo_record(rid: str):
    if rid not in {r["id"] for r in record_index()}:
        raise KeyError(rid)
    z = np.load(ARTIFACTS / "records" / f"{rid}.npz")
    return z["signal_uv"].astype(np.float32) / 1000.0, z["beat_samples"], z["beat_labels"]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "model_loaded": classifier() is not None}


@app.get("/api/records")
def records() -> list[dict]:
    return record_index()


@app.get("/api/model")
def model_card() -> dict:
    return json.loads((ARTIFACTS / "metrics.json").read_text())


class ClassifyRequest(BaseModel):
    beats: list[list[float]] = Field(..., description=f"z-normalised {WINDOW}-sample windows")
    rr: list[list[float]] = Field(..., description=f"{N_RR} RR features per beat")

    @field_validator("beats")
    @classmethod
    def _beats(cls, v):
        if not 1 <= len(v) <= 512 or any(len(b) != WINDOW for b in v):
            raise ValueError(f"1-512 beats of exactly {WINDOW} samples each")
        return v

    @field_validator("rr")
    @classmethod
    def _rr(cls, v):
        if any(len(r) != N_RR for r in v):
            raise ValueError(f"each rr row must have {N_RR} values")
        return v


@app.post("/api/classify")
def classify(req: ClassifyRequest) -> dict:
    if len(req.beats) != len(req.rr):
        raise HTTPException(422, "beats and rr must have the same length")
    t0 = time.perf_counter()
    p = classifier().predict_proba(np.asarray(req.beats), np.asarray(req.rr))
    ms = (time.perf_counter() - t0) * 1000
    return {
        "predictions": [
            {"label": CLASSES[int(row.argmax())], "probs": dict(zip(CLASSES, map(float, row)))} for row in p
        ],
        "latency_ms": round(ms, 3),
    }


def _truth(samples: np.ndarray, labels: np.ndarray, r: int) -> str | None:
    """Cardiologist label of the reference beat nearest to r (within 150 ms)."""
    i = int(np.searchsorted(samples, r))
    near = [j for j in (i - 1, i) if 0 <= j < len(samples)]
    if not near:
        return None
    j = min(near, key=lambda k: abs(int(samples[k]) - r))
    return str(labels[j]) if abs(int(samples[j]) - r) <= MATCH_TOL else None


@app.websocket("/api/stream")
async def stream(
    ws: WebSocket,
    record: str = Query(...),
    speed: float = Query(1.0, ge=0.5, le=8.0),
    start: float = Query(0.0, ge=0.0),
):
    global _active_streams
    await ws.accept()
    try:
        sig, ref_samples, ref_labels = load_demo_record(record)
    except KeyError:
        await ws.close(code=4404, reason="unknown record")
        return
    if _active_streams >= MAX_STREAMS:
        await ws.close(code=4429, reason="server busy")
        return
    _active_streams += 1
    try:
        pos = min(int(start * FS), len(sig) - FS)
        pipe = OnlinePipeline(classifier(), start=pos)
        await ws.send_json({"t": "meta", "record": record, "fs": FS, "start": pos, "length": len(sig)})
        step = max(1, int(round(FS * FRAME_S * speed)))
        next_tick = time.perf_counter()
        while pos < len(sig):
            chunk = sig[pos : pos + step]
            t0 = time.perf_counter()
            events = pipe.push(chunk)
            ms = (time.perf_counter() - t0) * 1000
            await ws.send_json({"t": "samples", "start": pos, "mv": np.round(chunk, 3).tolist()})
            for ev in events:
                await ws.send_json(
                    {
                        "t": "beat",
                        "sample": ev.sample,
                        "label": ev.label,
                        "probs": ev.probs,
                        "rr": ev.rr,
                        "hr": ev.heart_rate,
                        "truth": _truth(ref_samples, ref_labels, ev.sample),
                        "pipeline_ms": round(ms, 3),
                    }
                )
            pos += len(chunk)
            next_tick += FRAME_S
            await asyncio.sleep(max(0.0, next_tick - time.perf_counter()))
        await ws.send_json({"t": "end"})
        await ws.close()
    except WebSocketDisconnect:
        pass
    finally:
        _active_streams -= 1


if STATIC.is_dir():
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
