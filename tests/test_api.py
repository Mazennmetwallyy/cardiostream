"""API and end-to-end pipeline tests against the trained artifacts."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from cardiostream import api
from cardiostream.inference import OnlinePipeline
from cardiostream.signal import CLASSES, FS, N_RR, WINDOW

pytestmark = pytest.mark.skipif(
    not (api.ARTIFACTS / "model.onnx").exists(), reason="trained artifacts not present"
)


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_records_are_held_out_test_patients(client):
    from cardiostream.data import DS2, TRAIN

    ids = {r["id"] for r in client.get("/api/records").json()}
    assert ids == set(DS2)
    assert not ids & set(TRAIN)


def test_model_card_has_test_metrics(client):
    card = client.get("/api/model").json()
    assert card["classes"] == list(CLASSES)
    assert 0 <= card["test_streaming"]["macro_f1"] <= 1


def test_classify_returns_distribution(client):
    rng = np.random.default_rng(0)
    body = {"beats": rng.standard_normal((3, WINDOW)).tolist(), "rr": (np.ones((3, N_RR))).tolist()}
    r = client.post("/api/classify", json=body)
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 3
    for p in preds:
        assert p["label"] in CLASSES
        assert sum(p["probs"].values()) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(
    "body",
    [
        {"beats": [[0.0] * (WINDOW - 1)], "rr": [[1.0] * N_RR]},
        {"beats": [[0.0] * WINDOW], "rr": [[1.0] * (N_RR + 1)]},
        {"beats": [[0.0] * WINDOW] * 2, "rr": [[1.0] * N_RR]},
        {"beats": [], "rr": []},
    ],
)
def test_classify_rejects_bad_shapes(client, body):
    assert client.post("/api/classify", json=body).status_code == 422


def test_stream_rejects_unknown_record(client):
    with client.websocket_connect("/api/stream?record=999") as ws, pytest.raises(WebSocketDisconnect) as e:
        ws.receive_json()
    assert e.value.code == 4404


def test_stream_emits_samples_and_classified_beats(client, monkeypatch):
    monkeypatch.setattr(api, "FRAME_S", 0.0)  # don't wait in real time
    beats = []
    with client.websocket_connect("/api/stream?record=233&speed=8&start=60") as ws:
        meta = ws.receive_json()
        assert meta["t"] == "meta" and meta["start"] == 60 * FS
        while len(beats) < 20:
            m = ws.receive_json()
            if m["t"] == "beat":
                beats.append(m)
    assert all(b["sample"] >= 60 * FS for b in beats)
    assert all(b["label"] in CLASSES for b in beats)
    agree = np.mean([b["label"] == b["truth"] for b in beats])
    assert agree >= 0.8


def test_pipeline_on_held_out_record_agrees_with_cardiologists():
    sig, ref, labels = api.load_demo_record("233")
    seg = sig[: 120 * FS]
    pipe = OnlinePipeline(api.classifier())
    events = []
    for i in range(0, len(seg), 14):
        events += pipe.push(seg[i : i + 14])
    truth = [api._truth(ref, labels, e.sample) for e in events]
    matched = [(e.label, t) for e, t in zip(events, truth) if t in CLASSES]
    assert len(matched) > 100  # ~2 minutes of beats were found
    acc = np.mean([p == t for p, t in matched])
    v_true = [p for p, t in matched if t == "V"]
    assert acc > 0.85
    assert np.mean([p == "V" for p in v_true]) > 0.8  # PVCs are caught
