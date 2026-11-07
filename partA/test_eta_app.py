"""test_eta_app.py -- run from the submission root:  pytest partA -q"""
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))
import eta_app  # noqa: E402

URL = "/v1/models/eta:predict"
ROW = {"distance_km": 12.5, "n_stops": 4, "hour": 10, "weekday": 2,
       "traffic_index": 0.3, "rain": 0, "weight_kg": 5.0}


@pytest.fixture
def client():
    return TestClient(eta_app.app, raise_server_exceptions=False)


def row(**kw):
    return {**ROW, **kw}


def test_valid_request(client):
    r = client.post(URL, json={"instances": [ROW]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 1
    assert body["model_version"] == eta_app.MODEL_VERSION
    p = body["predictions"][0]
    assert isinstance(p, float) and round(p, 1) == p


def test_batch_order_preserved(client):
    rows = [row(distance_km=d) for d in (2, 150, 40)]
    batch = client.post(URL, json={"instances": rows}).json()["predictions"]
    singles = [client.post(URL, json={"instances": [x]}).json()["predictions"][0] for x in rows]
    assert batch == singles
    assert batch[1] > batch[2] > batch[0]          # distances 150 > 40 > 2


@pytest.mark.parametrize("bad, field", [
    (row(distance_km=0.05), "distance_km"),        # out of range
    (row(hour=24), "hour"),                         # out of range
    (row(rain=2), "rain"),                          # not 0/1
    (row(weight_kg=0), "weight_kg"),                # must be > 0
    (row(n_stops="many"), "n_stops"),               # wrong type
    ({k: v for k, v in ROW.items() if k != "traffic_index"}, "traffic_index"),  # missing
])
def test_invalid_input_returns_422(client, bad, field):
    r = client.post(URL, json={"instances": [bad]})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_failed"
    assert any(field in d["field"] for d in body["details"])
    assert body["request_id"] == r.headers["X-Request-ID"]


def test_cross_field_rule(client):
    r = client.post(URL, json={"instances": [row(distance_km=151, n_stops=21)]})
    assert r.status_code == 422
    assert "infeasible" in r.text
    # each condition alone is fine
    assert client.post(URL, json={"instances": [row(distance_km=151, n_stops=20)]}).status_code == 200
    assert client.post(URL, json={"instances": [row(distance_km=150, n_stops=21)]}).status_code == 200


def test_batch_size_limits(client):
    assert client.post(URL, json={"instances": []}).status_code == 422
    assert client.post(URL, json={"instances": [ROW] * 501}).status_code == 422
    ok = client.post(URL, json={"instances": [ROW] * 500})
    assert ok.status_code == 200 and len(ok.json()["predictions"]) == 500


def test_request_id_echoed(client):
    r = client.post(URL, json={"instances": [ROW]}, headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    assert r.json()["request_id"] == "abc-123"


def test_request_id_generated_on_error(client):
    r = client.post(URL, json={"instances": []})
    rid = r.headers["X-Request-ID"]
    assert re.fullmatch(r"[0-9a-f-]{36}", rid)
    assert r.json()["request_id"] == rid


def test_request_id_echoed_on_error(client):
    r = client.post(URL, json={"instances": [row(hour=99)]}, headers={"X-Request-ID": "err-1"})
    assert r.headers["X-Request-ID"] == "err-1" and r.json()["request_id"] == "err-1"


def test_metrics_output(client):
    client.post(URL, json={"instances": [ROW]})
    client.get("/metrics")
    text = client.get("/metrics").text
    assert re.search(r'eta_requests_total\{path="/v1/models/eta:predict",status="200"\} \d+', text)
    assert 'path="/metrics"' not in text
    assert text.count("eta_request_latency_seconds_bucket") >= 3
    assert 'le="+Inf"' in text
    assert "eta_request_latency_seconds_sum" in text and "eta_request_latency_seconds_count" in text


def test_healthz(client):
    r = client.get("/healthz")
    assert r.json() == {"status": "ok", "model_version": eta_app.MODEL_VERSION}
    assert "X-Request-ID" in r.headers


def test_generic_500(client, monkeypatch):
    class Boom:
        def predict(self, X):
            raise RuntimeError("secret internal detail")
    monkeypatch.setattr(eta_app, "model", Boom())
    r = client.post(URL, json={"instances": [ROW]}, headers={"X-Request-ID": "boom"})
    assert r.status_code == 500
    assert "secret" not in r.text and "RuntimeError" not in r.text
    assert r.headers["X-Request-ID"] == "boom"
