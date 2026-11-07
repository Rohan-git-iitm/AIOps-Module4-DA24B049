"""eta_app.py -- SwiftParcel ETA REST service (Part A).
Run from the submission root:  uvicorn eta_app:app --app-dir partA --port 8000
Model bundle path comes from ETA_MODEL (default data/eta_v1.joblib)."""
import os, time, uuid
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, model_validator


def _resolve(path):
    # relative paths are tried from the cwd first, then from the submission root
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = Path(__file__).resolve().parent.parent / path
    return p


MODEL_PATH = os.environ.get("ETA_MODEL", "data/eta_v1.joblib")
bundle = joblib.load(_resolve(MODEL_PATH))           # loaded once, at import
model = bundle["model"]
FEATURES = bundle["feature_columns"]
MODEL_VERSION = bundle["version"]


class Delivery(BaseModel):
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible: distance_km > 150 together with n_stops > 20")
        return self


class PredictRequest(BaseModel):
    instances: list[Delivery] = Field(min_length=1, max_length=500)


app = FastAPI(title="SwiftParcel ETA")

# ---- hand-rolled Prometheus metrics (single event loop thread updates them) ----
BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
req_count = {}                                       # (path, status) -> n
lat_buckets = [0] * len(BUCKETS)
lat_sum, lat_count = 0.0, 0


def observe(path, status, seconds):
    global lat_sum, lat_count
    req_count[(path, status)] = req_count.get((path, status), 0) + 1
    for i, le in enumerate(BUCKETS):
        if seconds <= le:
            lat_buckets[i] += 1
    lat_sum += seconds
    lat_count += 1


@app.middleware("http")
async def request_id_and_metrics(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:                                # anything unhandled -> generic 500
        response = JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)
    response.headers["X-Request-ID"] = rid
    if request.url.path != "/metrics":
        observe(request.url.path, response.status_code, time.perf_counter() - t0)
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(x) for x in e["loc"] if x != "body"), "msg": e["msg"]}
               for e in exc.errors()]
    return JSONResponse({"error": "validation_failed", "details": details,
                         "request_id": request.state.request_id}, status_code=422)


@app.post("/v1/models/eta:predict")
def predict(req: PredictRequest, request: Request):
    rid = request.state.request_id
    try:
        X = pd.DataFrame([d.model_dump() for d in req.instances])[FEATURES]
        preds = [round(float(p), 1) for p in model.predict(X)]
    except Exception:
        return JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)
    return {"predictions": preds, "model_version": MODEL_VERSION, "request_id": rid}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/metrics")
def metrics():
    lines = ["# HELP eta_requests_total Requests by path and status.",
             "# TYPE eta_requests_total counter"]
    for (path, status), n in sorted(req_count.items()):
        lines.append(f'eta_requests_total{{path="{path}",status="{status}"}} {n}')
    lines += ["# HELP eta_request_latency_seconds Request latency.",
              "# TYPE eta_request_latency_seconds histogram"]
    for le, n in zip(BUCKETS, lat_buckets):
        lines.append(f'eta_request_latency_seconds_bucket{{le="{le}"}} {n}')
    lines.append(f'eta_request_latency_seconds_bucket{{le="+Inf"}} {lat_count}')
    lines.append(f"eta_request_latency_seconds_sum {lat_sum}")
    lines.append(f"eta_request_latency_seconds_count {lat_count}")
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
