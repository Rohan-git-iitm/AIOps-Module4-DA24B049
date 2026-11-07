"""eta_ray.py -- Part C: Ray Serve app with a STICKY canary (per customer).
    client --HTTP--> Router (ingress, sha256 bucketing, X-Beta override) --> EtaModel v1 (2 replicas)
                                                                       +-> EtaModel v2 (autoscaling 1..3)
Run:  serve run eta_ray:app   (from partC/)   or use verify_ray.py.
Structure adapted from the Lecture 10 Ray_Serve_Exercise (ray_serve_app.py); routing rule is new."""
import hashlib, time
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from ray import serve
from ray.serve.handle import DeploymentHandle

DATA = Path(__file__).resolve().parent.parent / "data"


@serve.deployment(ray_actor_options={"num_cpus": 0.25}, max_ongoing_requests=16,
                  logging_config={"log_level": "WARNING"})
class EtaModel:
    def __init__(self, path: str, extra_work_ms: float = 0.0):
        b = joblib.load(path)                           # loaded once per replica
        self.model, self.cols, self.version = b["model"], b["feature_columns"], b["version"]
        self.extra = extra_work_ms / 1000.0             # simulated heavier model (blocking sleep)
        self.replica = serve.get_replica_context().replica_id.unique_id

    async def __call__(self, rows: list[dict]):
        X = pd.DataFrame(rows)[self.cols]
        preds = [round(float(p), 1) for p in self.model.predict(X)]
        if self.extra:
            time.sleep(self.extra)
        return {"predictions": preds, "model_version": self.version, "served_by": self.replica}


class Delivery(BaseModel):
    distance_km: float; n_stops: int; hour: int; weekday: int
    traffic_index: float; rain: int; weight_kg: float


class Req(BaseModel):
    customer_id: str = Field(min_length=1)
    instances: list[Delivery] = Field(min_length=1, max_length=500)


def bucket(customer_id: str) -> float:
    """Stable hash -> [0, 1). Same id gives the same value in every process, replica and restart."""
    return int(hashlib.sha256(customer_id.encode()).hexdigest()[:8], 16) / 2**32


api = FastAPI()


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.1}, logging_config={"log_level": "WARNING"})
@serve.ingress(api)
class Router:
    def __init__(self, v1: DeploymentHandle, v2: DeploymentHandle, weight: float):
        self.h = {"v1": v1, "v2": v2}
        self.w = weight

    @api.get("/healthz")
    def healthz(self):
        return {"status": "ok", "weight": self.w}

    @api.post("/v1/models/eta:predict")
    async def predict(self, req: Req, x_beta: Optional[str] = Header(None)):
        arm = "v2" if x_beta == "1" or bucket(req.customer_id) < self.w else "v1"
        try:
            out = await self.h[arm].remote([d.model_dump() for d in req.instances])
        except Exception:
            raise HTTPException(500, "inference failed")
        return {**out, "arm": arm}


def build(weight: float = 0.1, extra_work_ms: float = 0.0):
    """Router -> (v1: 2 fixed replicas, v2: autoscaling 1..3). Worst case 0.1 + 5 x 0.25 = 1.35 CPU."""
    v1 = EtaModel.options(name="EtaModelV1", num_replicas=2).bind(str(DATA / "eta_v1.joblib"), extra_work_ms)
    v2 = EtaModel.options(name="EtaModelV2", num_replicas="auto", autoscaling_config={
        "min_replicas": 1, "max_replicas": 3, "target_ongoing_requests": 2,
        "upscale_delay_s": 3, "downscale_delay_s": 30}).bind(str(DATA / "eta_v2.joblib"), extra_work_ms)
    return Router.bind(v1, v2, weight)


app = build()
