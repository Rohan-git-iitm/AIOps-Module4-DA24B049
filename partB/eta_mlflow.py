"""eta_mlflow.py -- Part B: register both ETA models in MLflow, gate the promotion, compare in shadow.
Run from the submission root:
    python partB/eta_mlflow.py register
    python partB/eta_mlflow.py gate [--min-rel-gain 0.60]
    python partB/eta_mlflow.py shadow            # needs prod on :5101 and candidate on :5102
Tracking URI comes from MLFLOW_TRACKING_URI (default sqlite:///mlflow.db).
Pattern adapted from the Lecture 10 MLflow_Serving_Exercise (mlflow_register.py, Solutions A/B)."""
import argparse, json, os, sys, time
from pathlib import Path

import joblib, mlflow, mlflow.pyfunc, numpy as np, pandas as pd, requests, sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eta_data  # noqa: E402

NAME = "eta-model"
RUSH_HOURS = [7, 8, 9, 16, 17, 18, 19]
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


class EtaPyfunc(mlflow.pyfunc.PythonModel):
    """Wraps a joblib bundle; returns eta_minutes (1 decimal) and model_version per row."""
    def load_context(self, context):
        b = joblib.load(context.artifacts["bundle"])
        self.m, self.cols, self.ver = b["model"], b["feature_columns"], b["version"]

    def predict(self, context, model_input, params=None):
        X = pd.DataFrame(model_input)[self.cols]
        return pd.DataFrame({"eta_minutes": self.m.predict(X).round(1), "model_version": self.ver})


# ---------------- B1: register ----------------
def register():
    mlflow.set_experiment("eta-serving")
    client = MlflowClient()
    X = eta_data.make_data(200, seed=1)[eta_data.FEATURES].astype("float64")  # float schema accepts ints too
    out_example = pd.DataFrame({"eta_minutes": [30.5], "model_version": ["v1"]})
    sig = infer_signature(X.head(5), out_example)
    pins = [f"mlflow=={mlflow.__version__}", f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pd.__version__}", f"numpy=={np.__version__}", f"joblib=={joblib.__version__}"]
    aliases = {"v1": "prod", "v2": "candidate"}
    for ver, alias in aliases.items():
        bundle = str(ROOT / f"data/eta_{ver}.joblib")
        with mlflow.start_run(run_name=f"eta-{ver}"):
            mlflow.log_param("algorithm", type(joblib.load(bundle)["model"]).__name__)
            info = mlflow.pyfunc.log_model(name="model", python_model=EtaPyfunc(),
                                           artifacts={"bundle": bundle}, signature=sig,
                                           input_example=X.head(2), pip_requirements=pins,
                                           registered_model_name=NAME)
        client.set_registered_model_alias(NAME, alias, info.registered_model_version)
    print("| version | alias | model_version |\n|---|---|---|")
    for alias in aliases.values():
        mv = client.get_model_version_by_alias(NAME, alias)
        tag = mlflow.pyfunc.load_model(f"models:/{NAME}@{alias}").predict(X.head(1))["model_version"][0]
        print(f"| {mv.version} | {alias} | {tag} |")


# ---------------- B2: gate ----------------
def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05):
    """Pure function: metric dicts in, decision out. Slice regression is relative to prod's slice MAE."""
    eps = 1e-9                                       # so a value exactly on a threshold counts as passing
    gain = (prod["mae"] - cand["mae"]) / prod["mae"]
    rush = cand["mae_rush"] / prod["mae_rush"] - 1
    rain = cand["mae_rain"] / prod["mae_rain"] - 1
    criteria = {
        "mae_rel_gain": {"value": round(gain, 4), "need": f">= {min_rel_gain}", "pass": gain >= min_rel_gain - eps},
        "p95_ms": {"value": round(cand["p95_ms"], 3), "need": f"<= {max_p95_ms}", "pass": cand["p95_ms"] <= max_p95_ms + eps},
        "rush_regression": {"value": round(rush, 4), "need": f"<= {max_slice_regression}",
                            "pass": rush <= max_slice_regression + eps},
        "rain_regression": {"value": round(rain, 4), "need": f"<= {max_slice_regression}",
                            "pass": rain <= max_slice_regression + eps},
    }
    return {"criteria": criteria, "promote": all(c["pass"] for c in criteria.values())}


def evaluate(model, fresh):
    pred = model.predict(fresh[eta_data.FEATURES].astype("float64"))["eta_minutes"].to_numpy()
    err = np.abs(pred - fresh["eta_minutes"].to_numpy())
    rush, rain = fresh["hour"].isin(RUSH_HOURS).to_numpy(), (fresh["rain"] == 1).to_numpy()
    rows = fresh[eta_data.FEATURES].astype("float64")
    lat = []
    for i in range(200):                             # single-row inference latency
        t0 = time.perf_counter()
        model.predict(rows.iloc[[i]])
        lat.append((time.perf_counter() - t0) * 1000)
    return {"mae": float(err.mean()), "mae_rush": float(err[rush].mean()),
            "mae_rain": float(err[rain].mean()), "p95_ms": float(np.percentile(lat, 95))}


def promotion_gate(**thresholds):
    fresh = eta_data.make_data(2000, seed=99)        # fresh data, never seen in training
    prod = evaluate(mlflow.pyfunc.load_model(f"models:/{NAME}@prod"), fresh)
    cand = evaluate(mlflow.pyfunc.load_model(f"models:/{NAME}@candidate"), fresh)
    result = decide(prod, cand, **thresholds)
    rnd = lambda d: {k: round(v, 3) for k, v in d.items()}
    print(json.dumps({"thresholds": thresholds or "defaults", "prod": rnd(prod),
                      "candidate": rnd(cand), **result}, indent=2))
    return result


# ---------------- B3: shadow ----------------
def shadow(n=200, prod_url="http://127.0.0.1:5101/invocations", cand_url="http://127.0.0.1:5102/invocations"):
    rows = pd.read_csv(ROOT / "data/eta_fresh.csv")[eta_data.FEATURES].head(n).to_dict(orient="records")
    sess, preds, lat = requests.Session(), {"prod": [], "cand": []}, {"prod": [], "cand": []}
    for rec in rows:                                 # identical request to both servers
        for arm, url in (("prod", prod_url), ("cand", cand_url)):
            t0 = time.perf_counter()
            r = sess.post(url, json={"dataframe_records": [rec]}, timeout=10)
            lat[arm].append((time.perf_counter() - t0) * 1000)
            r.raise_for_status()
            preds[arm].append(r.json()["predictions"][0]["eta_minutes"])
    diff = np.abs(np.array(preds["prod"]) - np.array(preds["cand"]))
    print(f"requests sent to each server : {n}")
    print(f"(i)   mean |prod - cand|       : {diff.mean():.2f} min")
    print(f"(ii)  share differing > 10 min : {(diff > 10).mean():.1%}")
    print(f"(iii) p95 latency prod (5101)  : {np.percentile(lat['prod'], 95):.1f} ms")
    print(f"      p95 latency cand (5102)  : {np.percentile(lat['cand'], 95):.1f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["register", "gate", "shadow"])
    ap.add_argument("--min-rel-gain", type=float)
    a = ap.parse_args()
    if a.cmd == "register":
        register()
    elif a.cmd == "gate":
        promotion_gate(**({"min_rel_gain": a.min_rel_gain} if a.min_rel_gain is not None else {}))
    else:
        shadow()
