"""smoke_eta.py -- release gate for the ETA service (Part A server, or the Part C router).
    python partD/smoke_eta.py --url http://localhost:8000/v1/models/eta:predict --expect-version v1 [--max-p95-ms 80]
Exit 0 if every check passes, 1 otherwise. Check list follows the Lecture 10 Minikube smoke_test.py idea
(healthz, version, predictions, p95, bad input -> 422) plus model-sanity checks."""
import argparse, sys, time

import numpy as np, requests

BASE_ROW = {"distance_km": 20.0, "n_stops": 5, "hour": 11, "weekday": 2,
            "traffic_index": 0.3, "rain": 0, "weight_kg": 5.0}


def predict(url, rows):
    # customer_id is required by the Part C router and ignored by the Part A service
    r = requests.post(url, json={"customer_id": "smoke-test", "instances": rows}, timeout=10)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--max-p95-ms", type=float, default=80.0)
    a = ap.parse_args()
    base = a.url.split("/v1/")[0]
    results = []

    def check(name, fn):
        try:
            ok, info = fn()
        except Exception as e:                                   # a crash is a failed check
            ok, info = False, f"{type(e).__name__}: {e}"
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")

    def healthz():
        r = requests.get(f"{base}/healthz", timeout=5)
        return r.status_code == 200 and r.json().get("status") == "ok", r.text.strip()

    def version_and_shape():
        rows = [{**BASE_ROW, "distance_km": d} for d in (3, 30, 90)]
        j = predict(a.url, rows)
        p = j["predictions"]
        ok = (j["model_version"] == a.expect_version and len(p) == 3
              and all(isinstance(x, (int, float)) and 0 < x < 2000 for x in p))
        return ok, f"model_version={j['model_version']} (expected {a.expect_version}), predictions={p}"

    def monotone(field, values):
        def fn():
            p = predict(a.url, [{**BASE_ROW, field: v} for v in values])["predictions"]
            return all(x <= y for x, y in zip(p, p[1:])), f"{field}={values} -> {p}"
        return fn

    def invalid_422():
        bad = [{**BASE_ROW, "n_stops": "many"}]                  # wrong type: rejected by both servers
        r = requests.post(a.url, json={"customer_id": "smoke-test", "instances": bad}, timeout=5)
        return r.status_code == 422, f"HTTP {r.status_code}"

    def p95():
        sess, lat = requests.Session(), []
        for i in range(30):
            t0 = time.perf_counter()
            r = sess.post(a.url, json={"customer_id": "smoke-test",
                                       "instances": [{**BASE_ROW, "distance_km": 5 + i}]}, timeout=10)
            lat.append((time.perf_counter() - t0) * 1000)
            r.raise_for_status()
        v = float(np.percentile(lat, 95))
        return v <= a.max_p95_ms, f"p95={v:.1f} ms over 30 requests (limit {a.max_p95_ms})"

    check("healthz", healthz)
    check("version + prediction shape", version_and_shape)
    check("sanity: longer distance never gives a shorter ETA", monotone("distance_km", [2, 10, 40, 120]))
    check("sanity: heavier traffic never gives a shorter ETA", monotone("traffic_index", [0.0, 0.3, 0.6, 0.9]))
    check("sanity: more stops never give a shorter ETA", monotone("n_stops", [1, 5, 10, 20]))
    check("invalid input -> 422", invalid_422)
    check("latency p95", p95)

    passed = all(results)
    print(f"SMOKE TEST {'PASSED' if passed else 'FAILED'} ({sum(results)}/{len(results)} checks)")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
