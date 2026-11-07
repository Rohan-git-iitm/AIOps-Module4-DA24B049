"""verify_ray.py -- deploy eta_ray on a 2-CPU Ray and check the sticky canary.
Run from the submission root:  python partC/verify_ray.py | tee partC/verify_output.txt"""
import os, sys, time
from pathlib import Path

import pandas as pd, ray, requests
from ray import serve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eta_ray  # noqa: E402

URL = "http://127.0.0.1:8100/v1/models/eta:predict"
ROW = pd.read_csv(HERE.parent / "data/eta_sample.csv").head(1).to_dict(orient="records")
IDS = [f"cust-{i:04d}" for i in range(1000)]
sess = requests.Session()


def deploy(weight):
    serve.run(eta_ray.build(weight=weight), name="eta", route_prefix="/")
    time.sleep(2)


def arm(cid, headers=None):
    r = sess.post(URL, json={"customer_id": cid, "instances": ROW}, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()["arm"]


def arms(ids):
    return {c: arm(c) for c in ids}


# workers need to import eta_ray, so put partC on their path
ray.init(num_cpus=2, include_dashboard=False, logging_level="ERROR", log_to_driver=False,
         runtime_env={"env_vars": {"PYTHONPATH": str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", "")}})
serve.start(http_options={"host": "127.0.0.1", "port": 8100}, logging_config={"log_level": "WARNING"})
print("Ray cluster resources:", ray.cluster_resources().get("CPU"), "CPU")

deploy(0.2)
st = serve.status().applications["eta"].deployments
print("deployments:", {n: dict(d.replica_states) for n, d in st.items()})
print(f"CPU reserved by the app: {2 - ray.available_resources().get('CPU', 0):.2f} of 2 "
      "(router 0.1 + v1 2x0.25 + v2 1x0.25 at min)")
print("router resp sample:", sess.post(URL, json={"customer_id": "demo", "instances": ROW}).json())

# (a) share of customers on the candidate
first = arms(IDS)
share = sum(a == "v2" for a in first.values()) / len(IDS)
print(f"\n(a) weight=0.2: {share:.1%} of 1000 customers on v2 (allowed 16.0%-24.0%)")
assert abs(share - 0.2) <= 0.04

# (b) same arm on a second call
sample = IDS[:200]
second = arms(sample)
same = sum(first[c] == second[c] for c in sample)
print(f"(b) second call: {same}/200 customers kept their arm")
assert same == 200

# (c) same after redeploying the identical app
serve.delete("eta")
deploy(0.2)
third = arms(sample)
same = sum(first[c] == third[c] for c in sample)
print(f"(c) after delete + redeploy: {same}/200 customers kept their arm")
assert same == 200

# (d) raise weight 0.2 -> 0.5: nobody on v2 moves back to v1
deploy(0.5)
after = arms(IDS)
was_v2 = [c for c in IDS if first[c] == "v2"]
moved_back = [c for c in was_v2 if after[c] == "v1"]
share5 = sum(a == "v2" for a in after.values()) / len(IDS)
newly = sum(first[c] == "v1" and after[c] == "v2" for c in IDS)
print(f"(d) weight=0.5: {share5:.1%} on v2; {len(was_v2)} former v2 customers, {len(moved_back)} moved back to v1; "
      f"{newly} v1 customers newly moved to v2")
assert not moved_back

# (e) X-Beta header forces the candidate
v1_ids = [c for c in IDS if after[c] == "v1"][:50]
forced = [arm(c, headers={"X-Beta": "1"}) for c in v1_ids]
print(f"(e) X-Beta: 1 on 50 requests from v1 customers -> {forced.count('v2')}/50 served by v2")
assert forced.count("v2") == 50 and all(arm(c) == "v1" for c in v1_ids[:5])  # without header: back on v1

st = serve.status().applications["eta"].deployments
print("\nfinal deployments:", {n: dict(d.replica_states) for n, d in st.items()})
print("ALL CHECKS PASSED")
serve.shutdown()
ray.shutdown()
