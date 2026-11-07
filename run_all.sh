#!/usr/bin/env bash
# run_all.sh -- regenerate every result file of the Module 4 assignment from a clean checkout.
# Usage (from this folder, inside the venv from README.md):   bash run_all.sh
# Writes partA/*.json, partB/*.txt, partC/verify_output.txt,
# partD/smoke_output.txt and machine_info.txt.
set -euo pipefail
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db MLFLOW_DISABLE_TELEMETRY=true
URL=http://127.0.0.1:8000/v1/models/eta:predict

wait_for() { for _ in $(seq 120); do curl -sf "$1" >/dev/null && return 0; sleep 1; done; echo "timeout: $1"; exit 1; }
cleanup() { pkill -f "uvicorn eta_app" || true; pkill -f "mlflow models serve" || true
            pkill -f "mlflow.pyfunc.scoring_server" || true; sleep 2; }
trap cleanup EXIT

echo "== machine"
{ uname -srm; lscpu | grep -E "Model name|^CPU\(s\)"; free -h | head -2; python --version
  pip list 2>/dev/null | grep -iE "^(fastapi|uvicorn|scikit-learn|pandas|mlflow|ray) "; } > machine_info.txt
cat machine_info.txt

echo "== data + models"
python eta_data.py
python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"

echo "== unit tests"
pytest partA partB -q

echo "== Part A load tests (1 worker)"
uvicorn eta_app:app --app-dir partA --port 8000 --log-level warning & wait_for http://127.0.0.1:8000/healthz
python load_test.py --url $URL --data data/eta_sample.csv --concurrency 1 2 4 8 16 --duration 4 --out partA/loadtest_results.json
python load_test.py --url $URL --data data/eta_sample.csv --concurrency 1 --duration 4 --batch 1 --out partA/loadtest_batch1.json
python load_test.py --url $URL --data data/eta_sample.csv --concurrency 1 --duration 4 --batch 100 --out partA/loadtest_batch100.json

echo "== Part D test against Part A v1 server"
python partD/smoke_eta.py --url $URL --expect-version v1 --max-p95-ms 80 | tee partD/smoke_output.txt
cleanup

echo "== Part A load tests (2 workers)"
uvicorn eta_app:app --app-dir partA --port 8000 --workers 2 --log-level warning & wait_for http://127.0.0.1:8000/healthz
sleep 2
python load_test.py --url $URL --data data/eta_sample.csv --concurrency 1 2 4 8 16 --duration 4 --out partA/loadtest_2workers.json
cleanup

echo "== Part B register + gate"
rm -rf mlflow.db mlruns mlartifacts
python partB/eta_mlflow.py register 2>/dev/null | tee partB/register_output.txt
{ echo "### default thresholds"; python partB/eta_mlflow.py gate 2>/dev/null
  echo; echo "### min_rel_gain=0.60"; python partB/eta_mlflow.py gate --min-rel-gain 0.60 2>/dev/null; } > partB/gate_output.txt
grep '"promote"' partB/gate_output.txt

echo "== Part B shadow"
mlflow models serve -m "models:/eta-model@prod" -p 5101 --env-manager local > /tmp/serve_prod.log 2>&1 &
mlflow models serve -m "models:/eta-model@candidate" -p 5102 --env-manager local > /tmp/serve_cand.log 2>&1 &
wait_for http://127.0.0.1:5101/ping; wait_for http://127.0.0.1:5102/ping
python partB/eta_mlflow.py shadow 2>/dev/null | tee partB/shadow_output.txt
cleanup

echo "== Part C Ray verification"
python partC/verify_ray.py 2>&1 | grep -v -E "raylet|^\s*$" | tee partC/verify_output.txt
grep -q "ALL CHECKS PASSED" partC/verify_output.txt

echo "== done: all result files written"
