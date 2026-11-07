# Module 4 assignment - starter (SwiftParcel ETA service)

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python eta_data.py            # writes data/eta_train.csv, data/eta_fresh.csv, data/eta_v1.joblib, data/eta_v2.joblib
    python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"

Load generator:

    python load_test.py --url http://localhost:8000/v1/models/eta:predict --data data/eta_sample.csv \
           --concurrency 1 2 4 8 16 --duration 4 --out loadtest_results.json

## to reproduce everything:

All commands run from this folder (the submission root). One script regenerates every result file:

    bash run_all.sh

Individual pieces:

    pytest partA partB -q
    uvicorn eta_app:app --app-dir partA --port 8000                    # Part A (ETA_MODEL=data/eta_v2.joblib for v2)
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db python partB/eta_mlflow.py register|gate|shadow
    mlflow models serve -m "models:/eta-model@prod" -p 5101 --env-manager local
    mlflow models serve -m "models:/eta-model@candidate" -p 5102 --env-manager local
    python partC/verify_ray.py
    python partD/smoke_eta.py --url http://localhost:8000/v1/models/eta:predict --expect-version v1

You also may first need: `sudo apt install python3-venv curl`.
