# GamePulse AI

Player inactivity prediction and engagement analytics for one historical game.

Predicts no recorded gameplay in the next **14 days** using the first **7 days** of observed activity. Device IDs represent player proxies. This is a local portfolio application, not a live game service.

## Implemented features

- Data audit, cleaning, EDA and 12 behavioral features.
- Chronological train/validation/test splits with label-maturity exclusions.
- Logistic Regression, Random Forest and XGBoost experiments; class weighting, oversampling, calibration and threshold analysis.
- Frozen Logistic Regression v1, C=0.01, decision threshold 0.94.
- Individual SHAP explanations with log-odds/probability reconstruction checks.
- Two K-Means engagement segments and deterministic retention suggestions.
- PostgreSQL storage, idempotent imports and versioned model metadata.
- FastAPI prediction and read endpoints; Streamlit player browser and model-performance page.
- 35 passing API tests in the latest reported run.

The experimental Ollama/LangGraph assistant has been removed. MLflow, automated drift monitoring, cloud deployment and CI/CD are not implemented claims.

## Dataset

Zenodo: https://zenodo.org/records/166035
File: `rawdata_game1.csv`; fields: `device`, `score`, `time`.
Recorded MD5: `30F76FFCF17EC1DF0822D64ED6410162`.
Check the source's license and attribution before redistributing raw data.

| Stage | Count |
|---|---:|
| Raw events | 153,929 |
| Missing-ID rows removed | 9 |
| Exact duplicates removed | 92 |
| Clean events | 153,828 |
| Unique devices | 25,956 |
| Eligible devices | 25,419 |
| Incomplete follow-up exclusions | 537 |
| Split-boundary exclusions | 1,262 |
| Train / validation / test | 14,786 / 4,287 / 5,084 |

## Results

Final chronological test cohort: 5,084 devices, including 4,798 inactive and 286 returning devices.

| Metric | Logistic Regression | Training-prior dummy | Recency |
|---|---:|---:|---:|
| ROC-AUC | 0.7915 | 0.5000 | 0.7904 |
| Balanced accuracy | 0.7418 | 0.5000 | Not evaluated |
| Accuracy | 0.8354 | 0.9437 | Not evaluated |
| Returning-class average precision | 0.3578 | 0.0563 | 0.2964 |
| Inactivity average precision | 0.9766 | 0.9437 | 0.9778 |
| Brier score | 0.0437 | 0.0535 | Not evaluated |

At threshold 0.94, inactivity precision/recall/F1 are 97.51%/84.72%/90.67%; returning precision/recall/F1 are 19.89%/63.64%/30.31%.
Confusion matrix, actual rows and predicted columns in `[returned, inactive]` order: `[[182, 104], [733, 4065]]`.

High inactivity prevalence makes accuracy alone misleading. Recency remains competitive and beats the model on inactivity AP and top-10%-risk precision. No retention uplift was measured.

## Architecture

Offline scripts prepare data, train/evaluate models and import verified records into PostgreSQL. FastAPI serves stored records and the frozen model. Streamlit accesses the API. Model artifacts stay on disk; database registry tables record metadata. Future outcome labels are stored separately from features.

## Start the existing local installation (PowerShell)

Run from the project root. Docker Desktop must be running. Preserve the existing `.env.database`, frozen artifacts and database volume.

```powershell
Set-Location D:\Projects\GamePulseAI

docker compose --env-file .env.database -f compose.yaml up -d --wait db
```

In terminal 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn gamepulse.api:app --app-dir src --host 127.0.0.1 --port 8000
```

In terminal 2:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Open http://127.0.0.1:8501 for the dashboard and http://127.0.0.1:8000/docs for API documentation.

In another terminal:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/health/live'
Invoke-RestMethod 'http://127.0.0.1:8000/health/ready'
Invoke-RestMethod 'http://127.0.0.1:8000/v1/summary'
.\.venv\Scripts\python.exe -m pytest tests\test_api.py -q -o pythonpath=src
```

## Dependencies and fresh-machine handoff

The demonstrated environment uses Python 3.12. A new clone is not sufficient if ignored datasets and artifacts are missing. This documentation has not been tested as a clean-machine installer.

Preserve the working environment before making package changes:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding utf8 requirements-working-lock.txt
```

This freeze may include unused packages from the retired agent experiment; it is an environment snapshot, not a minimal dependency specification. Review before using it for a public release. Do not blindly uninstall shared dependencies from the working environment.

For a new machine, create a Python 3.12 virtual environment, install the reviewed project requirements/lock, download the source file, recreate the ignored data/artifact directories, configure local database credentials, and run the offline stages below. Use the actual variable names expected by `compose.yaml` and the database configuration; never commit passwords. Exact reproducibility requires the retained package versions, feature definitions, random seeds and artifact hashes.

## Offline script inventory

These are the scripts used during development, grouped in execution order. Preserve existing v1 results: this is an inventory, not an instruction to overwrite the current release.

1. `audit_game1.py`, `inspect_churn_labels.py`, `eda_game1.py`
2. `build_features.py`, `split_dataset.py`
3. `train_baselines.py`, `train_random_forest.py`, `train_xgboost.py`, `train_oversampling.py`
4. `tune_logistic.py`, `check_calibration.py`, `analyze_decisions.py`
5. `evaluate_final.py` (freeze and final test; do not repeatedly tune against this output)
6. `explain_model.py`, `evaluate_subgroups.py`, `segment_players.py`, `recommend_retention.py`
7. `check_database.py`, `init_database.py`, `import_gameplay.py`, `import_snapshots.py`, `register_models.py`, `score_validation.py`

Run a script using `.\.venv\Scripts\python.exe scripts\<script_name>.py` after its prerequisites exist. Intermediate files, policies and reports are dependencies, not interchangeable outputs.

## API routes

| Route | Purpose |
|---|---|
| GET `/health/live` | Process liveness |
| GET `/health/ready` | Readiness |
| GET `/v1/summary` | Historical validation summary |
| GET `/v1/players?limit=25&offset=0` | Paginated validation predictions |
| GET `/v1/players/{device_id}` | Stored details and features |
| POST `/v1/predict` | Feature-based inference, no persistence |
| GET `/v1/model-performance` | Saved final-test results |
| GET `/v1/players/{device_id}/explanation` | SHAP explanation |

`/v1/predict` accepts the project's 12 named observation features. Use a valid stored player's `features` object as an example request; do not invent combinations inconsistent with the feature definitions.

## Data and artifact layout

- `data/raw/`, `data/processed/`: source, labels, features and splits.
- `artifacts/releases/v1/`: frozen model and decision policy.
- `artifacts/segmentation/`: segmentation pipeline, metadata and reviewed names.
- `reports/`: audit, EDA, experiments, calibration, decisions, final test, explanations, subgroups and suggestions.
- `reports/final_test/metrics.json`: authoritative saved final evaluation.
- `sql/`, `compose.yaml`: schema and database service.
- `src/gamepulse/`: application implementation.
- `dashboard/`, `tests/`: analyst UI and API tests.

## Limitations

- Historical data from one game; device IDs are proxies for people.
- First observation is not registration; inactivity is not permanent churn.
- Complete recording through the endpoint is assumed; exact-duplicate removal is provisional.
- Earlier descriptive EDA included the full dataset; model and policy selection used train/validation.
- Highly imbalanced labels; report both class recalls and baseline comparisons.
- Correlated/redundant features limit independent SHAP interpretation. SHAP explains model predictions, not causes.
- Suggestions are analyst-defined hypotheses, not proven treatments.
- Local operation is not production deployment; no scalability or intervention benchmark is claimed.

## Troubleshooting

- Database connection timeout: verify Docker Desktop, container health and port 5433. Do not delete the data volume to fix a connection problem.
- Missing artifacts or hash mismatch: restore the matching frozen release and metadata; do not bypass integrity checks.
- API unavailable: keep the Uvicorn terminal running and verify port 8000.
- Dashboard unavailable: keep Streamlit running and verify port 8501.
- Below-threshold probability seems high: 0.94 is a decision policy, not a definition of low risk.
- Conflicting dashboard counts: browser cards describe validation; performance results describe test.
