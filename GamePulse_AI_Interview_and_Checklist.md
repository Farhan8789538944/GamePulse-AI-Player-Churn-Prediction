# GamePulse AI - Interview and completion guide

## Two-minute project introduction

I built GamePulse AI to study early player engagement and predict 14-day inactivity from the first seven days of observed gameplay. I used a public dataset for one game, cleaned it to 153,828 gameplay records, and engineered 12 activity, recency, score and time-gap features.

The main challenge was class imbalance: most eligible devices became inactive. I used chronological splits with label-maturity exclusions and compared Logistic Regression, Random Forest and XGBoost, including class weighting and oversampling. I selected Logistic Regression based on validation probability performance, checked calibration, and froze its decision threshold before final testing. On 5,084 test devices, it achieved ROC-AUC 0.7915 and balanced accuracy 74.18%.

I added SHAP explanations, two engagement segments and rule-based suggestions. PostgreSQL stores events, features and predictions; FastAPI serves results and fresh predictions; Streamlit lets an analyst explore players and model performance. The API suite has 35 passing tests.

My main learning was to compare against strong simple baselines. Recency was competitive and better on some inactivity-ranking metrics. The suggestions are ideas for retention experiments, not proven improvements in retention.

## Questions and defensible answers

### Why this project?
It connects a practical engagement question with the full data science lifecycle: data quality, behavioral features, imbalanced learning, chronological evaluation, interpretation and application delivery.

### What is churn here?
An operational proxy: no recorded gameplay during the 14-day outcome window. It is not permanent churn. The first observed event is not necessarily registration.

### Why seven observation days and fourteen outcome days?
They are explicit project design choices that give a fixed early-behavior window and a measurable future outcome. I do not claim they are business-optimal. Alternative windows would require a new evaluation design.

### Why not a random split?
The intended task predicts later behavior. A chronological split better reflects time ordering. Purging examples near boundaries ensures earlier labels have matured before the next evaluation period.

### Why Logistic Regression instead of XGBoost?
On validation, Logistic Regression had competitive ranking and better probability scores than the tested alternatives. The selected C=0.01 marginally improved Brier score. Complexity alone was not a reason to select a model.

### Why is accuracy lower than the dummy?
The dummy predicts the dominant inactive class for everyone. It achieves 94.37% test accuracy but zero returning-player recall. My threshold trades majority-class recall for recognition of returning devices, yielding 74.18% balanced accuracy.

### What does the threshold mean?
Predicted inactivity at or above 0.94 is classified as inactive. This threshold was chosen on validation balanced accuracy. It is not a business-cost optimum, and a probability below it can still be substantial.

### Did weighting or oversampling help?
They improved returning-class recall at the default threshold but worsened probability metrics. Oversampling duplicates training observations; it does not create new independent information. The final model is unweighted.

### Is the model calibrated?
I measured probability quality and tested sigmoid calibration with a chronological holdout. It did not improve validation performance meaningfully. I retained the uncalibrated model; I do not claim perfect calibration.

### Does your model beat recency?
Not on every metric. Test ROC-AUC was 0.7915 versus 0.7904 and returning AP was 0.3578 versus 0.2964. Recency was better on inactivity AP and top-risk precision. These are descriptive point estimates, not significance-tested wins.

### What does SHAP explain?
How the model's features move its prediction relative to a fixed training background. Contributions are in log-odds. Their sum plus the base value reconstructs the prediction. They do not identify causes, and related features can share attribution.

### Why are there 14 SHAP bars for 12 features?
Two missing-gap indicators are added by preprocessing. Single-play devices have no between-play gaps. Both indicators describe the same missingness pattern, so they are not separate independent causes.

### How are retention suggestions generated?
Ordered deterministic rules based on plays, recency and the frozen probability threshold. They are distinct from SHAP explanations and have not been tested as interventions.

### Why use PostgreSQL and FastAPI?
PostgreSQL organizes persistent events, versioned snapshots and predictions. FastAPI provides validated interfaces for the dashboard and fresh inference. Separating future outcomes from features helps avoid accidental label use.

### How did you test it?
I checked import idempotency, source/database counts, artifact hashes, stored/fresh probability parity, pagination and invalid requests, and SHAP reconstruction. The latest API suite passed 35 tests. This is not a production load test.

### Is there an AI agent?
No retained agent is part of the finished project. I explored a local tool-calling prototype but removed it because its latency and inconsistent response completion did not justify its limited added value.

### What is not implemented?
MLflow, automated drift monitoring, a production cloud deployment, CI/CD, real-time ingestion and a validated retention intervention. I would not put these on the resume.

## Completion checklist

### Already supported by supplied evidence
- [x] Data preparation, features, splits and model experiments.
- [x] Frozen model, decision policy and held-out evaluation.
- [x] Explanations, segments and deterministic suggestions.
- [x] Database schema, idempotent imports and stored predictions.
- [x] API and dashboard with performance and SHAP views.
- [x] Agent prototype removed; 35 API tests still pass.

### Final repository handoff checks (not yet confirmed)
- [ ] Copy the supplied README into the repository after reviewing paths and requirements.
- [ ] Save and review a working dependency lock; remove unneeded dependencies only in a separately verified environment.
- [ ] Confirm secrets, large raw/processed data, model artifacts and local logs are excluded as intended.
- [ ] Provide dataset attribution and a clear plan for obtaining ignored artifacts on a new machine.
- [ ] Record actual project dates; do not reuse the 2015 dataset year as a development date.
- [ ] Capture a short demo or current screenshots without credentials.
- [ ] Commit the reviewed documentation and project code to your chosen repository; do not publish raw data without checking terms.

## Five-minute demonstration order
1. Overview: seven-day observation, fourteen-day inactivity, historical validation cohort.
2. Player browser: show one player, probability, segment and rule suggestion.
3. SHAP: show contribution directions and the reconstruction check.
4. Model performance: switch to the test cohort and explain the confusion matrix and recency baseline.
5. API/tests: show endpoint documentation and the latest passing test output.

## Resume and application wording

Short form description: Built an end-to-end player inactivity prediction platform with behavioral features, chronological model evaluation, SHAP explanations, engagement segmentation, FastAPI and a PostgreSQL-backed Streamlit dashboard.

Minimal relevant skills: Python, SQL, Pandas, Scikit-learn, XGBoost, EDA, Feature Engineering, Model Evaluation, SHAP, FastAPI, PostgreSQL, Streamlit, Docker, pytest.

Use only skills you can explain. Use real development dates and real repository links. Do not claim that the API tests cover every frontend workflow or that recommendations improved retention.
