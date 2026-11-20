import httpx
import pandas as pd
import streamlit as st


API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Model Performance | GamePulse AI",
    page_icon="📊",
    layout="wide",
)

st.title("Model performance")
st.caption("Saved final-test evaluation of the frozen model")


def fetch_performance():
    try:
        response = httpx.get(
            f"{API_BASE}/v1/model-performance",
            timeout=10.0,
            trust_env=False,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as error:
        st.error(
            f"The performance endpoint returned HTTP "
            f"{error.response.status_code}. Check the API terminal."
        )
        st.stop()

    except httpx.RequestError:
        st.error(
            "Cannot reach the API. Start FastAPI at "
            "http://127.0.0.1:8000 and refresh this page."
        )
        st.stop()

    except ValueError:
        st.error("The API returned an unexpected response.")
        st.stop()


with st.sidebar:
    st.header("Evaluation context")
    st.write("Partition: **Final test**")
    st.caption(
        "The player browser uses validation data. "
        "This page reports a different, later cohort."
    )

    if st.button("Refresh results", width="stretch"):
        st.rerun()


results = fetch_performance()
metrics = results["model_metrics"]
report = results["classification_report"]

st.info(
    "These are saved results from the chronological test partition. "
    "Opening this page does not retrain the model, change its threshold, "
    "or rerun evaluation."
)

st.write(
    f"**Model:** {results['model_name']}  |  "
    f"**Version:** {results['model_version']}  |  "
    f"**Decision threshold:** {results['decision_threshold']:.0%}"
)

cards = st.columns(4)
cards[0].metric("Test devices", f"{results['test_devices']:,}")
cards[1].metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
cards[2].metric(
    "Balanced accuracy",
    f"{metrics['balanced_accuracy']:.2%}",
)
cards[3].metric("Brier score", f"{metrics['brier_score']:.4f}")

st.caption(
    "Higher ROC-AUC and balanced accuracy are better. "
    "Lower Brier score indicates better probability predictions."
)

st.warning(
    f"{results['observed_inactivity_rate']:.2%} of test devices were inactive. "
    "A classifier that always predicts inactivity therefore achieves high "
    "accuracy without identifying any returning players."
)

st.divider()
st.subheader("Classification results")

class_rows = []
for class_name in ["returned", "inactive"]:
    values = report[class_name]
    class_rows.append(
        {
            "Actual class": class_name.capitalize(),
            "Precision (%)": 100 * values["precision"],
            "Recall (%)": 100 * values["recall"],
            "F1-score (%)": 100 * values["f1-score"],
            "Devices": int(values["support"]),
        }
    )

st.dataframe(
    pd.DataFrame(class_rows),
    hide_index=True,
    width="stretch",
    column_config={
        "Precision (%)": st.column_config.NumberColumn(format="%.2f"),
        "Recall (%)": st.column_config.NumberColumn(format="%.2f"),
        "F1-score (%)": st.column_config.NumberColumn(format="%.2f"),
        "Devices": st.column_config.NumberColumn(format="%d"),
    },
)

st.caption(
    "Returned: at least one gameplay event in the outcome window. "
    "Inactive: no gameplay events in that window."
)

st.subheader("Confusion matrix")
matrix = results["confusion_matrix"]

labels = [name.capitalize() for name in matrix["class_order"]]
matrix_frame = pd.DataFrame(
    matrix["counts"],
    index=[f"Actual: {name}" for name in labels],
    columns=[f"Predicted: {name}" for name in labels],
)

st.dataframe(matrix_frame, width="stretch")

st.caption(
    "Rows show actual outcomes; columns show model decisions "
    f"at the {results['decision_threshold']:.0%} threshold."
)

returned_correct, returned_flagged = matrix["counts"][0]
inactive_missed, inactive_correct = matrix["counts"][1]

st.write(
    f"Of the {returned_correct + returned_flagged:,} returning devices, "
    f"the model identified {returned_correct:,} as returning. "
    f"Of the {inactive_missed + inactive_correct:,} inactive devices, "
    f"it identified {inactive_correct:,} as inactive."
)

st.divider()
st.subheader("Comparison with simple baselines")

metric_definitions = [
    ("roc_auc", "ROC-AUC", "Higher"),
    ("balanced_accuracy", "Balanced accuracy", "Higher"),
    ("accuracy", "Accuracy", "Higher"),
    ("return_average_precision", "Returning-player average precision", "Higher"),
    ("churn_average_precision", "Inactivity average precision", "Higher"),
    ("brier_score", "Brier score", "Lower"),
    ("log_loss", "Log loss", "Lower"),
]

dummy = results["baselines"]["training_prior_dummy"]
recency = results["baselines"]["recency_ranking"]


def display_metric(values, key):
    value = values.get(key)
    return "Not evaluated" if value is None else f"{value:.4f}"


comparison = []
for key, label, direction in metric_definitions:
    comparison.append(
        {
            "Metric": label,
            "Better direction": direction,
            "Logistic Regression": display_metric(metrics, key),
            "Training-prior dummy": display_metric(dummy, key),
            "Recency ranking": display_metric(recency, key),
        }
    )

st.dataframe(
    pd.DataFrame(comparison),
    hide_index=True,
    width="stretch",
)

st.caption(
    "Values in this table use their original numeric scale. "
    "The dummy uses the training class prior and predicts the majority "
    "class at its default decision rule. Recency is a ranking baseline; "
    "metrics not evaluated for it are not treated as zero."
)

st.write(
    "Logistic Regression improves on the dummy in balanced accuracy, "
    "ranking metrics, and probability quality, but has lower overall accuracy. "
    "Recency remains competitive."
)

st.divider()
st.subheader("Highest-risk 10% of test devices")

ranking_names = {
    "logistic_regression": "Logistic Regression",
    "recency": "Recency",
}

ranking_rows = []
for item in results["top_risk_ranking"]:
    ranking_rows.append(
        {
            "Ranking": ranking_names.get(item["ranking"], item["ranking"]),
            "Selected devices": item["selected_devices"],
            "Inactivity precision (%)": 100 * item["precision_at_k"],
            "Inactivity recall (%)": 100 * item["recall_at_k"],
            "Lift over random": item["lift_over_random"],
            "Expected inactive devices": item["expected_churners_selected"],
        }
    )

st.dataframe(
    pd.DataFrame(ranking_rows),
    hide_index=True,
    width="stretch",
    column_config={
        "Inactivity precision (%)": st.column_config.NumberColumn(format="%.2f"),
        "Inactivity recall (%)": st.column_config.NumberColumn(format="%.2f"),
        "Lift over random": st.column_config.NumberColumn(format="%.3f"),
        "Expected inactive devices": st.column_config.NumberColumn(format="%.2f"),
    },
)

st.caption(
    "Each ranking selects 509 devices, approximately 10% of the test set. "
    "Results average over ties at the selection boundary, so expected "
    "inactive counts can be fractional. Lift measures targeting enrichment, "
    "not an improvement in player retention."
)

with st.expander("Limitations and evaluation notes"):
    for note in results["limitations"] + results["notes"]:
        st.write(f"- {note}")

    st.write(
        "Model and policy hashes match the artifacts referenced by the "
        "saved evaluation report."
    )