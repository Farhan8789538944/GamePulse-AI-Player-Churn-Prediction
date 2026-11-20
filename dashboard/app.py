import math
from datetime import datetime, timezone
from urllib.parse import quote
from matplotlib.figure import Figure
import httpx
import pandas as pd
import streamlit as st


API_BASE = "http://127.0.0.1:8000"
PAGE_SIZE = 25

st.set_page_config(
    page_title="GamePulse AI",
    page_icon="🎮",
    layout="wide",
)


def api_get(path, params=None):
    try:
        response = httpx.get(
            f"{API_BASE}{path}",
            params=params,
            timeout=10.0,
            trust_env=False,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as error:
        status = error.response.status_code

        if status == 404:
            st.error("No scored validation record was found for this player.")
        elif status == 503:
            st.error(
                "The API cannot complete this request. "
                "Check PostgreSQL and the API terminal."
            )
        else:
            st.error(f"The API returned HTTP {status}.")

        st.stop()

    except httpx.RequestError:
        st.error(
            "Cannot reach the GamePulse API at http://127.0.0.1:8000. "
            "Start the FastAPI server and refresh this page."
        )
        st.stop()

    except ValueError:
        st.error("The API returned an unexpected response.")
        st.stop()


def format_utc(value):
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return timestamp.astimezone(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


FEATURE_LABELS = {
    "plays_7d": "Gameplay events",
    "active_days_7d": "Active days",
    "days_since_last_play": "Days since last play",
    "activity_span_days": "Activity span in days",
    "plays_first_3d": "Plays in first 3 days",
    "plays_last_3d": "Plays in last 3 days",
    "play_count_change": "Last 3 days minus first 3 days",
    "mean_score_7d": "Mean score",
    "best_score_7d": "Best score",
    "score_std_7d": "Score standard deviation",
    "mean_gap_hours": "Mean time between plays in hours",
    "max_gap_hours": "Maximum time between plays in hours",
}


st.title("GamePulse AI")
st.caption("Player inactivity predictions and early engagement analytics")

with st.sidebar:
    st.header("About this view")
    st.write("Historical validation cohort")
    st.write("Observation window: **7 days**")
    st.write("Prediction window: **14 days**")
    st.caption(
        "A player is represented by a device ID. "
        "First observed gameplay is not necessarily registration."
    )

    if st.button("Refresh data", width="stretch"):
        st.rerun()

    st.divider()
    st.caption(
        "This view reads stored predictions. "
        "It does not train models or send player messages."
    )

summary = api_get("/v1/summary")

st.info(
    "Historical validation data — these are not live players. "
    "Inactivity means no recorded gameplay during the outcome window."
)

threshold = float(summary["decision_threshold"])

cards = st.columns(4)
cards[0].metric("Scored players", f"{summary['scored_snapshots']:,}")
cards[1].metric(
    "At or above threshold",
    f"{summary['at_or_above_threshold']:,}",
)
cards[2].metric(
    "Below threshold",
    f"{summary['below_threshold']:,}",
)

mean_probability = summary["mean_predicted_inactivity"]
cards[3].metric(
    "Mean predicted inactivity",
    f"{mean_probability:.1%}" if mean_probability is not None else "—",
)

st.caption(
    f"Decision threshold: {threshold:.0%}. "
    "Mean predicted inactivity is a model estimate, not the observed "
    "inactivity rate. Below the threshold does not necessarily mean low risk."
)

if summary["unscored_snapshots"]:
    st.warning(
        f"{summary['unscored_snapshots']:,} snapshots have no stored prediction."
    )

total = int(summary["scored_snapshots"])
if total == 0:
    st.info("No scored players are available yet.")
    st.stop()

st.divider()
st.subheader("Browse players")
st.caption(
    "Players are ordered by predicted inactivity, highest first. "
    "Equal probabilities are ordered by snapshot ID."
)

page_count = max(1, math.ceil(total / PAGE_SIZE))

page_number = int(
    st.number_input(
        "Page",
        min_value=1,
        max_value=page_count,
        value=1,
        step=1,
    )
)

offset = (page_number - 1) * PAGE_SIZE

page = api_get(
    "/v1/players",
    params={"limit": PAGE_SIZE, "offset": offset},
)

items = page["items"]

if not items:
    st.info("No players were returned for this page.")
    st.stop()

table_rows = []
for item in items:
    recommendation = item["recommendation"]
    table_rows.append(
        {
            "Device ID": str(item["device_id"]),
            "Predicted inactivity (%)": (
                100 * item["churn_probability"]
            ),
            "Decision": (
                "At or above threshold"
                if item["at_or_above_threshold"]
                else "Below threshold"
            ),
            "Engagement segment": recommendation["segment_name"],
            "Prediction cutoff (UTC)": format_utc(
                item["prediction_cutoff"]
            ),
        }
    )

st.dataframe(
    pd.DataFrame(table_rows),
    hide_index=True,
    width="stretch",
    column_config={
        "Device ID": st.column_config.TextColumn("Device ID"),
        "Predicted inactivity (%)": st.column_config.NumberColumn(
            "Predicted inactivity (%)",
            format="%.2f",
        ),
    },
)

st.caption(
    f"Showing {offset + 1}–{offset + len(items)} of {total:,} players. "
    f"Page {page_number} of {page_count}."
)

st.divider()
st.subheader("Player details")

device_ids = [str(item["device_id"]) for item in items]
selected_device = st.selectbox(
    "Choose a player from this page",
    options=device_ids,
    key=f"selected_player_page_{page_number}",
)

player = api_get(
    f"/v1/players/{quote(selected_device, safe='')}"
)
recommendation = player["recommendation"]

left, right = st.columns(2)

with left:
    st.metric(
        "Predicted 14-day inactivity",
        f"{player['churn_probability']:.2%}",
    )
    st.write(
        "**Decision:** "
        + (
            "At or above the inactivity threshold"
            if player["at_or_above_threshold"]
            else "Below the inactivity threshold"
        )
    )
    st.write(f"**Decision threshold:** {player['decision_threshold']:.0%}")
    st.write(f"**Segment:** {recommendation['segment_name']}")
    st.caption(
        f"Prediction cutoff: {format_utc(player['prediction_cutoff'])}"
    )

with right:
    st.markdown("**Suggested retention action**")
    st.write(recommendation["suggested_action"])
    st.markdown("**Why this rule matched**")
    st.write(recommendation["reason"])
    st.caption(
        "This is a rule-based suggestion, not a SHAP explanation or "
        "evidence that the action improves retention."
    )

with st.expander("View observation-window features"):
    feature_rows = []
    for name, value in player["features"].items():
        feature_rows.append(
            {
                "Feature": FEATURE_LABELS.get(name, name),
                "Value": (
                    "Not available"
                    if value is None
                    else f"{value:,.4f}".rstrip("0").rstrip(".")
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(feature_rows),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Gap features are unavailable for players with only one gameplay "
        "event. Missing values are not zero."
    )

with st.expander("Record versions"):
    st.json(
        {
            "snapshot_id": player["snapshot_id"],
            "partition": player["data_partition"],
            "model_version": player["model_version"],
            "rules_version": player["rules_version"],
            "segment_version": recommendation["segment_version"],
        }
    )

st.divider()
st.subheader("Why did the model predict this?")
st.caption(
    "SHAP explains the selected player's model prediction. "
    "It is separate from the rule that generated the retention suggestion."
)

if st.button(
    "Explain this player's prediction",
    key=f"explain_{player['snapshot_id']}",
):
    with st.spinner("Calculating and checking the explanation..."):
        explanation = api_get(
            f"/v1/players/{quote(str(player['device_id']), safe='')}/explanation"
        )

    if (
        explanation["snapshot_id"] != player["snapshot_id"]
        or explanation["model_version"] != player["model_version"]
        or not explanation["reconstruction_verified"]
        or abs(
            explanation["churn_probability"] - player["churn_probability"]
        ) > 1e-10
    ):
        st.error("The explanation does not match the displayed prediction.")
        st.stop()

    st.session_state["player_explanation"] = explanation


explanation = st.session_state.get("player_explanation")

# Never display an explanation belonging to a previously selected player.
if (
    explanation is not None
    and explanation["snapshot_id"] == player["snapshot_id"]
    and explanation["model_version"] == player["model_version"]
):
    st.success(
        "Explanation verified: the contributions reconstruct "
        f"the model's {explanation['churn_probability']:.2%} "
        "inactivity probability."
    )

    contributions = pd.DataFrame(explanation["contributions"])

    display_labels = {
        **FEATURE_LABELS,
        "missingindicator_mean_gap_hours": "Mean gap missing indicator",
        "missingindicator_max_gap_hours": "Maximum gap missing indicator",
    }

    contributions["label"] = contributions["feature"].map(
        lambda name: display_labels.get(name, name)
    )

    # Show all contributions, with the largest magnitudes at the top.
    contributions["absolute_contribution"] = (
        contributions["shap_log_odds"].abs()
    )
    contributions = contributions.sort_values(
        "absolute_contribution",
        ascending=True,
    )

    colours = [
        "#b45309" if value > 0 else "#2563eb" if value < 0 else "#64748b"
        for value in contributions["shap_log_odds"]
    ]

    figure = Figure(figsize=(10, 6), layout="constrained")
    axis = figure.subplots()

    axis.barh(
        contributions["label"],
        contributions["shap_log_odds"],
        color=colours,
    )
    axis.axvline(0, color="#334155", linewidth=1)
    axis.set_xlabel("Contribution to inactivity log-odds")
    axis.set_title("Feature contributions relative to the training background")
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    st.pyplot(figure)
    figure.clear()

    st.caption(
        "Blue bars decrease predicted inactivity relative to the reference; "
        "orange bars increase it. Bar values are log-odds contributions, "
        "not percentage-point changes."
    )

    st.info(
        "Related features can share attribution. Recency and activity span "
        "are mathematically related, and both missing-gap indicators describe "
        "the same single-play missingness. These are not independent causes."
    )

    with st.expander("View exact contributions and reconstruction"):
        details = contributions.sort_values(
            "absolute_contribution",
            ascending=False,
        )[["feature", "shap_log_odds", "direction"]]

        st.dataframe(
            details,
            hide_index=True,
            width="stretch",
            column_config={
                "shap_log_odds": st.column_config.NumberColumn(
                    "SHAP contribution (log-odds)",
                    format="%.6f",
                )
            },
        )

        final_log_odds = (
            explanation["base_log_odds"]
            + explanation["contribution_sum"]
        )

        st.write(
            f"Base log-odds: **{explanation['base_log_odds']:.6f}**"
        )
        st.write(
            f"Sum of contributions: **{explanation['contribution_sum']:.6f}**"
        )
        st.write(f"Final log-odds: **{final_log_odds:.6f}**")
        st.write(
            "Reconstructed probability: "
            f"**{explanation['reconstructed_probability']:.6%}**"
        )

        st.caption(
            f"Reference background: {explanation['background_devices']} "
            "training devices. The base value is a reference in log-odds; "
            "it is not the average predicted probability."
        )

    with st.expander("Explanation limitations"):
        for limitation in explanation["limitations"]:
            st.write(f"- {limitation}")

else:
    st.caption(
        "Select a player and click the button to load their explanation."
    )