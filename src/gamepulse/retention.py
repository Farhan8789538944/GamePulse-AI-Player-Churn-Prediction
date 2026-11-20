import math

SEGMENT_NAMES = {
    0: "Limited early activity",
    1: "Higher early activity",
}

RULE_CONFIG = {
    "version": "rules_v1",
    "recent_activity_days": 2.0,
    "extended_inactivity_days": 5.0,
}

def recommend(
    plays_7d,
    days_since_last_play,
    churn_probability,
    threshold,
    config,
):
    values = [
        plays_7d,
        days_since_last_play,
        churn_probability,
        threshold,
    ]

    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Recommendation inputs must be finite.")

    if plays_7d < 1 or float(plays_7d) != int(plays_7d):
        raise ValueError("plays_7d must be a positive integer.")

    if not 0 <= days_since_last_play <= 7:
        raise ValueError("Recency must be between 0 and 7 days.")

    if not 0 <= churn_probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")

    predicted_inactive = churn_probability >= threshold

    decision = (
        "At or above inactivity decision threshold"
        if predicted_inactive
        else "Below inactivity decision threshold"
    )

    # Ordered rules: a device receives the first matching suggestion.
    if plays_7d == 1:
        rule_id = "single_observed_play"
        reason = "Only one gameplay event occurred in the first seven days."
        action = (
            "Review the first-play experience for this cohort; "
            "consider testing optional onboarding help or a return reminder."
        )

    elif days_since_last_play <= config["recent_activity_days"]:
        rule_id = "recently_active"
        reason = "Gameplay occurred within two days before prediction."
        action = (
            "Continue monitoring; consider testing relevant gameplay tips "
            "or content discovery rather than an inactivity reminder."
        )

    elif (
        days_since_last_play
        > config["extended_inactivity_days"]
    ):
        rule_id = "extended_inactivity"
        reason = "No gameplay was observed for more than five days before prediction."
        action = (
            "Consider testing a re-engagement reminder for this cohort "
            "and investigate common points where activity stops."
        )

    elif predicted_inactive:
        rule_id = "model_flagged"
        reason = (
            "The predicted inactivity probability meets the "
            "frozen decision threshold."
        )
        action = (
            "Review the player activity and model explanation before "
            "choosing a retention experiment."
        )

    else:
        rule_id = "monitor"
        reason = (
            "The probability is below the decision threshold, "
            "with no earlier rule matching."
        )
        action = (
            "Continue monitoring observed activity; being below the "
            "threshold does not establish low risk."
        )

    return {
        "rule_id": rule_id,
        "decision": decision,
        "reason": reason,
        "suggested_action": action,
    }

def check_rule_boundaries(threshold):
    """Check rule precedence, boundaries, and invalid input handling."""
    def call(plays, recency, probability):
        return recommend(
            plays,
            recency,
            probability,
            threshold,
            RULE_CONFIG,
        )

    assert call(1, 7, 0.97)["rule_id"] == "single_observed_play"
    assert call(5, 2, 0.97)["rule_id"] == "recently_active"
    assert call(5, 5.01, 0.50)["rule_id"] == "extended_inactivity"
    assert call(5, 5, threshold)["rule_id"] == "model_flagged"
    assert call(5, 3, threshold - 0.01)["rule_id"] == "monitor"

    assert (
        call(5, 3, threshold)["decision"]
        == "At or above inactivity decision threshold"
    )

    for invalid_probability in [-0.1, 1.1, float("nan")]:
        try:
            call(5, 3, invalid_probability)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid probability was accepted.")
