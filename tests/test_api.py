import copy

import pytest
from fastapi.testclient import TestClient

from gamepulse.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_features():
    return {
        "plays_7d": 1,
        "active_days_7d": 1,
        "days_since_last_play": 7.0,
        "activity_span_days": 0.0,
        "plays_first_3d": 1,
        "plays_last_3d": 0,
        "play_count_change": -1,
        "mean_score_7d": 0.0,
        "best_score_7d": 0.0,
        "score_std_7d": 0.0,
        "mean_gap_hours": None,
        "max_gap_hours": None,
    }


def test_liveness(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_database_readiness(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["model_version"] == "v1"


def test_validation_summary(client):
    response = client.get("/v1/summary")
    assert response.status_code == 200

    body = response.json()
    assert body["partition"] == "validation"
    assert body["snapshots"] == 4287
    assert body["scored_snapshots"] == 4287
    assert body["unscored_snapshots"] == 0
    assert body["at_or_above_threshold"] == 3555
    assert body["below_threshold"] == 732


def test_pagination_has_no_overlap(client):
    first_response = client.get("/v1/players?limit=2&offset=0")
    second_response = client.get("/v1/players?limit=2&offset=2")

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first = first_response.json()
    second = second_response.json()

    assert first["returned"] == second["returned"] == 2
    assert first["next_offset"] == 2
    assert second["next_offset"] == 4

    first_ids = {item["snapshot_id"] for item in first["items"]}
    second_ids = {item["snapshot_id"] for item in second["items"]}
    assert first_ids.isdisjoint(second_ids)


@pytest.mark.parametrize(
    "query",
    ["limit=101", "limit=0", "offset=-1"],
)
def test_invalid_pagination(client, query):
    response = client.get(f"/v1/players?{query}")
    assert response.status_code == 422


def test_unknown_device(client):
    response = client.get("/v1/players/nonexistent-device")
    assert response.status_code == 404


@pytest.mark.parametrize("offset", [0, 2000, 4200])
def test_fresh_prediction_matches_stored_result(client, offset):
    page_response = client.get(
        f"/v1/players?limit=1&offset={offset}"
    )
    assert page_response.status_code == 200

    device_id = page_response.json()["items"][0]["device_id"]

    stored_response = client.get(f"/v1/players/{device_id}")
    assert stored_response.status_code == 200
    stored = stored_response.json()

    fresh_response = client.post(
        "/v1/predict",
        json=stored["features"],
    )
    assert fresh_response.status_code == 200
    fresh = fresh_response.json()

    assert fresh["churn_probability"] == pytest.approx(
        stored["churn_probability"],
        rel=0,
        abs=1e-12,
    )
    assert (
        fresh["at_or_above_threshold"]
        == stored["at_or_above_threshold"]
    )
    assert fresh["decision_threshold"] == stored["decision_threshold"]
    assert fresh["persisted"] is False


def test_single_play_accepts_null_gaps(client, valid_features):
    response = client.post("/v1/predict", json=valid_features)
    assert response.status_code == 200

    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["model_version"] == "v1"
    assert body["decision_threshold"] == 0.94


@pytest.mark.parametrize(
    "changes",
    [
        {"plays_7d": -1},
        {"plays_7d": 1.5},
        {"plays_7d": True},
        {"plays_7d": "1"},
        {"active_days_7d": 8},
        {"days_since_last_play": 8},
        {"play_count_change": 0},
        {"mean_score_7d": 10, "best_score_7d": 5},
        {"mean_gap_hours": 0},
        {"churn": 1},
        {"outcome_event_count": 0},
    ],
)
def test_invalid_or_extra_features_rejected(
    client, valid_features, changes
):
    payload = copy.deepcopy(valid_features)
    payload.update(changes)

    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "missing_field",
    ["plays_7d", "mean_gap_hours"],
)
def test_required_features_cannot_be_omitted(
    client, valid_features, missing_field
):
    payload = copy.deepcopy(valid_features)
    del payload[missing_field]

    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422

@pytest.mark.parametrize("offset", [0, 2000, 4200])
def test_fresh_engagement_matches_stored_result(client, offset):
    page_response = client.get(
        f"/v1/players?limit=1&offset={offset}"
    )
    assert page_response.status_code == 200

    device_id = page_response.json()["items"][0]["device_id"]

    stored_response = client.get(f"/v1/players/{device_id}")
    assert stored_response.status_code == 200
    stored = stored_response.json()

    fresh_response = client.post(
        "/v1/predict",
        json=stored["features"],
    )
    assert fresh_response.status_code == 200
    fresh = fresh_response.json()

    # Includes segment, rule, reason, action, settings, and versions.
    assert fresh["recommendation"] == stored["recommendation"]
    assert fresh["rules_version"] == stored["rules_version"]
    assert fresh["persisted"] is False


def test_shared_rule_boundaries():
    from gamepulse.retention import check_rule_boundaries

    check_rule_boundaries(0.94)

def test_saved_model_performance(client):
    response = client.get("/v1/model-performance")
    assert response.status_code == 200

    body = response.json()

    assert body["model_version"] == "v1"
    assert body["evaluation_partition"] == "test"
    assert body["test_devices"] == 5084
    assert body["decision_threshold"] == 0.94
    assert body["artifact_hashes_verified"] is True

    assert body["confusion_matrix"]["class_order"] == [
        "returned", "inactive"
    ]
    assert body["confusion_matrix"]["counts"] == [
        [182, 104],
        [733, 4065],
    ]

    assert body["model_metrics"]["roc_auc"] == pytest.approx(
        0.7914730642429684
    )
    assert body["model_metrics"]["balanced_accuracy"] == pytest.approx(
        0.741795824017583
    )

    # A ranking-only baseline must not acquire invented metrics.
    recency = body["baselines"]["recency_ranking"]
    assert "roc_auc" in recency
    assert "brier_score" not in recency


def test_performance_rejects_wrong_artifact_hash(
    client, monkeypatch, tmp_path
):
    import json
    import gamepulse.performance as performance

    metrics = json.loads(
        performance.METRICS_PATH.read_text(encoding="utf-8-sig")
    )
    metrics["model_sha256"] = "0" * 64

    temporary_report = tmp_path / "metrics.json"
    temporary_report.write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )

    # Only this test uses the temporary report.
    # The real report and frozen artifacts are unchanged.
    monkeypatch.setattr(
        performance,
        "METRICS_PATH",
        temporary_report,
    )

    response = client.get("/v1/model-performance")
    assert response.status_code == 503

@pytest.mark.parametrize("offset", [0, 2000, 4200])
def test_player_shap_reconstructs_prediction(client, offset):
    import math

    page_response = client.get(
        f"/v1/players?limit=1&offset={offset}"
    )
    assert page_response.status_code == 200

    stored = page_response.json()["items"][0]
    device_id = stored["device_id"]

    response = client.get(
        f"/v1/players/{device_id}/explanation"
    )
    assert response.status_code == 200

    body = response.json()

    assert body["partition"] == "validation"
    assert body["background_partition"] == "train"
    assert body["background_devices"] == 200
    assert body["units"] == "log_odds_of_inactivity"
    assert body["reconstruction_verified"] is True

    assert len(body["contributions"]) == 14
    assert len({
        item["feature"] for item in body["contributions"]
    }) == 14

    total_contribution = sum(
        item["shap_log_odds"]
        for item in body["contributions"]
    )

    # Independently reconstruct probability from the returned contributions.
    reconstructed = 1 / (
        1 + math.exp(
            -(body["base_log_odds"] + total_contribution)
        )
    )

    assert reconstructed == pytest.approx(
        stored["churn_probability"],
        rel=0,
        abs=1e-6,
    )
    assert body["churn_probability"] == pytest.approx(
        stored["churn_probability"],
        rel=0,
        abs=1e-12,
    )
    assert total_contribution == pytest.approx(
        body["contribution_sum"]
    )


def test_explanation_unknown_device(client):
    response = client.get(
        "/v1/players/nonexistent-device/explanation"
    )
    assert response.status_code == 404