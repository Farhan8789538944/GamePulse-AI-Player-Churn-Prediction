CREATE SCHEMA IF NOT EXISTS gamepulse;

CREATE TABLE IF NOT EXISTS gamepulse.players (
    device_id TEXT PRIMARY KEY,
    first_observed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS gamepulse.gameplay_events (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    device_id TEXT NOT NULL
        REFERENCES gamepulse.players(device_id),
    event_time TIMESTAMPTZ NOT NULL,
    score BIGINT NOT NULL CHECK (score >= 0),

    -- Matches our documented exact-row deduplication policy.
    UNIQUE (device_id, event_time, score)
);

CREATE TABLE IF NOT EXISTS gamepulse.feature_snapshots (
    snapshot_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    device_id TEXT NOT NULL
        REFERENCES gamepulse.players(device_id),

    first_event TIMESTAMPTZ NOT NULL,
    cutoff TIMESTAMPTZ NOT NULL,
    feature_version TEXT NOT NULL,

    -- Contains only the approved model input features.
    features JSONB NOT NULL
        CHECK (jsonb_typeof(features) = 'object'),

    data_partition TEXT NOT NULL
        CHECK (data_partition IN ('train', 'validation', 'test')),

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (cutoff > first_event),
    UNIQUE (device_id, cutoff, feature_version)
);

CREATE TABLE IF NOT EXISTS gamepulse.model_versions (
    model_version TEXT PRIMARY KEY,
    model_type TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL
        CHECK (artifact_sha256 ~ '^[0-9a-f]{64}$'),

    feature_names JSONB NOT NULL
        CHECK (jsonb_typeof(feature_names) = 'array'),

    decision_threshold DOUBLE PRECISION NOT NULL
        CHECK (decision_threshold > 0 AND decision_threshold < 1),

    evaluation_metrics JSONB NOT NULL
        CHECK (jsonb_typeof(evaluation_metrics) = 'object'),

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS gamepulse.segment_versions (
    segment_version TEXT PRIMARY KEY,
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL
        CHECK (artifact_sha256 ~ '^[0-9a-f]{64}$'),

    feature_names JSONB NOT NULL
        CHECK (jsonb_typeof(feature_names) = 'array'),

    segment_names JSONB NOT NULL
        CHECK (jsonb_typeof(segment_names) = 'object'),

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS gamepulse.player_segments (
    snapshot_id BIGINT NOT NULL
        REFERENCES gamepulse.feature_snapshots(snapshot_id),

    segment_version TEXT NOT NULL
        REFERENCES gamepulse.segment_versions(segment_version),

    segment_id INTEGER NOT NULL CHECK (segment_id >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (snapshot_id, segment_version)
);

CREATE TABLE IF NOT EXISTS gamepulse.predictions (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    snapshot_id BIGINT NOT NULL
        REFERENCES gamepulse.feature_snapshots(snapshot_id),

    model_version TEXT NOT NULL
        REFERENCES gamepulse.model_versions(model_version),

    churn_probability DOUBLE PRECISION NOT NULL
        CHECK (churn_probability >= 0 AND churn_probability <= 1),

    predicted_churn SMALLINT NOT NULL
        CHECK (predicted_churn IN (0, 1)),

    rules_version TEXT NOT NULL,
    recommendation JSONB NOT NULL
        CHECK (jsonb_typeof(recommendation) = 'object'),

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (snapshot_id, model_version, rules_version)
);

CREATE TABLE IF NOT EXISTS gamepulse.observed_outcomes (
    snapshot_id BIGINT NOT NULL
        REFERENCES gamepulse.feature_snapshots(snapshot_id),

    label_version TEXT NOT NULL,
    outcome_start TIMESTAMPTZ NOT NULL,
    outcome_end TIMESTAMPTZ NOT NULL,

    outcome_event_count INTEGER NOT NULL
        CHECK (outcome_event_count >= 0),

    churn SMALLINT NOT NULL CHECK (churn IN (0, 1)),

    -- Historical time at which the full window has elapsed.
    label_available_at TIMESTAMPTZ NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (snapshot_id, label_version),

    CHECK (outcome_end > outcome_start),
    CHECK (label_available_at >= outcome_end),

    CHECK (
        (churn = 1 AND outcome_event_count = 0)
        OR
        (churn = 0 AND outcome_event_count > 0)
    )
);

CREATE INDEX IF NOT EXISTS idx_snapshots_partition_cutoff
    ON gamepulse.feature_snapshots (data_partition, cutoff);

CREATE INDEX IF NOT EXISTS idx_predictions_model_probability
    ON gamepulse.predictions (model_version, churn_probability DESC);

CREATE INDEX IF NOT EXISTS idx_segments_version_id
    ON gamepulse.player_segments (segment_version, segment_id);

CREATE INDEX IF NOT EXISTS idx_outcomes_available_at
    ON gamepulse.observed_outcomes (label_available_at);