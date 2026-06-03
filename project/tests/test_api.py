"""
Smoke-тесты FastAPI сервиса.
Запускать из папки project/:  pytest tests/ -v

Используем TestClient от FastAPI — реальный HTTP-сервер не нужен.
Модель подменяем фиктивным бандлом (mock), чтобы тесты работали
без обученных артефактов.
"""

import os
import sys
import pickle
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Добавляем src/ в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Минимальный тестовый бандл (такой же, как в test_predict.py) ──────────────

FEATURE_COLS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
CAT_COLS = ["protocol_type", "service", "flag"]

SAMPLE_PAYLOAD = {
    "duration": 0, "protocol_type": "tcp", "service": "http",
    "flag": "SF", "src_bytes": 232, "dst_bytes": 8153,
    "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
    "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
    "root_shell": 0, "su_attempted": 0, "num_root": 0,
    "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
    "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
    "count": 9, "srv_count": 9, "serror_rate": 0.0,
    "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
    "dst_host_count": 9, "dst_host_srv_count": 9,
    "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 0.11, "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
}


def make_mock_bundle():
    encoders = {}
    for col, vals in {
        "protocol_type": ["tcp", "udp", "icmp"],
        "service":        ["http", "ftp", "smtp", "other"],
        "flag":           ["SF", "S0", "REJ", "RSTO"],
    }.items():
        le = LabelEncoder()
        le.fit(vals)
        encoders[col] = le

    # Обучаем tiny-модель на двух примерах
    X = np.zeros((2, len(FEATURE_COLS)))
    y = np.array([0, 1])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_s, y)

    return {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "cat_cols": CAT_COLS,
        "encoders": encoders,
        "scaler": scaler,
    }


# ── Фикстура: TestClient с подменённым бандлом ────────────────────────────────

@pytest.fixture
def client():
    mock_bundle = make_mock_bundle()

    # Подменяем load_artifacts и BUNDLE ещё до импорта app
    with patch("predict.load_artifacts", return_value=mock_bundle):
        import app as app_module
        app_module.BUNDLE = mock_bundle   # подкладываем напрямую

        with TestClient(app_module.app) as c:
            yield c

        # Сбрасываем глобальные счётчики между тестами
        app_module.COUNTERS.clear()


# ── Тесты ─────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_field(self, client):
        r = client.get("/health")
        assert "status" in r.json()

    def test_health_model_loaded_true(self, client):
        r = client.get("/health")
        assert r.json()["model_loaded"] is True


class TestMetrics:
    def test_metrics_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_has_counters(self, client):
        r = client.get("/metrics")
        assert "counters" in r.json()

    def test_metrics_counts_requests(self, client):
        client.get("/health")
        client.get("/health")
        r = client.get("/metrics")
        # После 2 запросов /health счётчик должен быть >= 2
        assert r.json()["counters"].get("requests_total", 0) >= 2


class TestPredict:
    def test_predict_returns_200(self, client):
        r = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert r.status_code == 200

    def test_predict_response_schema(self, client):
        r = client.post("/predict", json=SAMPLE_PAYLOAD)
        body = r.json()
        assert "prediction" in body
        assert "label" in body
        assert "probability_attack" in body
        assert "probability_normal" in body

    def test_predict_prediction_is_valid_class(self, client):
        r = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert r.json()["prediction"] in {"normal", "attack"}

    def test_predict_label_is_binary(self, client):
        r = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert r.json()["label"] in {0, 1}

    def test_predict_probabilities_sum_to_one(self, client):
        r = client.post("/predict", json=SAMPLE_PAYLOAD)
        body = r.json()
        total = body["probability_attack"] + body["probability_normal"]
        assert abs(total - 1.0) < 0.01

    def test_predict_missing_field_returns_422(self, client):
        bad_payload = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "src_bytes"}
        r = client.post("/predict", json=bad_payload)
        assert r.status_code == 422

    def test_predict_wrong_type_returns_422(self, client):
        bad_payload = {**SAMPLE_PAYLOAD, "duration": "not_a_number"}
        r = client.post("/predict", json=bad_payload)
        assert r.status_code == 422

    def test_predict_increments_counter(self, client):
        client.post("/predict", json=SAMPLE_PAYLOAD)
        r = client.get("/metrics")
        counters = r.json()["counters"]
        total_predictions = (
            counters.get("predictions_normal", 0) +
            counters.get("predictions_attack", 0)
        )
        assert total_predictions >= 1
