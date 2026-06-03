"""
Тесты модуля predict.py.
Запускать из папки project/:  pytest tests/ -v
"""

import os
import sys
import pickle
import tempfile

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import preprocess_input, predict, load_artifacts


# ── Фикстуры ──────────────────────────────────────────────────────────────────

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

NORMAL_SAMPLE = {
    "duration": 0, "protocol_type": "tcp", "service": "http",
    "flag": "SF", "src_bytes": 232, "dst_bytes": 8153,
    "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
    "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
    "root_shell": 0, "su_attempted": 0, "num_root": 0,
    "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
    "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
    "count": 9, "srv_count": 9, "serror_rate": 0.0,
    "srv_serror_rate": 0.0, "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
    "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
    "dst_host_count": 9, "dst_host_srv_count": 9,
    "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 0.11,
    "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
}

ATTACK_SAMPLE = {**NORMAL_SAMPLE, "src_bytes": 0, "dst_bytes": 0,
                 "serror_rate": 1.0, "srv_serror_rate": 1.0,
                 "dst_host_serror_rate": 1.0, "count": 511,
                 "srv_count": 511, "flag": "S0", "logged_in": 0}


def make_bundle():
    """Создаём минимальный тестовый бандл с настоящим sklearn-пайплайном."""
    import pandas as pd

    # Кодируем категориальные признаки
    encoders = {}
    for col, vals in {
        "protocol_type": ["tcp", "udp", "icmp"],
        "service":        ["http", "ftp", "smtp", "other"],
        "flag":           ["SF", "S0", "REJ", "RSTO"],
    }.items():
        le = LabelEncoder()
        le.fit(vals)
        encoders[col] = le

    # Простая линейная модель, обученная на двух примерах
    rows = []
    for raw in [NORMAL_SAMPLE, ATTACK_SAMPLE]:
        row = []
        for col in FEATURE_COLS:
            val = raw[col]
            if col in CAT_COLS:
                val = encoders[col].transform([str(val)])[0]
            row.append(float(val))
        rows.append(row)

    X = np.array(rows)
    y = np.array([0, 1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    return {
        "model":        model,
        "feature_cols": FEATURE_COLS,
        "cat_cols":     CAT_COLS,
        "encoders":     encoders,
        "scaler":       scaler,
    }


@pytest.fixture
def bundle():
    return make_bundle()


# ── Тесты preprocess_input ────────────────────────────────────────────────────

def test_preprocess_returns_correct_shape(bundle):
    X = preprocess_input(NORMAL_SAMPLE, bundle)
    assert X.shape == (1, len(FEATURE_COLS)), "Должен вернуть массив (1, 41)"


def test_preprocess_missing_feature_raises(bundle):
    bad = {k: v for k, v in NORMAL_SAMPLE.items() if k != "src_bytes"}
    with pytest.raises(ValueError, match="Отсутствуют признаки"):
        preprocess_input(bad, bundle)


def test_preprocess_unknown_categorical_no_crash(bundle):
    """Неизвестное категориальное значение не должно ронять сервис."""
    sample = {**NORMAL_SAMPLE, "protocol_type": "UNKNOWN_PROTO"}
    X = preprocess_input(sample, bundle)  # должен отработать без исключения
    assert X.shape == (1, len(FEATURE_COLS))


# ── Тесты predict ─────────────────────────────────────────────────────────────

def test_predict_returns_expected_keys(bundle):
    result = predict(NORMAL_SAMPLE, bundle)
    assert set(result.keys()) == {
        "prediction", "label", "probability_attack", "probability_normal"
    }


def test_predict_label_matches_prediction(bundle):
    result = predict(NORMAL_SAMPLE, bundle)
    if result["prediction"] == "normal":
        assert result["label"] == 0
    else:
        assert result["label"] == 1


def test_predict_probabilities_sum_to_one(bundle):
    result = predict(NORMAL_SAMPLE, bundle)
    total = result["probability_attack"] + result["probability_normal"]
    assert abs(total - 1.0) < 1e-3, f"Вероятности должны суммироваться в 1, получили {total}"


def test_predict_probabilities_in_range(bundle):
    for sample in [NORMAL_SAMPLE, ATTACK_SAMPLE]:
        result = predict(sample, bundle)
        assert 0.0 <= result["probability_attack"] <= 1.0
        assert 0.0 <= result["probability_normal"] <= 1.0


# ── Тест load_artifacts ───────────────────────────────────────────────────────

def test_load_artifacts_file_not_found():
    import predict as pred_module
    pred_module._BUNDLE = None  # сбрасываем кэш
    with pytest.raises(FileNotFoundError):
        load_artifacts("/nonexistent/path/model.pkl")


def test_load_artifacts_loads_correctly(bundle):
    import predict as pred_module
    pred_module._BUNDLE = None  # сбрасываем кэш

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(bundle, f)
        tmp_path = f.name

    try:
        loaded = load_artifacts(tmp_path)
        assert "model" in loaded
        assert "scaler" in loaded
        assert "encoders" in loaded
    finally:
        os.unlink(tmp_path)
        pred_module._BUNDLE = None  # чистим кэш после теста
