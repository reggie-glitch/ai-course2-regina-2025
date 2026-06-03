"""
Модуль инференса: загрузка артефактов и предсказание.
Используется из app.py — напрямую не запускается.
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Глобальный кэш: загружаем pkl один раз при старте сервиса
_BUNDLE = None


def load_artifacts(path: str) -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Артефакты не найдены: {path}\n"
                "Сначала обучи модель: python src/train.py"
            )
        with open(path, "rb") as f:
            _BUNDLE = pickle.load(f)
        log.info(f"Модель загружена из {path}")
    return _BUNDLE


def preprocess_input(raw: dict, bundle: dict) -> np.ndarray:
    """
    Принимает словарь {feature: value} и возвращает
    numpy-массив готовый к predict().
    """
    feature_cols = bundle["feature_cols"]
    cat_cols     = bundle["cat_cols"]
    encoders     = bundle["encoders"]
    scaler       = bundle["scaler"]

    df = pd.DataFrame([raw])

    # Проверка полноты признаков
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют признаки: {missing}")

    # Кодируем категориальные признаки
    for col in cat_cols:
        le  = encoders[col]
        val = str(df[col].iloc[0])
        if val not in le.classes_:
            log.warning(f"Неизвестное значение '{val}' для '{col}' → заменяем на первый класс")
            df[col] = 0
        else:
            df[col] = le.transform([val])

    X = df[feature_cols].values.astype(float)
    X = scaler.transform(X)
    return X


def predict(raw: dict, bundle: dict) -> dict:
    """
    Предсказывает: normal (0) или attack (1).

    Возвращает:
        {
            "prediction": "attack",
            "label": 1,
            "probability_attack": 0.97,
            "probability_normal": 0.03,
        }
    """
    model = bundle["model"]
    X = preprocess_input(raw, bundle)

    label = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]   # [P(normal), P(attack)]

    return {
        "prediction":         "attack" if label == 1 else "normal",
        "label":              label,
        "probability_attack": round(float(proba[1]), 4),
        "probability_normal": round(float(proba[0]), 4),
    }
