"""
Обучение моделей IDS (бинарная классификация: normal vs attack).
Запускать из папки project/:  python src/train.py

Этапы:
  1. Загрузка данных NSL-KDD
  2. Предобработка (бинарные метки, кодирование, масштабирование)
  3. Обучение 4 моделей: LogReg → DecisionTree → RandomForest → XGBoost
  4. Сравнение по F1, Precision, Recall
  5. Сохранение лучшей модели + артефактов предобработки
"""

import os
import sys
import pickle
import logging

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── 1. Конфиг ─────────────────────────────────────────────────────────────────

def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 2. Загрузка данных ────────────────────────────────────────────────────────

def load_data(cfg):
    columns = cfg["data"]["columns"]
    train_path = cfg["data"]["train_path"]
    test_path  = cfg["data"]["test_path"]

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        log.error("Данные не найдены! Сначала запусти: python data/download_data.py")
        sys.exit(1)

    df_train = pd.read_csv(train_path, header=None, names=columns)
    df_test  = pd.read_csv(test_path,  header=None, names=columns)
    log.info(f"Train: {df_train.shape}, Test: {df_test.shape}")
    return df_train, df_test


# ── 3. Предобработка ──────────────────────────────────────────────────────────

def make_binary_label(series):
    """'normal' → 0, всё остальное (любая атака) → 1."""
    return (series != "normal").astype(int)


def preprocess(df_train, df_test, cfg):
    cat_cols  = cfg["data"]["categorical_cols"]
    drop_cols = ["label", "difficulty"]
    feature_cols = [c for c in df_train.columns if c not in drop_cols]

    # Бинарные метки
    y_train = make_binary_label(df_train["label"]).values
    y_test  = make_binary_label(df_test["label"]).values

    log.info(
        f"Train: {(y_train==0).sum()} normal, {(y_train==1).sum()} attack  "
        f"| Test: {(y_test==0).sum()} normal, {(y_test==1).sum()} attack"
    )

    # Label-encoding категориальных признаков
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([df_train[col], df_test[col]], axis=0).astype(str)
        le.fit(combined)
        df_train[col] = le.transform(df_train[col].astype(str))
        df_test[col]  = le.transform(df_test[col].astype(str))
        encoders[col] = le

    X_train = df_train[feature_cols].values.astype(float)
    X_test  = df_test[feature_cols].values.astype(float)

    # Масштабирование: fit только на train!
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    artifacts = {
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "encoders": encoders,
        "scaler": scaler,
    }
    return X_train, X_test, y_train, y_test, artifacts


# ── 4. Обучение и оценка ──────────────────────────────────────────────────────

def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    log.info(f"\n{'='*55}")
    log.info(f"  {name}  |  F1={f1:.4f}  |  ROC-AUC={auc:.4f}")
    log.info(f"{'='*55}")
    print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    return f1, auc


def train_all(X_train, X_test, y_train, y_test, cfg):
    rs = cfg["model"]["random_state"]
    results = {}

    # ── Logistic Regression (baseline) ──────────────────────────────────────
    log.info("Обучаем: Logistic Regression (baseline)...")
    lr = LogisticRegression(max_iter=1000, random_state=rs, n_jobs=-1)
    lr.fit(X_train, y_train)
    results["LogisticRegression"] = (lr, *evaluate("Logistic Regression", lr, X_test, y_test))

    # ── Decision Tree ────────────────────────────────────────────────────────
    log.info("Обучаем: Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=15, random_state=rs)
    dt.fit(X_train, y_train)
    results["DecisionTree"] = (dt, *evaluate("Decision Tree", dt, X_test, y_test))

    # ── Random Forest ────────────────────────────────────────────────────────
    log.info("Обучаем: Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=rs, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["RandomForest"] = (rf, *evaluate("Random Forest", rf, X_test, y_test))

    # ── XGBoost (финальная модель) ───────────────────────────────────────────
    log.info("Обучаем: XGBoost (финальная модель)...")
    p = cfg["model"]["xgb_params"]
    xgb = XGBClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        random_state=rs,
        n_jobs=-1,
        verbosity=0,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    results["XGBoost"] = (xgb, *evaluate("XGBoost", xgb, X_test, y_test))

    return results


# ── 5. Выбор лучшей и сохранение ─────────────────────────────────────────────

def select_best(results):
    # results[name] = (model, f1, auc)
    best = max(results.items(), key=lambda kv: kv[1][1])  # по F1
    name, (model, f1, auc) = best
    log.info(f"\n>>> Лучшая модель: {name}  F1={f1:.4f}  ROC-AUC={auc:.4f}")
    return name, model


def save_artifacts(model, artifacts, cfg):
    path = cfg["model"]["artifact_path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {"model": model, **artifacts}
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    log.info(f"Артефакты сохранены → {path}")


# ── Точка входа ───────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    log.info("=== 1. Загрузка данных ===")
    df_train, df_test = load_data(cfg)

    log.info("=== 2. Предобработка ===")
    X_train, X_test, y_train, y_test, artifacts = preprocess(df_train, df_test, cfg)

    log.info("=== 3. Обучение моделей ===")
    results = train_all(X_train, X_test, y_train, y_test, cfg)

    log.info("=== 4. Выбор финальной модели ===")
    name, model = select_best(results)

    log.info("=== 5. Сохранение ===")
    save_artifacts(model, artifacts, cfg)

    log.info(f"=== Готово! Финальная модель: {name} ===")


if __name__ == "__main__":
    main()
