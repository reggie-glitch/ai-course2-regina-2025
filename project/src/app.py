"""
FastAPI сервис IDS — бинарный классификатор сетевых вторжений.

Эндпоинты:
  GET  /health   — проверка работоспособности
  GET  /metrics  — счётчики запросов и предсказаний
  POST /predict  — классификация соединения: normal / attack
"""

import os
import sys
import time
import logging
from collections import defaultdict
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Добавляем src/ в путь (нужно когда запускаем из корня project/)
sys.path.insert(0, os.path.dirname(__file__))
from predict import load_artifacts, predict as run_predict

# ── Логирование ───────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("ids_service")

# ── Конфиг и пути ─────────────────────────────────────────────────────────────

CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/config.yaml")
MODEL_PATH  = os.getenv("MODEL_PATH",  "data/model_artifacts.pkl")

with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# ── Глобальное состояние ──────────────────────────────────────────────────────

BUNDLE   = {}
COUNTERS = defaultdict(int)
# COUNTERS["requests_total"]
# COUNTERS["predictions_normal"]
# COUNTERS["predictions_attack"]
# COUNTERS["errors_total"]

# ── Lifespan: загружаем модель один раз при старте ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global BUNDLE
    log.info("Загрузка модели...")
    BUNDLE = load_artifacts(MODEL_PATH)
    log.info("Сервис готов.")
    yield
    log.info("Сервис остановлен.")

# ── Приложение ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IDS — Network Intrusion Detection",
    description="Бинарный классификатор сетевого трафика: **normal** vs **attack**. Датасет NSL-KDD.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Middleware: логируем каждый запрос ────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    ms = (time.time() - start) * 1000
    log.info(f"{request.method} {request.url.path} → {response.status_code} ({ms:.1f}ms)")
    COUNTERS["requests_total"] += 1
    return response

# ── Схемы данных ──────────────────────────────────────────────────────────────

class ConnectionFeatures(BaseModel):
    """41 числовой/категориальный признак сетевого соединения (NSL-KDD)."""

    duration:                     float = Field(..., description="Длительность соединения (сек)")
    protocol_type:                str   = Field(..., description="Протокол: tcp / udp / icmp")
    service:                      str   = Field(..., description="Сервис: http, ftp, smtp, ...")
    flag:                         str   = Field(..., description="Статус: SF, S0, REJ, ...")
    src_bytes:                    float = Field(...)
    dst_bytes:                    float = Field(...)
    land:                         int   = Field(...)
    wrong_fragment:               float = Field(...)
    urgent:                       float = Field(...)
    hot:                          float = Field(...)
    num_failed_logins:            float = Field(...)
    logged_in:                    int   = Field(...)
    num_compromised:              float = Field(...)
    root_shell:                   float = Field(...)
    su_attempted:                 float = Field(...)
    num_root:                     float = Field(...)
    num_file_creations:           float = Field(...)
    num_shells:                   float = Field(...)
    num_access_files:             float = Field(...)
    num_outbound_cmds:            float = Field(...)
    is_host_login:                int   = Field(...)
    is_guest_login:               int   = Field(...)
    count:                        float = Field(...)
    srv_count:                    float = Field(...)
    serror_rate:                  float = Field(...)
    srv_serror_rate:              float = Field(...)
    rerror_rate:                  float = Field(...)
    srv_rerror_rate:              float = Field(...)
    same_srv_rate:                float = Field(...)
    diff_srv_rate:                float = Field(...)
    srv_diff_host_rate:           float = Field(...)
    dst_host_count:               float = Field(...)
    dst_host_srv_count:           float = Field(...)
    dst_host_same_srv_rate:       float = Field(...)
    dst_host_diff_srv_rate:       float = Field(...)
    dst_host_same_src_port_rate:  float = Field(...)
    dst_host_srv_diff_host_rate:  float = Field(...)
    dst_host_serror_rate:         float = Field(...)
    dst_host_srv_serror_rate:     float = Field(...)
    dst_host_rerror_rate:         float = Field(...)
    dst_host_srv_rerror_rate:     float = Field(...)

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    prediction:         str   = Field(..., description="normal или attack")
    label:              int   = Field(..., description="0=normal, 1=attack")
    probability_attack: float = Field(...)
    probability_normal: float = Field(...)


# ── Эндпоинты ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Observability"])
def health():
    """Проверка работоспособности сервиса и наличия модели."""
    ok = bool(BUNDLE)
    return {"status": "ok" if ok else "degraded", "model_loaded": ok}


@app.get("/metrics", tags=["Observability"])
def metrics():
    """Счётчики: запросы, предсказания по классам, ошибки."""
    return {"counters": dict(COUNTERS)}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(features: ConnectionFeatures):
    """
    Классифицирует сетевое соединение.

    - **normal** — обычный трафик
    - **attack** — вторжение / аномалия
    """
    if not BUNDLE:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        result = run_predict(features.model_dump(), BUNDLE)
    except ValueError as e:
        COUNTERS["errors_total"] += 1
        log.warning(f"Ошибка входных данных: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        COUNTERS["errors_total"] += 1
        log.error(f"Внутренняя ошибка: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

    COUNTERS[f"predictions_{result['prediction']}"] += 1
    log.info(
        f"Результат: {result['prediction']} "
        f"(P_attack={result['probability_attack']:.3f})"
    )
    return result


# ── Прямой запуск ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=os.getenv("SERVICE_HOST", CONFIG["service"]["host"]),
        port=int(os.getenv("SERVICE_PORT", CONFIG["service"]["port"])),
        reload=False,
    )
