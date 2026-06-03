# IDS — Классификатор сетевых вторжений

**Курс:** Инженерия Искусственного Интеллекта  
**Датасет:** NSL-KDD  
**Задача:** Бинарная классификация сетевого трафика — `normal` vs `attack`  
**Уровень:** [BASIC]

---

## Постановка задачи

**Пользователь:** аналитик SOC (Security Operations Center).  
**Боль:** поток сетевых соединений слишком велик для ручного анализа.  
**Решение:** ML-сервис, который автоматически помечает каждое соединение как нормальное или атаку.

**Метрика успеха:** F1-score (weighted) ≥ 0.95 на тестовой выборке NSL-KDD.  
Выбор F1 обоснован дисбалансом классов и тем, что нам одинаково важны и recall (не пропустить атаку) и precision (не заваливать аналитика ложными срабатываниями).

---

## Данные

| Параметр | Значение |
|---|---|
| Датасет | NSL-KDD (улучшенный KDD Cup 1999) |
| Train | 125 973 записи |
| Test  | 22 544 записи |
| Признаки | 41 (числовые + 3 категориальных) |
| Метка | `normal` → 0, любая атака → 1 |

Признаки описывают параметры TCP/IP-соединения: длительность, протокол, сервис, флаги, число байт, статистические признаки по окну последних соединений и т.д.

---

## Структура репозитория

```
project/
├── README.md                  ← этот файл (паспорт проекта)
├── report.md                  ← отчёт: данные, эксперименты, результаты
├── self-checklist.md          ← чеклист самопроверки
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── configs/
│   ├── config.yaml            ← все настройки (пути, гиперпараметры)
│   └── .env.example           ← шаблон переменных окружения
│
├── data/
│   └── download_data.py       ← скрипт загрузки NSL-KDD
│
├── src/
│   ├── train.py               ← обучение 4 моделей + сохранение лучшей
│   ├── predict.py             ← инференс (загрузка артефактов, predict)
│   └── app.py                 ← FastAPI сервис
│
├── notebooks/
│   └── 01_eda_and_models.ipynb ← EDA + визуализация экспериментов
│
└── tests/
    ├── test_predict.py        ← юнит-тесты predict.py
    └── test_api.py            ← smoke-тесты FastAPI
```

---

## Быстрый старт (локально)

### 1. Установка зависимостей

```bash
cd project/

# Создаём виртуальное окружение (рекомендуется)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Загрузка данных

```bash
python data/download_data.py
# Скачает KDDTrain+.csv и KDDTest+.csv в папку data/
```

### 3. Обучение модели

```bash
python src/train.py
# Обучит 4 модели, сравнит метрики, сохранит лучшую в data/model_artifacts.pkl
```

Пример вывода:
```
=== 1. Загрузка данных ===
Train: (125973, 43), Test: (22544, 43)

=== 3. Обучение моделей ===
...
>>> Лучшая модель: XGBoost  F1=0.9981  ROC-AUC=0.9997
=== Готово! ===
```

### 4. Запуск сервиса

```bash
# Из папки project/
uvicorn src.app:app --host 0.0.0.0 --port 8000

# Интерактивная документация:
# http://localhost:8000/docs
```

### 5. Запуск тестов

```bash
pytest tests/ -v
```

---

## Быстрый старт (Docker)

```bash
# 1. Сначала обучи модель локально (нужен data/model_artifacts.pkl)
python src/train.py

# 2. Собери и запусти контейнер
docker-compose up --build

# 3. Проверь
curl http://localhost:8000/health
```

---

## API

### `GET /health`
Проверка работоспособности.
```json
{"status": "ok", "model_loaded": true}
```

### `GET /metrics`
Счётчики запросов и предсказаний.
```json
{
  "counters": {
    "requests_total": 42,
    "predictions_normal": 35,
    "predictions_attack": 7,
    "errors_total": 0
  }
}
```

### `POST /predict`
Классификация соединения.

**Пример запроса:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
  }'
```

**Ответ:**
```json
{
  "prediction": "normal",
  "label": 0,
  "probability_attack": 0.0021,
  "probability_normal": 0.9979
}
```

---

## Модели и результаты

| Модель | F1 (weighted) | ROC-AUC | Примечание |
|---|---|---|---|
| Logistic Regression | ~0.98 | ~0.99 | Baseline |
| Decision Tree | ~0.99 | ~0.99 | |
| Random Forest | ~0.998 | ~0.9995 | |
| **XGBoost** | **~0.998** | **~0.9997** | ✅ Финальная |

Подробные результаты и confusion matrix — в `report.md` и `notebooks/01_eda_and_models.ipynb`.
