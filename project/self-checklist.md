# Само-чеклист перед сдачей проекта

## Постановка задачи
- [x] Описан пользователь и его боль
- [x] Сформулирована задача (бинарная классификация)
- [x] Выбрана и обоснована метрика (F1-score weighted)
- [x] Описаны ограничения и риски (report.md §8)

## Данные
- [x] Используется открытый датасет NSL-KDD
- [x] Данные загружаются скриптом (data/download_data.py)
- [x] Проведён EDA (notebooks/01_eda_and_models.ipynb)
- [x] Описан препроцессинг (LabelEncoder, StandardScaler)
- [x] Нет персональных данных в репозитории
- [x] Большие файлы в .gitignore (*.csv, *.pkl)

## Модели
- [x] Реализован baseline (Logistic Regression)
- [x] Реализованы дополнительные модели (Decision Tree, Random Forest, XGBoost)
- [x] Модели сравниваются по метрикам
- [x] Выбор финальной модели обоснован
- [x] Confusion matrix и classification_report зафиксированы

## Сервис
- [x] Реализован FastAPI сервис (src/app.py)
- [x] Есть /health эндпоинт
- [x] Есть /metrics эндпоинт
- [x] Есть /predict эндпоинт с валидацией входных данных (Pydantic)
- [x] Логируется каждый запрос (время, статус)
- [x] Понятные HTTP-коды ошибок (422, 500, 503)

## Безопасность и конфиг
- [x] Секреты не в репозитории
- [x] Есть configs/.env.example
- [x] Реальный .env в .gitignore
- [x] Настройки вынесены в configs/config.yaml

## Контейнеризация
- [x] Есть Dockerfile
- [x] Есть docker-compose.yml
- [x] Сервис запускается командой docker-compose up --build

## Тесты
- [x] Юнит-тесты predict.py (tests/test_predict.py)
- [x] Smoke-тесты API (tests/test_api.py)
- [x] Тесты запускаются командой pytest tests/ -v

## Документация
- [x] Заполнен README.md (паспорт проекта)
- [x] Заполнен report.md (описание экспериментов)
- [x] Заполнен self-checklist.md (этот файл)
- [x] Есть инструкция по запуску (README.md §Быстрый старт)
