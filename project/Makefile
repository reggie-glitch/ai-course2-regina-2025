# Makefile — удобные команды для работы с проектом
# Запускать из папки project/

.PHONY: install data train run test docker-build docker-run clean

## Установка зависимостей
install:
	pip install -r requirements.txt

## Загрузка датасета NSL-KDD
data:
	python3 data/download_data.py

## Обучение модели
train:
	python3 src/train.py

## Запуск сервиса локально
run:
	python3 -m uvicorn src.app:app --host 0.0.0.0 --port 8000

## Запуск тестов
test:
	python3 -m pytest tests/ -v

## Сборка Docker-образа
docker-build:
	docker-compose build

## Запуск через Docker
docker-run:
	docker-compose up

## Полный pipeline: данные → обучение → запуск
all: data train run

## Очистка артефактов
clean:
	rm -f data/model_artifacts.pkl
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
