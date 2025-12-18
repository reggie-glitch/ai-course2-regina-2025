
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse


from .core import load_data, compute_quality_flags, compute_basic_stats


app = FastAPI(
    title="EDA API",
    description="HTTP API для анализа качества данных",
    version="0.2.0"
)


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Проверка работоспособности сервиса."""
    return {"status": "healthy", "service": "eda-api"}


@app.post("/quality")
def compute_quality(
    n_rows: int = Query(..., description="Количество строк в данных"),
    n_cols: int = Query(..., description="Количество колонок в данных"),
    missing_ratio: float = Query(..., description="Доля пропущенных значений"),
    duplicate_ratio: float = Query(0.0, description="Доля дубликатов"),
) -> Dict[str, Any]:
    
    # Эвристики из семинара
    too_few_rows = bool(n_rows < 100)
    too_many_missing = bool(missing_ratio > 0.3)
    too_many_duplicates = bool(duplicate_ratio > 0.1)
    
    # Интегральный показатель качества (0-100)
    quality_score = 100
    if too_few_rows:
        quality_score -= 20
    if too_many_missing:
        quality_score -= min(int(missing_ratio * 100), 40)
    if too_many_duplicates:
        quality_score -= min(int(duplicate_ratio * 100), 30)
    
    ok_for_model = bool(quality_score >= 70)
    
    return {
        "quality_score": int(quality_score),
        "ok_for_model": ok_for_model,
        "flags": {
            "too_few_rows": too_few_rows,
            "too_many_missing": too_many_missing,
            "too_many_duplicates": too_many_duplicates,
        },
        "metadata": {
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "missing_ratio": float(missing_ratio),
            "duplicate_ratio": float(duplicate_ratio),
        }
    }


@app.post("/quality-from-csv")
async def compute_quality_from_csv(
    file: UploadFile = File(..., description="CSV файл для анализа")
) -> Dict[str, Any]:
    """
    Анализ качества данных из CSV файла.
    
    Основа - код из семинара S04.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только CSV файлы"
        )
    
    try:
        # Сохраняем файл во временную директорию
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Загружаем данные
        df = pd.read_csv(tmp_path)
        
        # Вычисляем метрики с явным преобразованием типов
        n_rows, n_cols = df.shape
        missing_count = int(df.isnull().sum().sum())
        duplicate_count = int(df.duplicated().sum())
        
        missing_ratio = float(missing_count / (n_rows * n_cols)) if n_rows * n_cols > 0 else 0.0
        duplicate_ratio = float(duplicate_count / n_rows) if n_rows > 0 else 0.0
        
        # Вызываем существующий эндпоинт
        return compute_quality(
            n_rows=int(n_rows),
            n_cols=int(n_cols),
            missing_ratio=float(missing_ratio),
            duplicate_ratio=float(duplicate_ratio)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке файла: {str(e)}"
        )
    finally:
        # Удаляем временный файл
        if 'tmp_path' in locals():
            Path(tmp_path).unlink(missing_ok=True)


@app.post("/quality-flags-from-csv")
async def compute_quality_flags_from_csv(
    file: UploadFile = File(..., description="CSV файл для анализа"),
    high_cardinality_threshold: int = Query(50, description="Порог для высококардинальных категориальных признаков"),
    zero_threshold: float = Query(0.3, description="Порог для нулевых значений в числовых колонках"),
    min_missing_share: float = Query(0.1, description="Минимальная доля пропусков для флагирования колонки"),
) -> Dict[str, Any]:
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только CSV файлы"
        )
    
    try:
        # Сохраняем файл во временную директорию
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Загружаем данные используя нашу функцию из HW03
        df = load_data(tmp_path)
        
        # Вычисляем базовые метрики
        n_rows, n_cols = df.shape
        missing_count = int(df.isnull().sum().sum())
        duplicate_count = int(df.duplicated().sum())
        
        missing_ratio = float(missing_count / (n_rows * n_cols)) if n_rows * n_cols > 0 else 0.0
        duplicate_ratio = float(duplicate_count / n_rows) if n_rows > 0 else 0.0
        
        # Эвристики из семинара
        too_few_rows = bool(n_rows < 100)
        too_many_missing = bool(missing_ratio > 0.3)
        too_many_duplicates = bool(duplicate_ratio > 0.1)
        
        # ВЫПОЛНЕНИЕ HW04: Используем наши эвристики из HW03
        quality_info = compute_quality_flags(
            df,
            high_cardinality_threshold=high_cardinality_threshold,
            zero_threshold=zero_threshold,
            min_missing_share=min_missing_share
        )
        
        # Формируем полный ответ с флагами из HW03
        response = {
            "quality_score": quality_info.get("quality_score", 100),
            "ok_for_model": bool(quality_info.get("quality_score", 100) >= 70),
            "flags": {
                # Флаги из семинара
                "too_few_rows": too_few_rows,
                "too_many_missing": too_many_missing,
                "too_many_duplicates": too_many_duplicates,
                
                # НОВЫЕ ФЛАГИ ИЗ HW03:
                "has_constant_columns": bool(quality_info.get("has_constant_columns", False)),
                "has_high_cardinality_categoricals": bool(quality_info.get("has_high_cardinality", False)),
                "has_suspicious_id_duplicates": bool(quality_info.get("has_id_duplicates", False)),
                "has_many_zero_values": bool(quality_info.get("has_many_zeros", False)),
            },
            "details": {
                # Детали по новым эвристикам
                "constant_columns": quality_info.get("constant_columns", []),
                "high_cardinality_columns": quality_info.get("high_cardinality_cols", []),
                "high_cardinality_details": quality_info.get("high_cardinality_details", []),
                "duplicate_id_info": quality_info.get("duplicate_id_info", {}),
                "zero_columns": quality_info.get("zero_columns", []),
                "columns_with_missing": quality_info.get("columns_with_missing", []),
            },
            "metadata": {
                "n_rows": int(n_rows),
                "n_cols": int(n_cols),
                "missing_ratio": float(missing_ratio),
                "duplicate_ratio": float(duplicate_ratio),
                "missing_count": int(missing_count),
                "duplicate_count": int(duplicate_count),
                "parameters": {
                    "high_cardinality_threshold": int(high_cardinality_threshold),
                    "zero_threshold": float(zero_threshold),
                    "min_missing_share": float(min_missing_share),
                }
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке файла: {str(e)}"
        )
    finally:
        # Удаляем временный файл
        if 'tmp_path' in locals():
            Path(tmp_path).unlink(missing_ok=True)


@app.get("/")
def root():
    """Корневой эндпоинт с информацией о доступных методах."""
    return {
        "service": "EDA API",
        "version": "0.2.0",
        "description": "HTTP API для анализа качества данных",
        "endpoints": {
            "GET /health": "Проверка работоспособности сервиса",
            "POST /quality": "Анализ качества по метрикам",
            "POST /quality-from-csv": "Анализ качества из CSV файла",
            "POST /quality-flags-from-csv": "Полный анализ с эвристиками из HW03 (новый)"
        },
        "hw04_features": [
            "Новый эндпоинт /quality-flags-from-csv",
            "Использование константныx колонок, высококардинальных признаков и др.",
            "Параметризация порогов через query-параметры"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
