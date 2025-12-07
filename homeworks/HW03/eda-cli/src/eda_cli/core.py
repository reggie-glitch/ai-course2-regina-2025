import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV file."""
    return pd.read_csv(filepath)

def compute_basic_stats(df: pd.DataFrame) -> dict:
    """Compute basic statistics."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "dtypes": df.dtypes.to_dict(),
    }

def compute_quality_flags(df: pd.DataFrame, **kwargs) -> dict:
    """Compute data quality flags."""
    # Базовые проверки
    quality = {
        "has_missing": df.isnull().any().any(),
        "missing_count": df.isnull().sum().sum(),
        "missing_ratio": df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
        "has_duplicates": df.duplicated().any(),
        "duplicate_count": df.duplicated().sum(),
        "duplicate_ratio": df.duplicated().sum() / df.shape[0] if df.shape[0] > 0 else 0,
    }
    
    # HW03: Новые эвристики
    
    # 1. Константные колонки (все значения одинаковые)
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() == 1:
            constant_cols.append(col)
    quality["has_constant_columns"] = len(constant_cols) > 0
    quality["constant_columns"] = constant_cols
    
    # 2. Высококардинальные категориальные признаки
    high_card_threshold = kwargs.get('high_cardinality_threshold', 50)
    high_card_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        unique_count = df[col].nunique()
        if unique_count > high_card_threshold:
            high_card_cols.append((col, unique_count))
    quality["has_high_cardinality"] = len(high_card_cols) > 0
    quality["high_cardinality_cols"] = [col for col, _ in high_card_cols]
    quality["high_cardinality_details"] = high_card_cols
    quality["high_cardinality_threshold"] = high_card_threshold
    
    # 3. Дубликаты в ID колонках
    id_cols = [col for col in df.columns if 'id' in col.lower() or 'ID' in col]
    duplicate_id_info = {}
    for col in id_cols:
        duplicate_count = df[col].duplicated().sum()
        if duplicate_count > 0:
            duplicate_id_info[col] = duplicate_count
    quality["has_id_duplicates"] = len(duplicate_id_info) > 0
    quality["duplicate_id_info"] = duplicate_id_info
    
    # 4. Много нулевых значений в числовых колонках
    zero_threshold = kwargs.get('zero_threshold', 0.3)
    zero_cols = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        zero_ratio = (df[col] == 0).sum() / len(df)
        if zero_ratio > zero_threshold:
            zero_cols.append((col, zero_ratio))
    quality["has_many_zeros"] = len(zero_cols) > 0
    quality["zero_columns"] = zero_cols
    quality["zero_threshold"] = zero_threshold
    
    # 5. Интегральный показатель качества (опционально)
    quality_score = 100
    
    # Штрафы за проблемы
    if quality["has_missing"]:
        quality_score -= min(quality["missing_ratio"] * 100, 30)
    
    if quality["has_duplicates"]:
        quality_score -= min(quality["duplicate_ratio"] * 50, 20)
    
    if quality["has_constant_columns"]:
        quality_score -= len(quality["constant_columns"]) * 10
    
    if quality["has_high_cardinality"]:
        quality_score -= len(quality["high_cardinality_cols"]) * 5
    
    if quality["has_id_duplicates"]:
        quality_score -= 15
    
    if quality["has_many_zeros"]:
        for col, ratio in quality["zero_columns"]:
            quality_score -= min(ratio * 20, 10)
    
    quality["quality_score"] = max(0, min(100, round(quality_score, 1)))
    
    return quality
