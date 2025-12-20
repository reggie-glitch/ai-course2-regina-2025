"""Core functions for EDA CLI."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV file."""
    return pd.read_csv(filepath)


def compute_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute basic statistics."""
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


def ensure_python_types(obj):
    """Convert numpy types to standard Python types."""
    if isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [ensure_python_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: ensure_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_python_types(item) for item in obj]
    else:
        return obj


def compute_quality_flags(
    df: pd.DataFrame,
    high_cardinality_threshold: int = 50,
    zero_threshold: float = 0.3,
    min_missing_share: float = 0.1,
    # ============= НОВЫЕ ПАРАМЕТРЫ =============
    outlier_threshold: float = 1.5,      # для выбросов (IQR multiplier)
    imbalance_threshold: float = 0.9     # для несбалансированных категорий
) -> Dict[str, Any]:
    """Compute quality flags with heuristic checks."""
    
    # Initialize results with Python types
    results = {
        "quality_score": 100,
        "has_constant_columns": False,
        "has_high_cardinality": False,
        "has_id_duplicates": False,
        "has_many_zeros": False,
        # ============= НОВЫЕ ФЛАГИ =============
        "has_outliers": False,
        "has_imbalanced_categories": False,
        
        "constant_columns": [],
        "high_cardinality_cols": [],
        "high_cardinality_details": [],
        "duplicate_id_info": {},
        "zero_columns": [],
        # ============= НОВАЯ ИНФОРМАЦИЯ =============
        "outlier_columns": [],
        "imbalanced_columns": [],
        "columns_with_missing": [],
        
        "missing_count": 0,
        "duplicate_count": 0,
    }
    
    # 1. Check for constant columns
    for col in df.columns:
        if int(df[col].nunique()) == 1:
            results["has_constant_columns"] = True
            results["constant_columns"].append(col)
            results["quality_score"] -= 10
    
    # 2. Check for high cardinality categorical columns
    categorical_dtypes = ['object', 'category', 'string']
    categorical_cols = df.select_dtypes(include=categorical_dtypes).columns
    
    for col in categorical_cols:
        unique_count = int(df[col].nunique())
        if unique_count > high_cardinality_threshold:
            results["has_high_cardinality"] = True
            results["high_cardinality_cols"].append(col)
            results["high_cardinality_details"].append({
                "column": col,
                "unique_count": unique_count,
                "threshold": int(high_cardinality_threshold)
            })
            results["quality_score"] -= 5
    
    # 3. Check for ID duplicates
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    for col in id_cols:
        duplicates = df[col].duplicated()
        if bool(duplicates.any()):
            results["has_id_duplicates"] = True
            results["duplicate_id_info"][col] = {
                "duplicate_count": int(duplicates.sum()),
                "total": int(len(df[col]))
            }
            results["quality_score"] -= 15
    
    # 4. Check for many zero values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        zero_count = int((df[col] == 0).sum())
        zero_ratio = float(zero_count / len(df)) if len(df) > 0 else 0.0
        if zero_ratio > zero_threshold:
            results["has_many_zeros"] = True
            results["zero_columns"].append({
                "column": col,
                "zero_ratio": float(zero_ratio),
                "threshold": float(zero_threshold)
            })
            results["quality_score"] -= 8
    
    # 5. Check for missing values
    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        missing_ratio = float(missing_count / len(df)) if len(df) > 0 else 0.0
        if missing_ratio > min_missing_share:
            results["columns_with_missing"].append({
                "column": col,
                "missing_ratio": float(missing_ratio),
                "threshold": float(min_missing_share)
            })
    
    # ============================================
    # НОВАЯ ЭВРИСТИКА 1: Обнаружение выбросов (IQR метод)
    # ============================================
    for col in numeric_cols:
        if len(df[col].dropna()) < 10:  # нужно минимум 10 значений
            continue
            
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        
        if iqr > 0:  # избегаем деления на ноль
            lower_bound = q1 - outlier_threshold * iqr
            upper_bound = q3 + outlier_threshold * iqr
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
            outliers_count = outliers.sum()
            outliers_ratio = outliers_count / len(df[col])
            
            if outliers_ratio > 0.05:  # если больше 5% выбросов
                results["has_outliers"] = True
                results["outlier_columns"].append({
                    "column": col,
                    "outliers_count": int(outliers_count),
                    "outliers_ratio": float(outliers_ratio),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound)
                })
                results["quality_score"] -= 7
    
    # ============================================
    # НОВАЯ ЭВРИСТИКА 2: Несбалансированные категории
    # ============================================
    for col in categorical_cols:
        value_counts = df[col].value_counts(normalize=True)
        if len(value_counts) > 1:  # не проверяем константные колонки
            dominant_ratio = value_counts.iloc[0]  # доля самой частой категории
            
            if dominant_ratio > imbalance_threshold:
                results["has_imbalanced_categories"] = True
                results["imbalanced_columns"].append({
                    "column": col,
                    "dominant_category": value_counts.index[0],
                    "dominant_ratio": float(dominant_ratio),
                    "threshold": float(imbalance_threshold),
                    "unique_categories": int(len(value_counts))
                })
                results["quality_score"] -= 6
    
    # Calculate counts
    results["missing_count"] = int(df.isnull().sum().sum())
    results["duplicate_count"] = int(df.duplicated().sum())
    
    # Ensure boolean values are Python bool
    results["has_constant_columns"] = bool(results["has_constant_columns"])
    results["has_high_cardinality"] = bool(results["has_high_cardinality"])
    results["has_id_duplicates"] = bool(results["has_id_duplicates"])
    results["has_many_zeros"] = bool(results["has_many_zeros"])
    results["has_outliers"] = bool(results["has_outliers"])
    results["has_imbalanced_categories"] = bool(results["has_imbalanced_categories"])
    
    # Limit quality score
    results["quality_score"] = int(max(0, min(100, results["quality_score"])))
    
    # Convert all numpy types to Python types
    return ensure_python_types(results)
