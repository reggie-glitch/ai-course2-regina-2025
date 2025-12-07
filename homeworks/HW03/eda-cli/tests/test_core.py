import pandas as pd
import numpy as np
from eda_cli.core import load_data, compute_quality_flags

def test_load_data(tmp_path):
    """Test loading data."""
    test_file = tmp_path / "test.csv"
    test_file.write_text("a,b,c\n1,2,3\n4,5,6")
    
    df = load_data(str(test_file))
    assert df.shape == (2, 3)

def test_quality_flags_constant():
    """Test constant columns detection."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'constant': [5, 5, 5],  # константная колонка
        'value': [10, 20, 30]
    })
    
    quality = compute_quality_flags(df)
    assert quality['has_constant_columns'] == True
    assert 'constant' in quality['constant_columns']
    assert len(quality['constant_columns']) == 1

def test_quality_flags_high_cardinality():
    """Test high cardinality detection."""
    df = pd.DataFrame({
        'id': range(100),
        'category': [f'cat_{i}' for i in range(100)]  # 100 уникальных значений
    })
    
    # С порогом 50 должен обнаружить
    quality = compute_quality_flags(df, high_cardinality_threshold=50)
    assert quality['has_high_cardinality'] == True
    assert 'category' in quality['high_cardinality_cols']
    
    # С порогом 150 НЕ должен обнаружить
    quality = compute_quality_flags(df, high_cardinality_threshold=150)
    assert quality['has_high_cardinality'] == False

def test_quality_flags_id_duplicates():
    """Test ID duplicates detection."""
    df = pd.DataFrame({
        'user_id': [1, 2, 1, 4, 5],  # дубликат ID 1
        'item_id': [100, 200, 300, 100, 500],  # дубликат ID 100
        'value': [10, 20, 30, 40, 50]
    })
    
    quality = compute_quality_flags(df)
    assert quality['has_id_duplicates'] == True
    assert 'user_id' in quality['duplicate_id_info']
    assert 'item_id' in quality['duplicate_id_info']
    assert quality['duplicate_id_info']['user_id'] == 1
    assert quality['duplicate_id_info']['item_id'] == 1

def test_quality_flags_zero_values():
    """Test zero values detection."""
    df = pd.DataFrame({
        'id': range(10),
        'zeros_high': [0, 0, 0, 0, 0, 0, 0, 1, 2, 3],  # 70% нулей
        'zeros_low': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # 10% нулей
        'text': ['a'] * 10
    })
    
    # С порогом 0.5 (50%) - должна обнаружить zeros_high
    quality = compute_quality_flags(df, zero_threshold=0.5)
    assert quality['has_many_zeros'] == True
    assert len(quality['zero_columns']) == 1
    assert quality['zero_columns'][0][0] == 'zeros_high'
    assert quality['zero_columns'][0][1] == 0.7
    
    # С порогом 0.8 (80%) - не должна обнаружить
    quality = compute_quality_flags(df, zero_threshold=0.8)
    assert quality['has_many_zeros'] == False

def test_quality_score():
    """Test quality score calculation."""
    df = pd.DataFrame({
        'id': [1, 2, 3, 1],  # дубликат ID
        'constant': [5, 5, 5, 5],  # константная колонка
        'value': [10, 20, 30, 40]
    })
    
    quality = compute_quality_flags(df)
    assert 'quality_score' in quality
    assert 0 <= quality['quality_score'] <= 100
    # При проблемах score должен быть < 100
    assert quality['quality_score'] < 100
