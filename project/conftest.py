"""
conftest.py — автоматически добавляет src/ в путь для pytest.
Благодаря этому тесты можно запускать из любой папки.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
