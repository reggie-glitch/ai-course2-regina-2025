"""
Скрипт загрузки датасета NSL-KDD.
Запускать из папки project/:  python data/download_data.py
"""

import os
import urllib.request
import yaml


def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def download(url, dest):
    if os.path.exists(dest):
        print(f"  Уже есть: {dest} — пропускаем.")
        return
    print(f"  Скачиваем {url} ...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  Сохранено: {dest}")


def main():
    cfg = load_config()
    print("=== Загрузка NSL-KDD ===")
    download(cfg["data"]["train_url"], cfg["data"]["train_path"])
    download(cfg["data"]["test_url"],  cfg["data"]["test_path"])
    print("=== Готово! ===")


if __name__ == "__main__":
    main()
