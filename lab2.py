import os
from datetime import datetime

import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError

ROOT = "Lab2_Media"
INPUT_DIR = os.path.join(ROOT, "data", "input")
OUTPUT_PATH = os.path.join(ROOT, "data", "output", "result.jpg")
LOG_PATH = os.path.join(ROOT, "logs", "history.txt")

FILTER_NAME = "Box-фильтр + Серый мир"


def box_filter(img, radius=2):
    return img.filter(ImageFilter.BoxBlur(radius))


def gray_world(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    means = arr.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    arr = arr * (gray / means)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def write_log(filename, status):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] Файл: {filename} | Фильтр: {FILTER_NAME} | Статус: {status}\n"
    try:
        with open(LOG_PATH, "a", encoding="UTF-8") as log:
            log.write(line)
    except OSError:
        print("Предупреждение: не удалось записать лог.")


filename = input("Какой файл обработать (например, image1.jpg или image2.png)?\n").strip()
filename = os.path.basename(filename)
path = os.path.join(INPUT_DIR, filename)

try:
    with open(path, "rb") as f:
        img = Image.open(f)
        img.load()

    result = gray_world(box_filter(img.convert("RGB")))

    result.save(OUTPUT_PATH, "JPEG")

except FileNotFoundError:
    print("Ошибка: Файл не найден")
    write_log(filename, "Файл не найден")
except PermissionError:
    print(f"Ошибка: Нет прав на чтение файла {filename}")
    write_log(filename, "Ошибка доступа")
except (UnidentifiedImageError, OSError):
    print("Ошибка: Не удалось открыть файл как изображение")
    write_log(filename, "Ошибка чтения")
else:
    print(f"Готово! Результат сохранён: {OUTPUT_PATH}")
    write_log(filename, "Успешно")
