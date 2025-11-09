import time
import os
import re
import cv2
import numpy as np
import easyocr
import requests
import colorama
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from difflib import SequenceMatcher
from typing import List, Optional, Dict

# --- Константы и конфигурация ---
# Директория для отслеживания новых скриншотов. "." означает текущую директорию.
IMAGE_DIR = "."
# Минимальный порог схожести (от 0.0 до 1.0) для поиска названий предметов.
MIN_SIMILARITY_SCORE = 0.8
# Инициализация colorama для цветного вывода в консоли.
colorama.init()

# --- Функции обработки изображений ---

def process_image_hsv(image: np.ndarray) -> np.ndarray:
    """
    Применяет серию фильтров к изображению для выделения текста.

    Изображение переводится в цветовое пространство HSV для фильтрации по цвету,
    затем опционально преобразуется в оттенки серого, проходит через фильтр
    резкости и бинаризацию для улучшения читаемости текста.

    Args:
        image: Исходное изображение в формате OpenCV (BGR).

    Returns:
        Обработанное изображение, готовое для распознавания текста.
    """
    # Преобразование в HSV и фильтрация по жёлтому цвету, характерному для наград
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([20, 80, 80])
    upper_color = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Убираем шумы
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Применяем маску, чтобы оставить только нужные области
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Преобразуем в оттенки серого для лучшей работы OCR
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # Применяем фильтр для повышения резкости
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_img = cv2.filter2D(gray_result, -1, sharpen_kernel)
    
    # Применяем пороговую бинаризацию для получения чёткого чёрно-белого изображения
    _, threshold_img = cv2.threshold(sharpened_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Возвращаем в формат RGB для совместимости с EasyOCR
    return cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB)

def upscale_image(image: np.ndarray, scale_factor: int = 3) -> np.ndarray:
    """
    Увеличивает разрешение изображения для улучшения качества распознавания.

    Args:
        image: Исходное изображение.
        scale_factor: Коэффициент масштабирования.

    Returns:
        Масштабированное изображение.
    """
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)

def extract_text_from_image(image_path: str, reader: easyocr.Reader) -> Optional[List[str]]:
    """
    Загружает изображение, обрабатывает его и извлекает текст с наград.

    Args:
        image_path: Путь к файлу изображения.
        reader: Экземпляр easyocr.Reader для распознавания текста.

    Returns:
        Список распознанных названий предметов или None в случае ошибки.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Ошибка: Не удалось прочитать изображение: {image_path}")
            return None

        upscaled_img = upscale_image(img)
        height, width, _ = upscaled_img.shape
        
        extracted_names = []
        # Изображение экрана наград делится на 4 вертикальные части
        for i in range(4):
            # Вырезаем одну из четырёх колонок с наградами
            cropped_img = upscaled_img[:, i * width // 4:(i + 1) * width // 4]
            processed_img = process_image_hsv(cropped_img)
            
            # Распознаём текст с помощью EasyOCR
            results = reader.readtext(processed_img, text_threshold=0.9)
            
            raw_text_parts = [text.strip().replace("`", "").replace("'", "") for _, text, _ in results]
            cleaned_text = " ".join(raw_text_parts)

            # Стандартизация названий чертежей
            if cleaned_text.lower().startswith("чертёж"):
                cleaned_text = cleaned_text.replace("Чертёж", "Чертеж", 1).replace("чертёж", "чертеж", 1)
            
            if cleaned_text.startswith("Чертеж"):
                item_name_part = cleaned_text[len("Чертеж"):].lstrip(' :').strip()
                cleaned_text = f"{item_name_part} (Чертеж)"

            if cleaned_text:
                extracted_names.append(cleaned_text)
                
        return extracted_names
        
    except Exception as e:
        print(f"Критическая ошибка при обработке изображения: {e}")
        return None

# --- Функции для работы с API Warframe.Market ---

def find_best_match(query: str, choices: List[str], min_score: float = MIN_SIMILARITY_SCORE) -> Optional[str]:
    """
    Находит наиболее похожее название предмета в списке с помощью нечёткого поиска.

    Args:
        query: Распознанное название предмета.
        choices: Список всех официальных названий предметов из API.
        min_score: Минимальный порог схожеosti для совпадения.

    Returns:
        Наиболее подходящее название или None, если схожесть ниже порога.
    """
    best_match = None
    max_score = 0
    
    for choice in choices:
        score = SequenceMatcher(None, query, choice).ratio()
        if score > max_score:
            max_score = score
            best_match = choice
    
    if max_score >= min_score:
        return best_match
        
    return None

def build_items_dictionary() -> Optional[Dict[str, str]]:
    """
    Загружает с Warframe.Market API список всех предметов и создаёт словарь.

    Returns:
        Словарь, где ключ - название предмета на русском, значение - его 'slug' для API.
        Возвращает None в случае ошибки сети.
    """
    try:
        api_url = "https://api.warframe.market/v2/items"
        headers = {
            'Language': 'ru',
            'accept': 'application/json',
        }
        response = requests.get(api_url, headers=headers, timeout=15)
        response.raise_for_status() # Проверка на HTTP ошибки
        
        items_data = response.json().get("data", [])
        items_dict = {
            item["i18n"]["ru"]["name"]: item["slug"]
            for item in items_data
            if "ru" in item.get("i18n", {}) and "name" in item["i18n"]["ru"]
        }
        
        print(f"Словарь предметов успешно создан. Загружено {len(items_dict)} предметов.")
        return items_dict
        
    except requests.exceptions.RequestException as e:
        print(f"ОШИБКА: Не удалось подключиться к API Warframe.Market: {e}")
        return None

def fetch_price(item_name: str, items_slug_dict: Dict[str, str]):
    """
    Получает и выводит в консоль топ-5 цен для указанного предмета.

    Использует нечёткий поиск для нахождения корректного 'slug' предмета,
    а затем делает запрос к API для получения цен.
    
    Args:
        item_name: Название предмета, распознанное с изображения.
        items_slug_dict: Словарь всех предметов и их 'slug'.
    """
    # Формы не продаются на маркете, пропускаем их
    if "форма" in item_name.lower():
        print(f"{colorama.Fore.CYAN}{item_name}{colorama.Style.RESET_ALL} {colorama.Fore.MAGENTA}- Не продается{colorama.Style.RESET_ALL}")
        return

    matched_item_name = find_best_match(item_name, list(items_slug_dict.keys()))
    
    if not matched_item_name:
        print(f"{colorama.Fore.CYAN}{item_name}{colorama.Style.RESET_ALL} {colorama.Fore.RED}- Не найдено (схожесть < {int(MIN_SIMILARITY_SCORE*100)}%){colorama.Style.RESET_ALL}")
        return

    item_slug = items_slug_dict.get(matched_item_name)
    
    # Для вывода используем исходное имя, если оно отличается от найденного в словаре
    display_name = item_name
    if item_name != matched_item_name:
        display_name = f"{item_name} -> [{matched_item_name}]"

    try:
        api_url = f"https://api.warframe.market/v2/orders/item/{item_slug}/top"
        headers = {'Platform': 'pc', 'accept': 'application/json'}
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        response_data = response.json()
        sell_orders = response_data.get("data", {}).get("sell", [])
        
        if sell_orders:
            # Берём до 5 самых дешёвых ордеров
            prices = [order["platinum"] for order in sell_orders[:5]]
            prices_str = ", ".join(map(str, prices))
            print(f"{colorama.Fore.CYAN}{display_name}{colorama.Style.RESET_ALL} {colorama.Fore.YELLOW}{prices_str} платины{colorama.Style.RESET_ALL}")
        else:
            print(f"{colorama.Fore.CYAN}{display_name}{colorama.Style.RESET_ALL} {colorama.Fore.YELLOW}Онлайн-продавцы не найдены.{colorama.Style.RESET_ALL}")
            
    except requests.exceptions.RequestException as e:
         print(f"{colorama.Fore.CYAN}{display_name}{colorama.Style.RESET_ALL} {colorama.Fore.RED}- Ошибка сети при запросе цены: {e}{colorama.Style.RESET_ALL}")
    except (KeyError, IndexError):
         print(f"{colorama.Fore.CYAN}{display_name}{colorama.Style.RESET_ALL} {colorama.Fore.RED}- Ошибка при обработке ответа от API.{colorama.Style.RESET_ALL}")

# --- Класс для отслеживания файловой системы ---

class ScreenshotHandler(FileSystemEventHandler):
    """
    Обработчик событий файловой системы, реагирующий на создание новых изображений.
    """
    def __init__(self, reader: easyocr.Reader, items_dict: Dict[str, str]):
        self.reader = reader
        self.items_dict = items_dict

    def on_created(self, event):
        """
        Вызывается при создании нового файла в отслеживаемой директории.
        """
        if event.is_directory or not event.src_path.lower().endswith((".png", ".jpg", ".jpeg")):
            return
        
        # Задержка, чтобы файл успел полностью записаться на диск
        time.sleep(0.5)
        
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Обнаружен новый скриншот: {os.path.basename(event.src_path)}")
        
        if not self.items_dict:
            print("Словарь предметов пуст. Проверка цен невозможна.")
            return
            
        names = extract_text_from_image(event.src_path, self.reader)
        if names:
            print("-" * 20)
            for name in names:
                fetch_price(name, self.items_dict)
            print("-" * 20)

# --- Основной блок исполнения ---

def main():
    """
    Главная функция: инициализирует OCR, загружает словарь предметов
    и запускает наблюдатель за файловой системой.
    """
    print("Инициализация EasyOCR (может занять некоторое время)...")
    # Используем CPU, т.к. GPU не всегда доступен. Замените gpu=False на gpu=True, если у вас настроен CUDA.
    try:
        reader = easyocr.Reader(['ru'], gpu=False)
    except Exception as e:
        print(f"Критическая ошибка при инициализации EasyOCR: {e}")
        return

    print("Загрузка словаря предметов с Warframe.Market API...")
    master_items_dict = build_items_dictionary()
    
    if master_items_dict is None:
        print("Не удалось создать словарь предметов. Проверьте подключение к интернету. Выход.")
        return
        
    if not master_items_dict:
        print(f"{colorama.Fore.YELLOW}Предупреждение: Словарь предметов пуст. Цены могут не определяться.{colorama.Style.RESET_ALL}")

    # Запуск наблюдателя
    event_handler = ScreenshotHandler(reader, master_items_dict)
    observer = Observer()
    observer.schedule(event_handler, IMAGE_DIR, recursive=False)
    observer.start()
    
    print(f"\n{colorama.Fore.GREEN}Наблюдатель запущен в директории: '{os.path.abspath(IMAGE_DIR)}'{colorama.Style.RESET_ALL}")
    print("Сохраняйте скриншоты экрана наград в эту папку.")
    print("Для остановки нажмите Ctrl+C")
    
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
        print("\nНаблюдатель остановлен.")
        
    observer.join()

if __name__ == "__main__":
    main()
