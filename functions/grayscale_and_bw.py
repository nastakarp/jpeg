from PIL import Image
import os

# Получаем абсолютный путь к директории скрипта
script_dir = os.path.dirname(os.path.abspath(__file__))

# Пути к изображениям (относительно директории скрипта)
images = {
    "Lenna": os.path.join(script_dir, "../images/input_images/Lenna.png"),
    "Flower": os.path.join(script_dir, "../images/input_images/Flower.jpg")
}

# Папка для сохранения результатов (создаётся автоматически)
output_folder = os.path.join(script_dir, "../images/converted/")
os.makedirs(output_folder, exist_ok=True)

def convert_and_save(name, path):
    try:
        img = Image.open(path)

        # 1. Оттенки серого
        gray = img.convert("L")
        gray.save(os.path.join(output_folder, f"{name}_gray.png"))

        # 2. Чёрно-белое без дизеринга (порог 128)
        bw_no_dither = gray.point(lambda x: 255 if x > 128 else 0, mode='1')
        bw_no_dither.save(os.path.join(output_folder, f"{name}_bw_nodither.png"))

        # 3. Чёрно-белое с дизерингом (Floyd–Steinberg)
        bw_dither = gray.convert("1")  # по умолчанию используется дизеринг
        bw_dither.save(os.path.join(output_folder, f"{name}_bw_dither.png"))

        print(f"{name} обработано и сохранено.")
    except FileNotFoundError:
        print(f"Ошибка: файл {path} не найден!")
    except Exception as e:
        print(f"Ошибка при обработке {name}: {e}")

if __name__ == "__main__":
    for name, path in images.items():
        convert_and_save(name, path)