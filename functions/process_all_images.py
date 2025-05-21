import os
import matplotlib.pyplot as plt
from PIL import Image
from jpeg_compressor import compress_to_file, decompress_from_file


def get_project_root():
    """Возвращает абсолютный путь к корню проекта"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)  # Поднимаемся на уровень выше (из functions в jpeg_compressor)


# Определяем пути относительно корня проекта
PROJECT_ROOT = get_project_root()
INPUT_FOLDER = os.path.join(PROJECT_ROOT, 'images', 'input_images')
CONVERTED_FOLDER = os.path.join(PROJECT_ROOT, 'images', 'converted')  # Новая папка с изображениями
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'images', 'output_images')
COMPRESSED_FOLDER = os.path.join(OUTPUT_FOLDER, 'compressed')
RESTORED_FOLDER = os.path.join(OUTPUT_FOLDER, 'restored')
GRAPHS_FOLDER = os.path.join(OUTPUT_FOLDER, 'graphs')

# Создаем все необходимые папки
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)
os.makedirs(RESTORED_FOLDER, exist_ok=True)
os.makedirs(GRAPHS_FOLDER, exist_ok=True)

# Основные уровни качества для восстановления изображений
RESTORE_QUALITIES = [1, 20, 40, 60, 80, 100]
# Все уровни качества для графика (включая дополнительные точки)
GRAPH_QUALITIES = sorted(set(RESTORE_QUALITIES + list(range(0, 101, 5))))


def process_image(image_path, quality, restore=False):
    """Обрабатывает изображение с заданным качеством"""
    try:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        compressed_path = os.path.join(COMPRESSED_FOLDER, f"{base_name}_q{quality}.bin")

        # Сжатие
        compress_to_file(image_path, compressed_path, quality=quality)
        compressed_size = os.path.getsize(compressed_path)

        result = {
            'quality': quality,
            'compressed_size': compressed_size,
            'compression_ratio': os.path.getsize(image_path) / compressed_size if compressed_size > 0 else 0
        }

        # Восстановление только если требуется
        if restore:
            output_path = os.path.join(RESTORED_FOLDER, f"{base_name}_q{quality}.png")
            decompress_from_file(compressed_path, output_path)

            # Конвертируем в PNG
            img = Image.open(output_path)
            if img.format != 'PNG':
                img.save(output_path, format='PNG')

            result['restored_size'] = os.path.getsize(output_path)

        return result
    except Exception as e:
        print(f"Ошибка обработки изображения {image_path} с качеством {quality}: {str(e)}")
        return {
            'quality': quality,
            'compressed_size': 0,
            'compression_ratio': 0
        }

def create_compression_graph(image_name, results):
    """Создает и сохраняет график сжатия"""
    try:
        qualities = [r['quality'] for r in results if r['compressed_size'] > 0]
        compressed_sizes = [r['compressed_size'] / 1024 for r in results if r['compressed_size'] > 0]  # в KB

        if not qualities:
            raise ValueError("Нет данных для построения графика")

        plt.figure(figsize=(12, 6))

        # Основной график
        plt.plot(qualities, compressed_sizes, 'b-', linewidth=2)

        # Точки восстановления выделяем красным
        restore_points = [(q, s) for q, s in zip(qualities, compressed_sizes)
                         if q in RESTORE_QUALITIES]
        if restore_points:
            restore_q, restore_s = zip(*restore_points)
            plt.scatter(restore_q, restore_s, c='red', s=100,
                       label='Точки восстановления', zorder=5)

        # Используем оригинальное имя изображения для заголовка
        image_title = os.path.splitext(image_name)[0]
        plt.title(f'Зависимость размера сжатого файла от качества\n{image_title}')
        plt.xlabel('Качество сжатия')
        plt.ylabel('Размер сжатого файла (KB)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Сохраняем график с оригинальным именем изображения
        graph_path = os.path.join(GRAPHS_FOLDER, f"{image_title}_compression.png")
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()

        return graph_path
    except Exception as e:
        print(f"Ошибка построения графика: {str(e)}")
        return None

def process_folder(folder_path, restore_results, graph_results, is_converted=False):
    """Обрабатывает все изображения в указанной папке"""
    if not os.path.exists(folder_path):
        print(f"Предупреждение: папка не существует: {folder_path}")
        return

    folder_name = "converted" if is_converted else "input_images"
    print(f"\n=== Обработка папки {folder_name} ===")

    for image_name in os.listdir(folder_path):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        print(f"\nОбработка {image_name} из {folder_name}:")
        input_path = os.path.join(folder_path, image_name)

        # 1. Обработка для восстановления
        print("Этап 1: Восстановление изображений")
        image_results = []
        for quality in RESTORE_QUALITIES:
            res = process_image(input_path, quality, restore=True)
            image_results.append(res)
            if 'restored_size' in res:
                print(f"q{quality:3d}: сжато {res['compressed_size'] / 1024:.1f} KB, "
                      f"восстановлено {res['restored_size'] / 1024:.1f} KB")
            else:
                print(f"q{quality:3d}: ошибка восстановления")

        restore_results[image_name] = image_results

        # 2. Дополнительная обработка для графиков
        print("Этап 2: Подготовка данных для графиков")
        existing_qualities = {r['quality'] for r in image_results}
        full_image_results = image_results.copy()

        for quality in GRAPH_QUALITIES:
            if quality in existing_qualities:
                continue

            res = process_image(input_path, quality, restore=False)
            full_image_results.append(res)
            print(f"q{quality:3d}: сжато {res['compressed_size'] / 1024:.1f} KB")

        graph_results[image_name] = sorted(full_image_results, key=lambda x: x['quality'])


def main():
    restore_results = {}
    graph_results = {}

    # Обрабатываем обе папки
    process_folder(INPUT_FOLDER, restore_results, graph_results, is_converted=False)
    process_folder(CONVERTED_FOLDER, restore_results, graph_results, is_converted=True)

    # 3. Строим графики для всех изображений
    print("\n=== Этап 3: Построение графиков ===")
    for image_name, results in graph_results.items():
        graph_path = create_compression_graph(image_name, results)
        if graph_path:
            print(f"График для {image_name} сохранен: {graph_path}")

    print("\n=== Обработка завершена ===")
    print(f"\nРезультаты сохранены в:")
    print(f"- Сжатые файлы: {COMPRESSED_FOLDER}")
    print(f"- Восстановленные PNG: {RESTORED_FOLDER}")
    print(f"- Графики: {GRAPHS_FOLDER}")


if __name__ == "__main__":
    main()