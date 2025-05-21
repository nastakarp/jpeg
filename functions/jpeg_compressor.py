import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import struct
import json
from functions.core.rgb_to_ycbcr import rgb_to_ycbcr, ycbcr_to_rgb
from functions.core.downsample import downsample_channel
from functions.core.block_processing import split_into_blocks
from functions.core.dct import dct2, idct2
from functions.core.quantization import get_quantization_matrix, quantize_blocks, dequantize_blocks
from functions.core.zigzag import zigzag_scan, inverse_zigzag_scan
from functions.core.dc_encoding import encode_dc_coefficients, decode_dc_coefficients
from functions.core.huffman_ac_codec import encode_ac_coefficients, decode_ac_coefficients

# Получаем абсолютный путь к директории текущего скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # os.path.abspath возвращает абсолютный путь к файлу
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Получаем родительскую директорию


def get_absolute_path(*relative_path_parts):
    """Собирает абсолютный путь из относительных частей"""
    # os.path.join объединяет пути с учетом ОС
    return os.path.join(BASE_DIR, *relative_path_parts)


def load_image(path):
    """Загружает изображение и конвертирует в RGB numpy array"""
    print("1. Загрузка изображения...")
    try:
        # Открываем изображение и конвертируем в RGB (на случай если оно в другом формате)
        img = Image.open(path).convert("RGB")
        # Преобразуем PIL Image в numpy array
        return np.array(img)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл изображения не найден: {path}")
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки изображения: {str(e)}")


def upsample_channel(channel: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Увеличивает разрешение канала до target_shape с использованием билинейной интерполяции"""
    print("  Увеличение разрешения канала...")
    from PIL import Image
    # Создаем PIL Image из numpy массива
    img = Image.fromarray(channel)
    # Масштабируем с использованием билинейной интерполяции
    return np.array(img.resize((target_shape[1], target_shape[0]), Image.BILINEAR))


def process_channel(channel: np.ndarray, quant_matrix=None, block_size: int = 8, is_luma: bool = True) -> tuple:
    """Полный процесс обработки канала: DCT + квантование + кодирование"""
    # Разбиваем канал на блоки указанного размера (по умолчанию 8x8)
    blocks = split_into_blocks(channel, block_size)
    # Создаем массив для хранения DCT коэффициентов такого же размера как блоки
    dct_blocks = np.zeros_like(blocks, dtype=np.float32)

    # Применяем DCT к каждому блоку
    for i in range(blocks.shape[0]):  # По вертикали
        for j in range(blocks.shape[1]):  # По горизонтали
            dct_blocks[i, j] = dct2(blocks[i, j])  # 2D DCT преобразование

    if quant_matrix is not None:
        # Квантуем коэффициенты DCT с использованием матрицы квантования
        quantized_blocks = quantize_blocks(dct_blocks, quant_matrix)

        # Извлекаем DC коэффициенты (левый верхний угол каждого блока)
        dc_coeffs = []
        for i in range(quantized_blocks.shape[0]):
            for j in range(quantized_blocks.shape[1]):
                # Округляем и преобразуем в целое число
                dc_value = int(round(float(quantized_blocks[i, j, 0, 0])))
                dc_coeffs.append(dc_value)

        # Кодируем DC коэффициенты (дельта-кодирование + Хаффман)
        dc_bitstream = encode_dc_coefficients(np.array(dc_coeffs), is_luma)

        # Обрабатываем AC коэффициенты
        ac_bitstreams = []
        for i in range(quantized_blocks.shape[0]):
            for j in range(quantized_blocks.shape[1]):
                # Копируем блок чтобы не изменять оригинал
                block = quantized_blocks[i, j].copy()
                block[0, 0] = 0  # Обнуляем DC коэффициент
                # Применяем зигзаг-сканирование (исключая первый элемент - DC)
                zigzag = zigzag_scan(block)[1:]
                # Округляем коэффициенты до целых чисел
                zigzag = [int(round(float(x))) for x in zigzag]
                # Кодируем AC коэффициенты (RLE + Хаффман)
                ac_bitstream = encode_ac_coefficients(zigzag, is_luma)
                ac_bitstreams.append(ac_bitstream)

        return dc_bitstream, ac_bitstreams  # Возвращаем битовые потоки

    return None, None  # Если матрица квантования не указана


def inverse_process_channel(dc_bitstream: str, ac_bitstreams: list,
                            quant_matrix=None, original_shape: tuple = None,
                            block_size: int = 8, is_luma: bool = True) -> np.ndarray:
    """Обратный процесс: декодирование + обратное DCT + восстановление изображения"""
    try:
        # Проверяем корректность параметров
        if not original_shape or len(original_shape) != 2:
            raise ValueError("Неверный формат original_shape")

        h, w = original_shape  # Высота и ширина исходного изображения
        if h <= 0 or w <= 0:
            raise ValueError("Неверные размеры изображения")

        # Вычисляем количество блоков по вертикали и горизонтали
        h_blocks = (h + block_size - 1) // block_size
        w_blocks = (w + block_size - 1) // block_size

        # Создаем пустое изображение для восстановления (с учетом padding если нужно)
        restored = np.zeros((h_blocks * block_size, w_blocks * block_size), dtype=np.float32)

        # Декодируем DC коэффициенты
        dc_coeffs = decode_dc_coefficients(dc_bitstream, h_blocks * w_blocks, is_luma)

        # Восстанавливаем каждый блок
        for i in range(h_blocks):
            for j in range(w_blocks):
                idx = i * w_blocks + j  # Линейный индекс блока
                try:
                    # Получаем DC коэффициент для текущего блока
                    dc_value = dc_coeffs[idx] if idx < len(dc_coeffs) else 0

                    # Декодируем AC коэффициенты
                    ac_coeffs = np.zeros(block_size * block_size - 1)
                    if idx < len(ac_bitstreams) and ac_bitstreams[idx]:
                        ac_coeffs = decode_ac_coefficients(ac_bitstreams[idx],
                                                           block_size * block_size - 1,
                                                           is_luma)

                    # Собираем блок из 1D в 2D
                    block_1d = np.insert(ac_coeffs, 0, dc_value)  # Добавляем DC коэффициент
                    block_2d = inverse_zigzag_scan(block_1d, block_size)  # Обратное зигзаг-сканирование

                    # Обратное квантование
                    dequant_block = dequantize_blocks(block_2d[np.newaxis, np.newaxis, :, :],
                                                      quant_matrix)[0, 0]

                    # Обратное DCT преобразование
                    idct_block = idct2(dequant_block)

                    # Вычисляем координаты блока в изображении
                    y_start, y_end = i * block_size, (i + 1) * block_size
                    x_start, x_end = j * block_size, (j + 1) * block_size

                    # Помещаем восстановленный блок в изображение
                    restored[y_start:y_end, x_start:x_end] = idct_block

                except Exception as e:
                    print(f"Ошибка в блоке ({i},{j}): {str(e)}")
                    # В случае ошибки заполняем блок значением DC коэффициента
                    restored[i * block_size:(i + 1) * block_size,
                    j * block_size:(j + 1) * block_size] = dc_value

        # Обрезаем до исходного размера (убираем padding если был)
        restored = restored[:h, :w]

        # Ограничиваем значения пикселей и преобразуем в uint8
        restored = np.clip(restored, 0, 255).astype(np.uint8)

        return restored

    except Exception as e:
        print(f"Критическая ошибка в inverse_process_channel: {str(e)}")
        # В случае критической ошибки возвращаем серое изображение
        return np.full(original_shape, 128, dtype=np.uint8)


def pack_data(image_size, luma_quant_matrix, chroma_quant_matrix,
              Y_dc_bits, Y_ac_bits, Cb_dc_bits, Cb_ac_bits, Cr_dc_bits, Cr_ac_bits):
    """Упаковывает все данные в байтовую строку для сохранения в файл"""

    def bits_to_bytes(bits):
        """Вспомогательная функция для преобразования битовой строки в байты"""
        padding = (8 - len(bits) % 8) % 8  # Вычисляем необходимое количество бит для дополнения
        bits += '0' * padding  # Дополняем нулями
        # Преобразуем битовую строку в байты (по 8 бит)
        return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)), padding, len(bits)

    # Упаковываем DC компоненты каждого канала
    Y_dc_bytes, Y_dc_pad, Y_dc_bitlen = bits_to_bytes(Y_dc_bits)
    Cb_dc_bytes, Cb_dc_pad, Cb_dc_bitlen = bits_to_bytes(Cb_dc_bits)
    Cr_dc_bytes, Cr_dc_pad, Cr_dc_bitlen = bits_to_bytes(Cr_dc_bits)

    # Упаковываем AC компоненты (сохраняем размер каждого блока)
    Y_ac_bytes_list = []
    Y_ac_bitlens = []
    for bits in Y_ac_bits:
        bytes_data, pad, bitlen = bits_to_bytes(bits)
        Y_ac_bytes_list.append(bytes_data)
        Y_ac_bitlens.append(bitlen)

    Cb_ac_bytes_list = []
    Cb_ac_bitlens = []
    for bits in Cb_ac_bits:
        bytes_data, pad, bitlen = bits_to_bytes(bits)
        Cb_ac_bytes_list.append(bytes_data)
        Cb_ac_bitlens.append(bitlen)

    Cr_ac_bytes_list = []
    Cr_ac_bitlens = []
    for bits in Cr_ac_bits:
        bytes_data, pad, bitlen = bits_to_bytes(bits)
        Cr_ac_bytes_list.append(bytes_data)
        Cr_ac_bitlens.append(bitlen)

    # Собираем метаданные в словарь
    meta = {
        'image_size': image_size,  # Размер изображения (высота, ширина)
        'quant_matrices': {  # Матрицы квантования
            'luma': luma_quant_matrix.tolist(),  # Для яркостной компоненты
            'chroma': chroma_quant_matrix.tolist()  # Для цветовых компонент
        },
        'dc_info': {  # Информация о DC коэффициентах
            'Y': {'padding': Y_dc_pad, 'bitlen': Y_dc_bitlen},  # Яркость
            'Cb': {'padding': Cb_dc_pad, 'bitlen': Cb_dc_bitlen},  # Цветность Cb
            'Cr': {'padding': Cr_dc_pad, 'bitlen': Cr_dc_bitlen}  # Цветность Cr
        },
        'ac_info': {  # Информация о AC коэффициентах
            'Y': {'bitlens': Y_ac_bitlens},  # Яркость
            'Cb': {'bitlens': Cb_ac_bitlens},  # Цветность Cb
            'Cr': {'bitlens': Cr_ac_bitlens}  # Цветность Cr
        }
    }

    # Сериализуем метаданные в JSON
    meta_json = json.dumps(meta).encode('utf-8')
    meta_len = len(meta_json)

    # Упаковываем все данные в бинарный формат:
    # 1. Длина метаданных (4 байта)
    # 2. Метаданные в JSON
    # 3. DC компоненты (Y, Cb, Cr)
    # 4. AC компоненты (Y, Cb, Cr)
    packed = (
            struct.pack('I', meta_len) +  # Длина метаданных (unsigned int)
            meta_json +  # Сериализованные метаданные
            Y_dc_bytes +  # DC коэффициенты яркости
            Cb_dc_bytes +  # DC коэффициенты Cb
            Cr_dc_bytes +  # DC коэффициенты Cr
            b''.join(Y_ac_bytes_list) +  # AC коэффициенты яркости
            b''.join(Cb_ac_bytes_list) +  # AC коэффициенты Cb
            b''.join(Cr_ac_bytes_list))  # AC коэффициенты Cr

    return packed


def unpack_data(packed):
    """Распаковывает данные из байтовой строки"""
    # Читаем длину метаданных (первые 4 байта)
    meta_len = struct.unpack('I', packed[:4])[0]
    # Читаем сами метаданные (JSON строка)
    meta_json = packed[4:4 + meta_len]
    # Десериализуем JSON в словарь
    meta = json.loads(meta_json.decode('utf-8'))

    # Остальные данные (после метаданных)
    data = packed[4 + meta_len:]
    pos = 0  # Текущая позиция в данных

    # Функция для чтения DC компонентов
    def read_dc_component(component):
        nonlocal pos
        info = meta['dc_info'][component]  # Получаем информацию о компоненте
        byte_len = (info['bitlen'] + 7) // 8  # Вычисляем длину в байтах
        bytes_data = data[pos:pos + byte_len]  # Читаем байты
        # Преобразуем байты в битовую строку
        bits = ''.join(f'{byte:08b}' for byte in bytes_data)
        # Удаляем padding (дополнительные биты)
        if info['padding'] > 0:
            bits = bits[:-info['padding']]
        pos += byte_len  # Сдвигаем позицию
        return bits, byte_len

    # Читаем DC компоненты для каждого канала
    Y_dc_bits, Y_dc_len = read_dc_component('Y')
    Cb_dc_bits, Cb_dc_len = read_dc_component('Cb')
    Cr_dc_bits, Cr_dc_len = read_dc_component('Cr')

    # Функция для чтения AC компонентов
    def read_ac_components(component):
        nonlocal pos
        ac_bits = []
        # Для каждого блока читаем AC коэффициенты
        for bitlen in meta['ac_info'][component]['bitlens']:
            byte_len = (bitlen + 7) // 8  # Длина в байтах
            bytes_data = data[pos:pos + byte_len]  # Читаем байты
            # Преобразуем в битовую строку
            bits = ''.join(f'{byte:08b}' for byte in bytes_data)
            padding = (8 - bitlen % 8) % 8  # Вычисляем padding
            if padding > 0:
                bits = bits[:-padding]  # Удаляем padding
            ac_bits.append(bits)
            pos += byte_len
        return ac_bits

    # Читаем AC компоненты для каждого канала
    Y_ac_bits = read_ac_components('Y')
    Cb_ac_bits = read_ac_components('Cb')
    Cr_ac_bits = read_ac_components('Cr')

    # Проверяем что прочитали все данные
    if pos != len(data):
        raise ValueError("Несоответствие размеров при распаковке данных")

    # Возвращаем словарь с распакованными данными
    return {
        'image_size': tuple(meta['image_size']),  # Размер изображения
        'luma_quant_matrix': np.array(meta['quant_matrices']['luma']),  # Матрица квантования яркости
        'chroma_quant_matrix': np.array(meta['quant_matrices']['chroma']),  # Матрица квантования цветности
        'Y_dc_bits': Y_dc_bits,  # DC коэффициенты яркости
        'Y_ac_bits': Y_ac_bits,  # AC коэффициенты яркости
        'Cb_dc_bits': Cb_dc_bits,  # DC коэффициенты Cb
        'Cb_ac_bits': Cb_ac_bits,  # AC коэффициенты Cb
        'Cr_dc_bits': Cr_dc_bits,  # DC коэффициенты Cr
        'Cr_ac_bits': Cr_ac_bits  # AC коэффициенты Cr
    }


def compress_to_file(image_path, output_file, quality=50, block_size=8):
    """Полный процесс сжатия изображения и сохранения в файл"""
    print("\nНачало обработки изображения...")

    # 1. Загрузка изображения
    rgb = load_image(image_path)
    height, width = rgb.shape[:2]  # Получаем высоту и ширину

    # 2. Конвертация в YCbCr
    print("2. Конвертация RGB -> YCbCr...")
    Y, Cb, Cr = rgb_to_ycbcr(rgb)  # Разделяем на компоненты
    Y = Y.astype(np.float32)  # Приводим к float32 для точности вычислений
    Cb = Cb.astype(np.float32)
    Cr = Cr.astype(np.float32)

    # 3. Даунсэмплинг цветовых компонентов (уменьшение разрешения)
    print("3. Даунсэмплинг цветовых компонентов...")
    Cb_down = downsample_channel(Cb, ratio=2)  # Уменьшаем Cb в 2 раза
    Cr_down = downsample_channel(Cr, ratio=2)  # Уменьшаем Cr в 2 раза

    # 4. Генерация матриц квантования
    print("4. Генерация матриц квантования...")
    luma_quant_matrix = get_quantization_matrix(quality, block_size)  # Для яркости
    chroma_quant_matrix = get_quantization_matrix(quality * 0.7, block_size)  # Для цветности (меньше качество)

    # 5. Обработка яркостного канала (Y)
    print("5. Обработка яркостного канала (Y)...")
    Y_dc_bits, Y_ac_bits = process_channel(Y, luma_quant_matrix, block_size, is_luma=True)

    # 6. Обработка цветовых каналов (Cb, Cr)
    print("6. Обработка цветовых каналов (Cb, Cr)...")
    Cb_dc_bits, Cb_ac_bits = process_channel(Cb_down, chroma_quant_matrix, block_size, is_luma=False)
    Cr_dc_bits, Cr_ac_bits = process_channel(Cr_down, chroma_quant_matrix, block_size, is_luma=False)

    # 7. Упаковка всех данных
    print("7. Упаковка данных...")
    packed = pack_data(
        image_size=(height, width),
        luma_quant_matrix=luma_quant_matrix,
        chroma_quant_matrix=chroma_quant_matrix,
        Y_dc_bits=Y_dc_bits,
        Y_ac_bits=Y_ac_bits,
        Cb_dc_bits=Cb_dc_bits,
        Cb_ac_bits=Cb_ac_bits,
        Cr_dc_bits=Cr_dc_bits,
        Cr_ac_bits=Cr_ac_bits
    )

    # 8. Сохранение в файл
    with open(output_file, 'wb') as f:  # Открываем файл в бинарном режиме
        f.write(packed)  # Записываем упакованные данные

    # 9. Расчет статистики сжатия
    original_size = rgb.nbytes  # Размер исходного изображения в байтах
    compressed_size = len(packed)  # Размер сжатых данных

    print(f"  Размер исходного изображения: {original_size / 1024:.2f} KB")
    print(f"  Размер после сжатия: {compressed_size / 1024:.2f} KB")
    print(f"  Коэффициент сжатия: {original_size / compressed_size:.2f}x")

    return compressed_size  # Возвращаем размер сжатых данных


def decompress_from_file(input_file, output_image_path=None):
    """Полный процесс декомпрессии изображения из файла"""
    print("\nДекомпрессия изображения...")

    # 1. Чтение файла
    with open(input_file, 'rb') as f:  # Открываем в бинарном режиме
        packed = f.read()  # Читаем все данные

    # 2. Распаковка данных
    print("1. Распаковка данных...")
    data = unpack_data(packed)  # Распаковываем метаданные и битовые потоки

    # 3. Восстановление яркостного канала (Y)
    print("2. Восстановление яркостного канала (Y)...")
    Y_restored = inverse_process_channel(
        data['Y_dc_bits'], data['Y_ac_bits'],
        data['luma_quant_matrix'], data['image_size'],
        block_size=8, is_luma=True
    )

    # 4. Восстановление цветовых каналов (Cb, Cr)
    print("3. Восстановление цветовых каналов (Cb, Cr)...")
    # Восстанавливаем с уменьшенным разрешением и затем увеличиваем
    Cb_restored = upsample_channel(
        inverse_process_channel(
            data['Cb_dc_bits'], data['Cb_ac_bits'],
            data['chroma_quant_matrix'],
            (data['image_size'][0] // 2, data['image_size'][1] // 2),
            block_size=8, is_luma=False
        ),
        data['image_size']
    )

    Cr_restored = upsample_channel(
        inverse_process_channel(
            data['Cr_dc_bits'], data['Cr_ac_bits'],
            data['chroma_quant_matrix'],
            (data['image_size'][0] // 2, data['image_size'][1] // 2),
            block_size=8, is_luma=False
        ),
        data['image_size']
    )

    # 5. Конвертация обратно в RGB
    print("4. Конвертация YCbCr -> RGB...")
    # Ограничиваем значения и преобразуем в uint8 перед конвертацией
    rgb_restored = ycbcr_to_rgb(
        np.clip(Y_restored, 0, 255).astype(np.uint8),
        np.clip(Cb_restored, 0, 255).astype(np.uint8),
        np.clip(Cr_restored, 0, 255).astype(np.uint8)
    )

    # 6. Сохранение или отображение результата
    if output_image_path:
        Image.fromarray(rgb_restored).save(output_image_path)  # Сохраняем изображение
        print(f"Изображение сохранено в {output_image_path}")

    return rgb_restored  # Возвращаем восстановленное изображение


def show_image_comparison(original, restored, original_size, compressed_size, quality):
    """Визуальное сравнение оригинального и восстановленного изображения"""
    plt.figure(figsize=(12, 6))  # Создаем фигуру размером 12x6 дюймов

    # Оригинальное изображение
    plt.subplot(1, 2, 1)  # 1 строка, 2 столбца, позиция 1
    plt.imshow(original)
    plt.title(f"Original\n{original_size / 1024:.2f} KB")  # Заголовок с размером

    # Восстановленное изображение
    plt.subplot(1, 2, 2)  # 1 строка, 2 столбца, позиция 2
    plt.imshow(restored)
    # Заголовок с качеством, размером и коэффициентом сжатия
    plt.title(f"Restored (Q={quality})\n{compressed_size / 1024:.2f} KB ({original_size / compressed_size:.2f}x)")

    plt.show()  # Отображаем график


if __name__ == "__main__":
    # Пример использования для изображения Lenna

    # Формируем пути к файлам
    input_image = get_absolute_path("images", "input_images", "Lenna.png")
    compressed_file = get_absolute_path("images", "compressed", "compressed_Lenna.bin")
    output_image = get_absolute_path("images", "output", "decompressed_Lenna.png")

    # Создаем необходимые директории если их нет
    os.makedirs(os.path.dirname(compressed_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_image), exist_ok=True)

    # Сжатие изображения
    print("=== Сжатие изображения Lenna ===")
    try:
        compress_to_file(input_image, compressed_file, quality=75)
    except Exception as e:
        print(f"Ошибка при сжатии: {str(e)}")
        exit(1)

    # Декомпрессия изображения
    print("\n=== Декомпрессия изображения Lenna ===")
    try:
        restored = decompress_from_file(compressed_file, output_image)
    except Exception as e:
        print(f"Ошибка при декомпрессии: {str(e)}")
        exit(1)

    # Сравнение результатов
    try:
        original = np.array(Image.open(input_image).convert("RGB"))
        show_image_comparison(original, restored,
                              original.nbytes,
                              os.path.getsize(compressed_file),
                              quality=75)
    except Exception as e:
        print(f"Ошибка при отображении результатов: {str(e)}")

    # Аналогичный процесс для изображения Flower
    input_image2 = get_absolute_path("images", "input_images", "Flower.jpg")
    compressed_file2 = get_absolute_path("images", "compressed", "compressed_Flower.bin")
    output_image2 = get_absolute_path("images", "output", "decompressed_Flower.png")

    print("\n=== Сжатие изображения Flower ===")
    try:
        compress_to_file(input_image2, compressed_file2, quality=75)
    except Exception as e:
        print(f"Ошибка при сжатии: {str(e)}")
        exit(1)

    print("\n=== Декомпрессия изображения Flower ===")
    try:
        restored2 = decompress_from_file(compressed_file2, output_image2)
    except Exception as e:
        print(f"Ошибка при декомпрессии: {str(e)}")
        exit(1)

    try:
        original2 = np.array(Image.open(input_image2).convert("RGB"))
        show_image_comparison(original2, restored2,
                              original2.nbytes,
                              os.path.getsize(compressed_file2),
                              quality=75)
    except Exception as e:
        print(f"Ошибка при отображении результатов: {str(e)}")