import numpy as np


def get_category(value):
    """
    Определяет категорию (Size) и дополнительные биты для значения согласно JPEG.
    Обрабатывает как целые, так и вещественные числа, округляя их.

    Args:
        value: Входное значение (целое или вещественное)

    Returns:
        tuple: (категория, биты амплитуды)
    """
    # Преобразуем в целое число с округлением для стандартизации
    int_value = int(round(float(value)))

    if int_value == 0:
        return 0, ""  # Нулевое значение не требует битов амплитуды

    abs_value = abs(int_value)
    # Вычисляем минимальное количество бит для представления числа
    category = int(np.ceil(np.log2(abs_value + 1)))

    # Вычисляем биты амплитуды
    if int_value > 0:
        amplitude = int_value  # Прямое представление для положительных
    else:
        # Дополнение для отрицательных чисел
        amplitude = int_value + (1 << category) - 1

    # Преобразуем в бинарную строку с ведущими нулями
    amplitude_bits = bin(amplitude)[2:].zfill(category)
    return category, amplitude_bits


# Стандартные таблицы Хаффмана для DC-коэффициентов
STD_LUMA_DC_HUFFMAN = {  # Таблица для яркостной компоненты
    0: '00',
    1: '010',
    2: '011',
    3: '100',
    4: '101',
    5: '110',
    6: '1110',
    7: '11110',
    8: '111110',
    9: '1111110',
    10: '11111110',
    11: '111111110'
}

STD_CHROMA_DC_HUFFMAN = {  # Таблица для цветностных компонент
    0: '00',
    1: '01',
    2: '10',
    3: '110',
    4: '1110',
    5: '11110',
    6: '111110',
    7: '1111110',
    8: '11111110',
    9: '111111110',
    10: '1111111110',
    11: '11111111110'
}


def encode_dc_coefficients(dc_coeffs, is_luma=True):
    """
    Кодирование DC-коэффициентов с использованием разностного кодирования и Хаффмана

    Args:
        dc_coeffs: Массив DC-коэффициентов
        is_luma: Флаг яркостной компоненты

    Returns:
        str: Битовая строка закодированных данных
    """
    # Преобразуем в список float для унификации обработки
    dc_coeffs = [float(x) for x in dc_coeffs]

    # Вычисляем разности между соседними DC-коэффициентами
    delta_dc = [dc_coeffs[0]]  # Первый элемент без разности
    for i in range(1, len(dc_coeffs)):
        delta_dc.append(dc_coeffs[i] - dc_coeffs[i - 1])

    # Выбираем таблицу Хаффмана в зависимости от типа компоненты
    huffman_table = STD_LUMA_DC_HUFFMAN if is_luma else STD_CHROMA_DC_HUFFMAN
    bitstream = ""

    for diff in delta_dc:
        category, amplitude_bits = get_category(diff)
        if category not in huffman_table:
            raise ValueError(f"Invalid category {category} for DC coefficient")
        # Добавляем код Хаффмана и биты амплитуды
        bitstream += huffman_table[category] + amplitude_bits

    return bitstream


def decode_dc_coefficients(bitstream, num_blocks, is_luma=True):
    """
    Декодирование DC-коэффициентов из битовой строки

    Args:
        bitstream: Битовая строка с закодированными данными
        num_blocks: Количество блоков для декодирования
        is_luma: Флаг яркостной компоненты

    Returns:
        np.ndarray: Массив восстановленных DC-коэффициентов
    """
    huffman_table = STD_LUMA_DC_HUFFMAN if is_luma else STD_CHROMA_DC_HUFFMAN
    # Создаем обратную таблицу Хаффмана для декодирования
    reverse_huffman = {v: k for k, v in huffman_table.items()}

    delta_dc = np.zeros(num_blocks)
    current_pos = 0

    for i in range(num_blocks):
        # Поиск валидного кода Хаффмана
        found = False
        for code_len in range(1, 13):  # Максимальная длина кода в таблицах
            if current_pos + code_len > len(bitstream):
                break
            code = bitstream[current_pos:current_pos + code_len]
            if code in reverse_huffman:
                category = reverse_huffman[code]
                current_pos += code_len
                found = True
                break

        if not found:
            raise ValueError("Invalid Huffman code")

        # Декодирование значения
        if category == 0:  # Особый случай нулевого значения
            delta_dc[i] = 0
            continue

        if current_pos + category > len(bitstream):
            raise ValueError("Not enough bits")

        # Читаем биты амплитуды
        amplitude_bits = bitstream[current_pos:current_pos + category]
        current_pos += category
        amplitude = int(amplitude_bits, 2)

        # Восстанавливаем отрицательные значения
        if amplitude < (1 << (category - 1)):
            delta_dc[i] = amplitude - (1 << category) + 1
        else:
            delta_dc[i] = amplitude

    # Восстановление DC-коэффициентов из разностей
    dc_coeffs = np.zeros(num_blocks)
    dc_coeffs[0] = delta_dc[0]  # Первый коэффициент без изменений
    for i in range(1, num_blocks):
        dc_coeffs[i] = dc_coeffs[i - 1] + delta_dc[i]  # Накопительная сумма

    return dc_coeffs