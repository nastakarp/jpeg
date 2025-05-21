import numpy as np


def split_into_blocks(image: np.ndarray, block_size: int = 8, fill_value: int = 0) -> np.ndarray:
    """
    Разбивает изображение на блоки NxN с заполнением при необходимости

    Параметры:
        image (np.ndarray): Входное изображение (H x W) или (H x W x C)
        block_size (int): Размер блока (по умолчанию 8)
        fill_value (int): Значение для заполнения (по умолчанию 0)

    Возвращает:
        np.ndarray: Массив блоков формы (num_blocks_h, num_blocks_w, block_size, block_size[, C])
    """
    # Получаем размеры изображения
    if len(image.shape) == 2:  # Проверяем grayscale (2D) изображение
        h, w = image.shape  # Высота и ширина
        channels = 1  # Один канал для grayscale
    else:  # Цветное изображение (3D)
        h, w, channels = image.shape  # Высота, ширина и количество каналов

    # Вычисляем количество блоков с округлением вверх
    num_blocks_h = (h + block_size - 1) // block_size  # Блоков по вертикали
    num_blocks_w = (w + block_size - 1) // block_size  # Блоков по горизонтали

    # Вычисляем размеры с дополнением
    padded_h = num_blocks_h * block_size  # Новая высота с padding
    padded_w = num_blocks_w * block_size  # Новая ширина с padding

    # Создаем массив с заполнением
    if len(image.shape) == 2:  # Для grayscale
        padded = np.full((padded_h, padded_w), fill_value, dtype=image.dtype)
        padded[:h, :w] = image  # Копируем оригинальное изображение
    else:  # Для цветного
        padded = np.full((padded_h, padded_w, channels), fill_value, dtype=image.dtype)
        padded[:h, :w, :] = image  # Копируем оригинальное изображение

    # Разбиваем на блоки
    if len(image.shape) == 2:  # Для grayscale
        blocks = padded.reshape(num_blocks_h, block_size, num_blocks_w, block_size)
        blocks = blocks.transpose(0, 2, 1, 3)  # Меняем оси для правильного порядка
    else:  # Для цветного
        blocks = padded.reshape(num_blocks_h, block_size, num_blocks_w, block_size, channels)
        blocks = blocks.transpose(0, 2, 1, 3, 4)  # Меняем оси для правильного порядка

    return blocks


def assemble_blocks(blocks):
    """
    Собирает блоки обратно в изображение

    Параметры:
        blocks (np.ndarray): Массив блоков от split_into_blocks()

    Возвращает:
        np.ndarray: Восстановленное изображение
    """
    if len(blocks.shape) == 4:  # Для grayscale (4D: num_h, num_w, bs, bs)
        num_h, num_w, bs, _ = blocks.shape
        # Транспонируем и собираем обратно
        return blocks.transpose(0, 2, 1, 3).reshape(num_h * bs, num_w * bs)
    else:  # Для color (5D: num_h, num_w, bs, bs, c)
        num_h, num_w, bs, _, c = blocks.shape
        # Транспонируем и собираем обратно
        return blocks.transpose(0, 2, 1, 3, 4).reshape(num_h * bs, num_w * bs, c)