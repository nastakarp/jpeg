import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def rgb_to_ycbcr(rgb_image):
    """
    Преобразует RGB изображение в YCbCr.

    Параметры:
        rgb_image (numpy.ndarray): Изображение в формате RGB (H x W x 3).

    Возвращает:
        tuple: (Y, Cb, Cr) — три канала в формате numpy.ndarray.
    """
    if rgb_image.dtype != np.float32 and rgb_image.dtype != np.float64:
        rgb_image = rgb_image.astype(np.float32)  # Чтобы избежать переполнения

    # Разделяем каналы
    R = rgb_image[:, :, 0]
    G = rgb_image[:, :, 1]
    B = rgb_image[:, :, 2]

    # Вычисляем Y, Cb, Cr
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    # Ограничиваем значения до [0, 255] (на случай округления)
    Y = np.clip(Y, 0, 255)
    Cb = np.clip(Cb, 0, 255)
    Cr = np.clip(Cr, 0, 255)

    return Y, Cb, Cr


def ycbcr_to_rgb(Y, Cb, Cr):
    Y = Y.astype(np.float32)
    Cb = Cb.astype(np.float32) - 128
    Cr = Cr.astype(np.float32) - 128

    R = Y + 1.402 * Cr
    G = Y - 0.34414 * Cb - 0.71414 * Cr
    B = Y + 1.772 * Cb

    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)
