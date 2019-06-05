# -*- coding: utf-8 -*-

from enum import IntEnum
import colorsys

import numpy as np

from src.common.math import normalize


class ColorModel(IntEnum):
    RGB = 0
    HSV = 1


class UnknownColorModel(Exception):
    def __init__(self, color_model):
        Exception.__init__(self, "The color model \"{color_model}\" is not supported or does not exists".format(
            color_model=color_model
        ))


def change_color_model(pixels: np.ndarray, color_model_src: ColorModel, color_model_dest: ColorModel) -> np.ndarray:
    """ Change pixels color model.
    :param pixels: the pixels from which to change the color model
    :param color_model_src:, the color model in which pixels are currently encoded.
    :param: color_model_dest, the color model in which you want your pixels to be encoded.
    :return: Encoded pixels in the `color_model_dest` color model.
    """
    if color_model_src == ColorModel.RGB:
        if color_model_dest == ColorModel.RGB:
            return pixels
        elif color_model_dest == ColorModel.HSV:
            return _rgb_to_hsv(pixels)
    elif color_model_src == ColorModel.HSV:
        if color_model_dest == ColorModel.RGB:
            return _hsv_to_rgb(pixels)
        elif color_model_dest == ColorModel.HSV:
            return pixels
    else:
        raise UnknownColorModel(color_model_dest)


def _rgb_to_hsv(pixels: np.ndarray) -> np.ndarray:
    # The colorsys function uses values normalized between 0 and 1
    pixels = normalize(pixels, 0, 1)

    for x in range(pixels.shape[0]):
        for y in range(pixels.shape[1]):
            pixels[x, y] = colorsys.rgb_to_hsv(*pixels[x, y])

    # Get back to the original -1, 1 normalisation
    pixels = normalize(pixels, -1, 1)

    return pixels


def _hsv_to_rgb(pixels: np.ndarray) -> np.ndarray:
    # The colorsys function uses values normalized between 0 and 1
    pixels = normalize(pixels, 0, 1)

    for x in range(pixels.shape[0]):
        for y in range(pixels.shape[1]):
            pixels[x, y] = colorsys.hsv_to_rgb(*pixels[x, y])

    # Get back to the original -1, 1 normalisation
    pixels = normalize(pixels, -1, 1)

    return pixels


if __name__ == "__main__":
    pass
