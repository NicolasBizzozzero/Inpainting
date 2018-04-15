# -*- coding: utf-8 -*-
""" Ce module contient des fonctions et methodes permettant de convertir le contenu d'une image dans un certain codage
vers un autre.
"""

from enum import IntEnum
import colorsys

import numpy as np

from src.common import normalize


class Codage(IntEnum):
    RGB = 0
    HSV = 1


class UnknownCodage(Exception):
    def __init__(self, codage):
        Exception.__init__("Le codage {} n'est actuellement pas supporté ou néexiste pas".format(codage))


def change_codage(pixels: np.ndarray, codage_src: Codage, codage_dest: Codage) -> np.ndarray:
    """ Change le codage des pixels puis les retourne.
    :param: pixels, les pixels dont on veut convertir le codage.
    :param: codage_src, le codage dans lequel les pixels sont actuellement encodés.
    :param: codage_dest, le codage dans lequel on veut convertir nos pixels.
    :return: Les pixels encodés dans e codage `codage_dest`.
    """
    if codage_src == Codage.RGB:
        if codage_dest == Codage.RGB:
            return pixels
        elif codage_dest == Codage.HSV:
            return _rgb_to_hsv(pixels)
    elif codage_src == Codage.HSV:
        if codage_dest == Codage.RGB:
            return _hsv_to_rgb(pixels)
        elif codage_dest == Codage.HSV:
            return pixels
    else:
        raise UnknownCodage(codage_dest)


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
