# -*- coding: utf-8 -*-
""" Ce module contient des fonctions et methodes permettant de convertir le contenu d'une image dans un certain codage
vers un autre.
"""

from enum import IntEnum

import numpy as np


class Codage(IntEnum):
    NOIR_ET_BLANC = 0
    RGB = 1
    HSV = 2


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
    if codage_src == Codage.NOIR_ET_BLANC:
        if codage_dest == Codage.NOIR_ET_BLANC:
            return pixels
        elif codage_dest == Codage.RGB:
            return _black_and_white_to_rgb(pixels)
        elif codage_dest == Codage.HSV:
            return _black_and_white_to_hsv(pixels)
    elif codage_src == Codage.RGB:
        if codage_dest == Codage.NOIR_ET_BLANC:
            return _rgb_to_black_and_white(pixels)
        elif codage_dest == Codage.RGB:
            return pixels
        elif codage_dest == Codage.HSV:
            return _rgb_to_hsv(pixels)
    elif codage_src == Codage.HSV:
        if codage_dest == Codage.NOIR_ET_BLANC:
            return _hsv_to_black_and_white(pixels)
        elif codage_dest == Codage.RGB:
            return _hsv_to_rgb(pixels)
        elif codage_dest == Codage.HSV:
            return pixels
    else:
        raise UnknownCodage(codage_dest)


def _black_and_white_to_rgb(pixels: np.ndarray) -> np.ndarray:
    pass


def _black_and_white_to_hsv(pixels: np.ndarray) -> np.ndarray:
    pass


def _rgb_to_hsv(pixels: np.ndarray) -> np.ndarray:
    pass


def _rgb_to_black_and_white(pixels: np.ndarray) -> np.ndarray:
    pass


def _hsv_to_rgb(pixels: np.ndarray) -> np.ndarray:
    pass


def _hsv_to_black_and_white(pixels: np.ndarray) -> np.ndarray:
    pass


if __name__ == "__main__":
    pass
