from enum import IntEnum
from typing import Tuple
from random import choice

import numpy as np

from src.picture_tools.picture import VALUE_MISSING_PIXEL


_CACHED_PRIORITIES = None


class PixelChoosingStrategy(IntEnum):
    FIRST_PIXEL = 0
    RANDOM = 1
    BOUNDARY_TO_CENTER = 2


class UnknownPixelChoosingStrategy(Exception):
    def __init__(self, pixel_choosing_strategy):
        Exception.__init__(("The pixel choosing strategy {} is currently not available or does not exists"
                            "").format(pixel_choosing_strategy))


def choose_pixel(pixels: np.ndarray, pixel_choosing_strategy: PixelChoosingStrategy,
                 value_missing_pixel: int = VALUE_MISSING_PIXEL) -> Tuple[int, int]:
    if pixel_choosing_strategy == PixelChoosingStrategy.FIRST_PIXEL:
        return _first_pixel(pixels, value_missing_pixel)
    elif pixel_choosing_strategy == PixelChoosingStrategy.RANDOM:
        return _random(pixels, value_missing_pixel)
    elif pixel_choosing_strategy == PixelChoosingStrategy.BOUNDARY_TO_CENTER:
        return _boundary_to_center(pixels, value_missing_pixel)
    else:
        raise UnknownPixelChoosingStrategy(pixel_choosing_strategy)


def _first_pixel(pixels: np.ndarray, value_missing_pixel) -> Tuple[int, int]:
    missing_pixels_x, missing_pixels_y, *_ = np.where(pixels == value_missing_pixel)
    return zip(missing_pixels_x, missing_pixels_y).__next__()


def _random(pixels: np.ndarray, value_missing_pixel) -> Tuple[int, int]:
    missing_pixels_x, missing_pixels_y, *_ = np.where(pixels == value_missing_pixel)
    return choice(zip(missing_pixels_x, missing_pixels_y))


def _boundary_to_center(pixels: np.ndarray, value_missing_pixel) -> Tuple[int, int]:
    global _CACHED_PRIORITIES

    if _CACHED_PRIORITIES is not None:
        # Chose a priority, delete it then return it
        # If empty, replace by None
        pass

    # Else, compute the priorities then assign them


if __name__ == "__main__":
    pass
