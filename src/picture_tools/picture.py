# -*- coding: utf-8 -*-
import os.path
import random

from typing import Tuple

import numpy as np

from matplotlib import pyplot as plt
from numpy import uint8

from src.picture_tools.colormodel import ColorModel, change_color_model
from src.common.math import normalize

VALUE_MISSING_PIXEL = np.ones((3,)) * -100    # Value representing a missing pixel
VALUE_OUT_OF_BOUNDS = np.ones((3,)) * -1000   # Value representing an out-of-bounds pixel
VALUE_SHOWING_MISSING_PIXEL = np.array([-1])  # Set missing pixels to this value before plotting


class Picture:
    """ Implement methods for loading a picture and manipulate it for ML purposes.
    Also support multiple color models.

    Attributes :
        - picture_path : str, the path of the picture after loading it.
        - codage : Codage, le codage des pixels de l'image (par défaut, HSV).
        - pixels : np.ndarray, the pixels of the loaded picture. They are always normalized between [-1, 1].
        - height : int, picture's height.
        - width : int, picture's width.
    """

    def __init__(self, picture_path: str, color_model: ColorModel = ColorModel.HSV):
        self.picture_path = picture_path
        self.color_model = color_model
        pixels, self.height, self.width = _load_pixels(picture_path)
        self.pixels = change_color_model(pixels, ColorModel.RGB, self.color_model)

    def show(self, show: bool = True) -> None:
        """ Plot the picture with the matplotlib library.
        :param show: If True, show the picture after having plotted it.
        """
        picture = self._get_showable_picture()
        plt.axis("off")
        plt.imshow(picture)
        if show:
            plt.show()

    def save(self, picture_path: str = None) -> None:
        """ Save the picture.
        If no `picture_path` is given, the picture will be saved with the same basename given during loading.
        """
        global VALUE_MISSING_PIXEL, VALUE_OUT_OF_BOUNDS, VALUE_SHOWING_MISSING_PIXEL

        if picture_path is None:
            picture_path = os.path.basename(self.picture_path)

        # Remove missing values
        picture = np.copy(self.pixels)
        picture[picture == VALUE_MISSING_PIXEL] = VALUE_SHOWING_MISSING_PIXEL
        picture[picture == VALUE_OUT_OF_BOUNDS] = VALUE_SHOWING_MISSING_PIXEL

        picture = change_color_model(self.pixels, self.color_model, ColorModel.RGB)
        picture = normalize(picture, 0, 255, -1, 1).astype(uint8)
        plt.imshow(picture)
        plt.savefig(picture_path)

    def copy(self):
        new_picture = Picture.__new__(Picture)
        new_picture.picture_path = self.picture_path
        new_picture.color_model = self.color_model
        new_picture.height, new_picture.width = self.height, self.width
        new_picture.pixels = np.copy(self.pixels)
        return new_picture

    def add_noise(self, threshold: float = 0.05):
        """ Add random noise to the picture.
        :param threshold: Chance of each pixel to be noisyfied.
        """
        global VALUE_MISSING_PIXEL

        for x in range(self.height):
            for y in range(self.width):
                self.pixels[x, y] = VALUE_MISSING_PIXEL if random.random(
                ) < threshold else self.pixels[x, y]

    def add_rectangle(self, x: int, y: int, height: int, width: int) -> None:
        """ Add a random rectangle of noise to the picture.
        :param x: x coordinate of the pixel in the upper left corner of the rectangle.
        :param y: y coordinate of the pixel in the upper left corner of the rectangle.
        :param height: Height of the rectangle.
        :param width: Width of the rectangle.
        """
        global VALUE_MISSING_PIXEL

        self.pixels[x:x + height, y:y + width] = VALUE_MISSING_PIXEL

    def get_pixel(self, x: int, y: int) -> np.ndarray:
        """ Return picture's pixel located at indexes (x, y).
        :param x: Column's index.
        :param y: Line's index.
        """
        return self.pixels[x, y]

    def get_patch(self, x: int, y: int, size: int) -> np.ndarray:
        """ Return the picture's patch centered in (x, y).
        :param x: Column's index.
        :param y: Line's index.
        :param size: Length of the patch.
        """
        global VALUE_OUT_OF_BOUNDS

        if not self.out_of_bounds_patch(x, y, size):
            return self.pixels[x - (size // 2):x + (size // 2) + 1, y - (size // 2): y + (size // 2) + 1]
        else:
            patch = []
            for index_x in range(x - (size // 2), x + (size // 2) + 1):
                new_line = []
                for index_y in range(y - (size // 2), y + (size // 2) + 1):
                    if not self.out_of_bounds(index_x, index_y):
                        new_line.append(self.pixels[index_x, index_y])
                    else:
                        new_line.append(VALUE_OUT_OF_BOUNDS)
                patch.append(np.array(new_line))
            return np.array(patch)

    def get_patches(self) -> np.ndarray:
        """ Return indexes of all patches center who contains a missing pixel. """
        global VALUE_MISSING_PIXEL

        return np.array([np.array([x, y]) for x in range(self.height) for y in range(self.width)
                         if (self.get_pixel(x, y) == VALUE_MISSING_PIXEL).all()])

    def get_dictionary(self, size: int, step: int = 1, max_missing_pixel: int = 0) -> np.ndarray:
        """ Return a learnable dictionary of patches from the picture.
        A dictionary is a list containing same-size patches with no missing pixel.
        :param size: Size of each patch.
        :param step: Size of iteration step.
        :param max_missing_pixel: The maximal number of missing pixels tolerated in a patch.
        """
        global VALUE_MISSING_PIXEL

        result = []
        for x in range(0, self.height, step):
            for y in range(0, self.width, step):
                if not self.out_of_bounds_patch(x, y, size):
                    patch = self.get_patch(x, y, size)
                    if len(patch[patch == VALUE_MISSING_PIXEL]) // 3 <= max_missing_pixel:
                        result.append(patch)
        return np.array(result)

    def out_of_bounds(self, x: int, y: int) -> bool:
        """ Check if the pixel located at (x, y) is out of the bounds of the picture.
        >>> picture = Picture(LENA_COLOR_512, color_model=ColorModel.RGB)
        >>> picture.out_of_bounds(0, 0)
        False
        >>> picture.out_of_bounds(-1, 0)
        True
        >>> picture.out_of_bounds(0, -1)
        True
        >>> picture.out_of_bounds(-1, -1)
        True
        >>> picture.out_of_bounds(511, 511)
        False
        >>> picture.out_of_bounds(512, 511)
        True
        >>> picture.out_of_bounds(512, 512)
        True
        >>> picture.out_of_bounds(512, -1)
        True
        >>> picture.out_of_bounds(-1, 512)
        True
        """
        return not (0 <= x < self.height and 0 <= y < self.width)

    def out_of_bounds_patch(self, x: int, y: int, size: int) -> bool:
        """ Check if a given patch is out of the bounds of the picture. """
        return (x - (size // 2) < 0) or \
               (x + (size // 2) + 1 >= self.height) or \
               (y - (size // 2) < 0) or \
               (y + (size // 2) + 1 >= self.width)

    def _get_showable_picture(self) -> np.ndarray:
        """ Return the picture in a showable format (as in a format which can be plotted by invocating `plt.imshow`on
        it.
        """
        global VALUE_MISSING_PIXEL, VALUE_OUT_OF_BOUNDS, VALUE_SHOWING_MISSING_PIXEL

        pixels = np.copy(self.pixels)

        # Fill the missing pixels with an interpretable value
        pixels[pixels == VALUE_MISSING_PIXEL] = VALUE_SHOWING_MISSING_PIXEL
        pixels[pixels == VALUE_OUT_OF_BOUNDS] = VALUE_SHOWING_MISSING_PIXEL

        # Change the format of the picture to the RGB format (for better visibility)
        pixels = change_color_model(pixels, self.color_model, ColorModel.RGB)

        # Normalise the values of the pixel from [-1, 1] to [0, 255]
        pixels = normalize(pixels, 0, 255, -1, 1).astype(uint8)

        return pixels


def get_center(pixels: np.ndarray) -> np.ndarray:
    """ Return the pixel at the center of the pixels. """
    return pixels[(pixels.shape[0] - 1) // 2, (pixels.shape[1] - 1) // 2]


def show_patch(patch: np.ndarray, color_model: ColorModel = ColorModel.HSV, show: bool = True):
    """ Plot a given patch with the matplotlib library.
    :param patch: The patch to plot
    :param color_model: The color model used in the patch
    :param show: If True, show the patch after having plotted it.
    """
    global VALUE_MISSING_PIXEL, VALUE_OUT_OF_BOUNDS, VALUE_SHOWING_MISSING_PIXEL

    # Remove missing values
    new_patch = np.copy(patch)
    new_patch[patch == VALUE_MISSING_PIXEL] = VALUE_SHOWING_MISSING_PIXEL
    new_patch[patch == VALUE_OUT_OF_BOUNDS] = VALUE_SHOWING_MISSING_PIXEL

    new_patch = change_color_model(new_patch, color_model, ColorModel.RGB)
    new_patch = normalize(new_patch, 0, 255, -1, 1).astype(uint8)
    plt.imshow(new_patch)
    if show:
        plt.show()


def flatten(patch: np.ndarray) -> np.ndarray:
    """ Convert a patch of pixels in a vector. """
    return np.copy(patch).reshape(-1)


def unflatten(vector: np.ndarray, size_patch: int) -> np.ndarray:
    """ Convert a vector in a patch of pixels. """
    return np.copy(vector).reshape(size_patch, size_patch, 3)


def out_of_bounds(pixels: np.ndarray, x: int, y: int) -> bool:
    """ Check if the pixel located at (x, y) is out of the bounds of the picture. """
    return not (0 <= x < pixels.shape[0] and 0 <= y < pixels.shape[1])


def out_of_bounds_patch(pixels: np.ndarray, x: int, y: int, size: int) -> bool:
    """ Check if the patch located at (x, y) is out of the bounds of the picture. """
    return (x - (size // 2) <= 0) or \
           (x + (size // 2) + 1 < pixels.shape[0]) or \
           (y - (size // 2) <= 0) or \
           (y + (size // 2) + 1 < pixels.shape[1])


def get_patch(pixels: np.ndarray, x: int, y: int, size: int,
              value_out_of_bounds: np.ndarray = VALUE_OUT_OF_BOUNDS) -> np.ndarray:
    """ Return the picture's patch centered in (x, y).
    :param pixels: The pixels from which the patch will be extracted.
    :param x: The column's index.
    :param y: The line's index.
    :param size: The length of the patch.
    :param value_out_of_bounds: The value too return if a pixel is out of the bounds of the picture.
    """
    if not out_of_bounds_patch(pixels, x, y, size):
        return pixels[x - (size // 2):x + (size // 2) + 1, y - (size // 2): y + (size // 2) + 1]
    else:
        patch = []
        for index_x in range(x - (size // 2), x + (size // 2) + 1):
            new_line = []
            for index_y in range(y - (size // 2), y + (size // 2) + 1):
                if not out_of_bounds(pixels, index_x, index_y):
                    new_line.append(pixels[index_x, index_y])
                else:
                    new_line.append(value_out_of_bounds)
            patch.append(np.array(new_line))
        return np.array(patch)


def iter_patch(x: int, y: int, size: int):
    """ Iterate trough the indexes of a patch. """
    for index_y in range(y - (size // 2), y + (size // 2) + 1):
        for index_x in range(x - (size // 2), x + (size // 2) + 1):
            yield index_x, index_y


def iter_patch_empty(pixels: np.ndarray, x: int, y: int, size: int):
    """ Yields indexes of an empty pixel inside the patch centered at (x, y) """
    global VALUE_MISSING_PIXEL

    for index_x, index_y in iter_patch(x, y, size):
        if not out_of_bounds(pixels, index_x, index_y):
            if all(pixels[index_x, index_y] == VALUE_MISSING_PIXEL):
                yield index_x, index_y


def _load_pixels(picture_path: str) -> Tuple[np.ndarray, int, int]:
    """ Load a picture's pixels an return them.
    If the picture is in color, return its pixels in the RGB color model (while ignoring alpha channel).
    If the picture is in B&W, return its pixels in the RGB color model with the same intensity in the 3 channels.
    """
    picture = plt.imread(picture_path)

    if len(picture.shape) == 3:
        # The picture is in the RGB color model, we then need to ignore alpha channel.
        pixels = picture[:, :, :3]
    else:
        # The picture is in B&W, we need to stack its intensity 3 times to mimick a RGB color model
        pixels = np.dstack([picture] * 3)

    picture_height, picture_width, _ = pixels.shape

    # Normalise pixels between the (-1, 1) interval, for better results with ML models
    pixels = normalize(pixels, -1, 1, 0, 255)

    return pixels, picture_height, picture_width


if __name__ == "__main__":
    pass
