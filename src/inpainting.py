from typing import List

import numpy as np

from src.picture_tools.picture import Picture, VALUE_MISSING_PIXEL
from src.linear.cost_function import *
from src.linear.gradient_descent import DescenteDeGradient
from src.linear.linear_regression import Initialisation, LinearRegression


class InPainting:
    def __init__(self, value_missing_pixel: int = VALUE_MISSING_PIXEL, loss: function = l1, loss_g: function  = l1_g,
                 max_iter: int = 10000, eps: float = 0.01, biais: bool = True,
                 type_descente: DescenteDeGradient = DescenteDeGradient.BATCH, taille_batch: int = 50,
                 initialisation: Initialisation = Initialisation.RANDOM, alpha: float = 1.0):
        self.loss = loss
        self.loss_g = loss_g
        self.max_iter = max_iter
        self.eps = eps
        self.biais = biais
        self.type_descente = type_descente
        self.taille_batch = taille_batch
        self.initialisation = initialisation
        self.alpha = alpha
        self.value_missing_pixel = value_missing_pixel

    def inpaint(self, picture: Picture) -> Picture:
        while self.value_missing_pixel in picture.pixels:
            next_pixel = self._get_next_pixel(picture.pixels)


    def _get_next_pixel(self, pixels: np.ndarray) -> np.ndarray:
        """ Return the next pixel to be painted.
        This pixel is found by assigning a priority value to all remaining pixels in the picture, then by returning the
        one with the maximal priority.
        """
        priorities = np.ndarray(shape=pixels.shape)

        # Set all not-missing pixels to zero
        priorities[pixels != self.value_missing_pixel] = 0

        # Set a random value for all not missing pixels
        # TODO: Find the pixel
        pixel_choiced = np.random.choice(pixels[pixels == self.value_missing_pixel])
        return pixels[pixel_choiced]

        return priorities.argmax()


if __name__ == "__main__":
    pass
