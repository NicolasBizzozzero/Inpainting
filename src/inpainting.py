from typing import List, Tuple
from copy import copy

import numpy as np

from src.picture_tools.picture import Picture, VALUE_MISSING_PIXEL
from src.linear.cost_function import *
from src.linear.gradient_descent import DescenteDeGradient
from src.linear.linear_regression import Initialisation, LinearRegression
from src.picture_tools.tools import get_center


class InPainting:
    def __init__(self, patch_size: int, step: int = 1, max_missing_pixel: int = 0,
                 value_missing_pixel: int = VALUE_MISSING_PIXEL, loss: callable = l1, loss_g: callable = l1_g,
                 max_iter: int = 10000, eps: float = 0.01, biais: bool = True,
                 type_descente: DescenteDeGradient = DescenteDeGradient.BATCH, taille_batch: int = 50,
                 initialisation: Initialisation = Initialisation.RANDOM, alpha: float = 1.0):
        self.patch_size = patch_size
        self.step = step
        self.max_missing_pixel = max_missing_pixel
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

        self.classifier = LinearRegression(loss=loss, loss_g=loss_g, max_iter=max_iter, eps=eps, biais=biais,
                                           type_descente=type_descente, taille_batch=taille_batch,
                                           initialisation=initialisation, alpha=alpha)

    def inpaint(self, picture: Picture):
        while self.value_missing_pixel in picture.pixels:
            # On récupère le patch centré sur le prochain pixel à traiter
            next_pixel = self._get_next_pixel(picture.pixels)
            patch = picture.get_patch(*next_pixel, size=self.patch_size)

            # On approxime le patch en fonction du dictionnaire
            dictionary = picture.get_dictionnaire(self.patch_size, self.step, self.max_missing_pixel)
            print(dictionary.shape)
            print(dictionary[0].shape)
            self.classifier.fit(dictionary, patch)

            # On remplace le pixel qu'on souhaitait remplir
            new_value = np.dot(picture.pixels[next_pixel], self.classifier.w.T)
            picture.pixels[next_pixel] = new_value

    def _get_next_pixel(self, pixels: np.ndarray) -> Tuple[int, int]:
        """ Return the next pixel to be painted.
        This pixel is found by assigning a priority value to all remaining pixels in the picture, then by returning the
        one with the maximal priority.
        """
        priorities = np.ndarray(shape=pixels.shape)

        # Set all not-missing pixels to zero
        priorities[pixels != self.value_missing_pixel] = 0

        # Set a random value for all not missing pixels
        # TODO: Find the pixel
        missing_pixels_x, missing_pixels_y, *_ = np.where(pixels == self.value_missing_pixel)
        pixel_choiced = zip(missing_pixels_x, missing_pixels_y).__next__()
        return pixel_choiced


if __name__ == "__main__":
    pass
