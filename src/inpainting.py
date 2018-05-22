from typing import List, Tuple
from copy import copy

import numpy as np
from sklearn.linear_model import Lasso
from progressbar import ProgressBar, Percentage, Counter, Timer, ETA

from src.common.decorators import time_this
from src.picture_tools.picture import Picture, VALUE_MISSING_PIXEL, get_center, flatten, unflatten, show_patch
from src.linear.cost_function import *


PB_WIDGETS = ["Inpainting: processed ", Counter(), " pixels [", Percentage(), "], ", Timer(), ", ", ETA()]


class InPainting:
    def __init__(self, patch_size: int, step: int = None, value_missing_pixel: int = VALUE_MISSING_PIXEL,
                 alpha: float = 1.0, max_iterations: int = 1000, tolerance: float = 0.0001):
        self.patch_size = patch_size
        self.step = patch_size if step is None else step
        self.alpha = alpha
        self.value_missing_pixel = value_missing_pixel

        classifiers_kwaargs = {"alpha": alpha, "copy_X": True, "fit_intercept": True, "max_iter": max_iterations,
                               "normalize": False, "positive": False, "precompute": False, "random_state": None,
                               "selection": 'cyclic', "tol": tolerance, "warm_start": False}
        self._classifier_hue = Lasso(**classifiers_kwaargs)
        self._classifier_saturation = Lasso(**classifiers_kwaargs)
        self._classifier_value = Lasso(**classifiers_kwaargs)

    def inpaint(self, picture: Picture) -> Picture:
        picture = picture.copy()

        # Initialisation de la barre de progression
        progress_bar = ProgressBar(widgets=PB_WIDGETS,
                                   maxval=len(picture.pixels[picture.pixels == self.value_missing_pixel]) // 3,
                                   minval=0)
        progress_bar.start()

        # On récupère le dictionnaire
        dictionary = picture.get_dictionnaire(self.patch_size, self.step, max_missing_pixel=0)

        while self.value_missing_pixel in picture.pixels:
            # On récupère le patch centré sur le prochain pixel à traiter
            next_pixel = self._get_next_pixel(picture.pixels)

            # On reconstruit le pixel choisi
            next_pixel_value = self._get_next_pixel_value(picture, dictionary, *next_pixel)
            picture.pixels[next_pixel] = next_pixel_value

            # On met à jour la barre de progression
            progress_bar.update(progress_bar.value + 1)

        progress_bar.finish()
        return picture

    def _get_next_pixel(self, pixels: np.ndarray, strategy) -> Tuple[int, int]:
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

    def _get_next_pixel_value(self, picture: Picture, dictionary, next_pixel_x, next_pixel_y) -> np.ndarray:
        # On récupère le patch à approximer
        # TODO: parfois, patch est vide
        patch = picture.get_patch(next_pixel_x, next_pixel_y, size=self.patch_size)

        # Construction des ensembles d'apprentissage
        datax_hue, datax_saturation, datax_value, datay_hue, datay_saturation, datay_value = [], [], [], [], [], []
        # On itère sur chaque pixel du patch à reconstruire
        for x, y in zip(range(self.patch_size), range(self.patch_size)):
            # Si on tombe sur une valeur manquante, on ne l'ajoute évidemment pas (impossible à apprendre)
            if all(self.value_missing_pixel != patch[x, y]):
                datax_hue.append(dictionary[:, x, y, 0])
                datax_saturation.append(dictionary[:, x, y, 1])
                datax_value.append(dictionary[:, x, y, 2])
                datay_hue.append(patch[x, y, 0])
                datay_saturation.append(patch[x, y, 1])
                datay_value.append(patch[x, y, 2])

        # Apprentissage
        self._classifier_hue.fit(datax_hue, datay_hue)
        self._classifier_saturation.fit(datax_saturation, datay_saturation)
        self._classifier_value.fit(datax_value, datay_value)

        # Prédiction
        x, y = self.patch_size // 2, self.patch_size // 2
        hue = self._classifier_hue.predict(dictionary[:, x, y, 0].reshape(1, -1))
        saturation = self._classifier_saturation.predict(dictionary[:, x, y, 1].reshape(1, -1))
        value = self._classifier_value.predict(dictionary[:, x, y, 2].reshape(1, -1))
        return np.hstack((hue, saturation, value))


if __name__ == "__main__":
    pass
