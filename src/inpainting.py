from multiprocessing import Process
from typing import List, Tuple
from copy import copy

import numpy as np
from sklearn.linear_model import Lasso
from progressbar import ProgressBar, Percentage, Counter, Timer, ETA

from src.common.decorators import time_this
from src.picture_tools.picture import Picture, VALUE_MISSING_PIXEL, get_center, flatten, unflatten, show_patch
from src.linear.cost_function import *
from src.pixel_choosing_strategy import PixelChoosingStrategy, choose_pixel


class InPainting:
    def __init__(self, patch_size: int, step: int = None, value_missing_pixel: int = VALUE_MISSING_PIXEL,
                 alpha: float = 1.0, max_iterations: int = 1000, tolerance: float = 0.0001,
                 pixel_choosing_strategy: PixelChoosingStrategy = PixelChoosingStrategy.FIRST_PIXEL):
        self.patch_size = patch_size
        self.step = patch_size if step is None else step
        self.value_missing_pixel = value_missing_pixel
        self.pixel_choosing_strategy = pixel_choosing_strategy

        # Initialize one classifier per channel
        classifiers_kwaargs = {"alpha": alpha, "copy_X": True, "fit_intercept": True, "max_iter": max_iterations,
                               "normalize": False, "positive": False, "precompute": False, "random_state": None,
                               "selection": 'cyclic', "tol": tolerance, "warm_start": False}
        self._classifier_hue = Lasso(**classifiers_kwaargs)
        self._classifier_saturation = Lasso(**classifiers_kwaargs)
        self._classifier_value = Lasso(**classifiers_kwaargs)

    def inpaint(self, picture: Picture) -> Picture:
        picture = picture.copy()

        # Initialisation de la barre de progression
        number_of_pixels_to_process = len(picture.pixels[picture.pixels == self.value_missing_pixel]) // 3
        progress_bar_widgets = ["Inpainting: processed ", Counter(), "/{} pixels [".format(number_of_pixels_to_process),
                                Percentage(), "], ", Timer(), ", ", ETA()]
        progress_bar = ProgressBar(widgets=progress_bar_widgets,
                                   minval=0,
                                   maxval=number_of_pixels_to_process)
        progress_bar.start()

        # On récupère le dictionnaire
        dictionary = picture.get_dictionnaire(self.patch_size, self.step, max_missing_pixel=0)
        dictionary_hue, dictionary_saturation, dictionary_value =\
            [dictionary[:, :, :, 0], dictionary[:, :, :, 1], dictionary[:, :, :, 2]]

        while self.value_missing_pixel in picture.pixels:
            # On récupère le patch centré sur le prochain pixel à traiter
            next_pixel = choose_pixel(pixels=picture.pixels,
                                      pixel_choosing_strategy=self.pixel_choosing_strategy,
                                      value_missing_pixel=self.value_missing_pixel)

            # On reconstruit le pixel choisit
            # TODO: parfois, patch est vide
            patch = picture.get_patch(*next_pixel, self.patch_size)
            next_pixel_value = self._get_next_pixel_value(patch, dictionary_hue, dictionary_saturation,
                                                          dictionary_value)
            picture.pixels[next_pixel] = next_pixel_value

            # On met à jour la barre de progression
            progress_bar.update(progress_bar.value + 1)

        progress_bar.finish()
        return picture

    def _get_next_pixel_value(self, patch, dictionary_hue, dictionary_saturation, dictionary_value) -> np.ndarray:
        # Construction des ensembles d'apprentissage
        datax_hue, datax_saturation, datax_value, datay_hue, datay_saturation, datay_value = [], [], [], [], [], []
        # On itère sur chaque pixel du patch à reconstruire
        for x in range(self.patch_size):
            for y in range(self.patch_size):
                # Si on tombe sur une valeur manquante, on ne l'ajoute évidemment pas (impossible à apprendre)
                if all(self.value_missing_pixel != patch[x, y]):
                    datax_hue.append(dictionary_hue[:, x, y])
                    datax_saturation.append(dictionary_saturation[:, x, y])
                    datax_value.append(dictionary_value[:, x, y])
                    datay_hue.append(patch[x, y, 0])
                    datay_saturation.append(patch[x, y, 1])
                    datay_value.append(patch[x, y, 2])

        # Apprentissage
        self._classifier_hue.fit(datax_hue, datay_hue)
        self._classifier_saturation.fit(datax_saturation, datay_saturation)
        self._classifier_value.fit(datax_value, datay_value)

        # Prédiction
        x, y = self.patch_size // 2, self.patch_size // 2
        hue = self._classifier_hue.predict(dictionary_hue[:, x, y].reshape(1, -1))
        saturation = self._classifier_saturation.predict(dictionary_saturation[:, x, y].reshape(1, -1))
        value = self._classifier_value.predict(dictionary_value[:, x, y].reshape(1, -1))
        return np.hstack((hue, saturation, value))


if __name__ == "__main__":
    pass
