from typing import List, Tuple
from copy import copy

import numpy as np
from sklearn.linear_model import Lasso
from progressbar import ProgressBar, Percentage, Counter, Timer

from src.common import time_this
from src.picture_tools.picture import Picture, VALUE_MISSING_PIXEL, get_center, flatten, unflatten, show_patch
from src.linear.cost_function import *
from src.linear.gradient_descent import DescenteDeGradient
from src.linear.linear_regression import Initialisation, LinearRegression


PB_WIDGETS = ["Inpainting: processed ", Counter(), " pixels [", Percentage(), "], ",
              Timer()]


class InPainting:
    def __init__(self, patch_size: int, step: int = None, max_missing_pixel: int = 0,
                 value_missing_pixel: int = VALUE_MISSING_PIXEL, loss: callable = l1, loss_g: callable = l1_g,
                 max_iter: int = 10000, eps: float = 0.01, biais: bool = True,
                 type_descente: DescenteDeGradient = DescenteDeGradient.BATCH, taille_batch: int = 50,
                 initialisation: Initialisation = Initialisation.RANDOM, alpha: float = 0.001):
        self.patch_size = patch_size
        self.step = patch_size if step is None else step
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
        # Initialisation de la barre de progression
        progress_bar = ProgressBar(widgets=PB_WIDGETS,
                                   maxval=len(picture.pixels[picture.pixels == self.value_missing_pixel]) // 3,
                                   minval=0)
        progress_bar.start()

        # On récupère le dictionnaire
        dictionary = picture.get_dictionnaire(self.patch_size, self.step, self.max_missing_pixel)

        while self.value_missing_pixel in picture.pixels:
            # On récupère le patch centré sur le prochain pixel à traiter
            next_pixel = self._get_next_pixel(picture.pixels)

            # On reconstruit le pixel choisit
            show_patch(picture.get_patch(*next_pixel, self.patch_size))
            next_pixel_value = self._get_next_pixel_value(picture, dictionary, *next_pixel)
            picture.pixels[next_pixel] = next_pixel_value

            # On met à jour la barre de progression
            progress_bar.update(progress_bar.value + 1)

        progress_bar.finish()

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
        classifier_hue = Lasso()
        classifier_saturation = Lasso()
        classifier_value = Lasso()
        classifier_hue.fit(datax_hue, datay_hue)
        classifier_saturation.fit(datax_saturation, datay_saturation)
        classifier_value.fit(datax_value, datay_value)

        # Prédiction
        x, y = self.patch_size // 2, self.patch_size // 2
        hue = classifier_hue.predict(dictionary[:, x, y, 0].reshape(1, -1))
        saturation = classifier_saturation.predict(dictionary[:, x, y, 1].reshape(1, -1))
        value = classifier_value.predict(dictionary[:, x, y, 2].reshape(1, -1))
        return np.hstack((hue, saturation, value))


if __name__ == "__main__":
    pass
