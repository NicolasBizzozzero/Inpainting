from typing import Tuple

import numpy as np

from sklearn.linear_model import Lasso
from progressbar import ProgressBar, Percentage, Counter, Timer, ETA

from src.common.math import Number
from src.picture_tools.picture import Picture, VALUE_MISSING_PIXEL, VALUE_OUT_OF_BOUNDS, get_center, flatten, \
    unflatten, show_patch, get_patch, iter_patch, iter_patch_empty


class InPainting:
    def __init__(self, patch_size: int, step: int = None, alpha: Number = 1.0, max_iterations: int = 1e+3,
                 tolerance: Number = 1e-4, value_missing_pixel: np.ndarray = VALUE_MISSING_PIXEL,
                 value_out_of_bounds: np.ndarray = VALUE_OUT_OF_BOUNDS):
        self.patch_size = patch_size
        self.step = patch_size if step is None else step
        self.value_missing_pixel = value_missing_pixel
        self.value_out_of_bounds = value_out_of_bounds

        # Initialize one classifier per channel
        classifiers_kwaargs = {"alpha": alpha, "copy_X": True, "fit_intercept": True, "max_iter": max_iterations,
                               "normalize": False, "positive": False, "precompute": False, "random_state": None,
                               "selection": 'cyclic', "tol": tolerance, "warm_start": False}
        self._classifier_hue = Lasso(**classifiers_kwaargs)
        self._classifier_saturation = Lasso(**classifiers_kwaargs)
        self._classifier_value = Lasso(**classifiers_kwaargs)

    def inpaint(self, picture: Picture) -> Picture:
        picture = picture.copy()

        # Init progressbar
        number_of_pixels_to_process = len(picture.pixels[picture.pixels == self.value_missing_pixel]) // 3
        progress_bar_widgets = ["Inpainting: processed ", Counter(), "/{} pixels [".format(number_of_pixels_to_process),
                                Percentage(), "], ", Timer(), ", ", ETA()]

        # Retrieve the picture's dictionary
        dictionary = picture.get_dictionary(self.patch_size, self.step, max_missing_pixel=0)

        with ProgressBar(widgets=progress_bar_widgets, minval=0, maxval=number_of_pixels_to_process) as progress_bar:
            while self.value_missing_pixel in picture.pixels:
                # Retrieve the next patch containing missing values
                next_pixel = self._get_next_patch(picture, self.patch_size, self.value_out_of_bounds,
                                                  self.value_missing_pixel)

                # Inpaint the selected patch
                next_patch = picture.get_patch(*next_pixel, self.patch_size)

                self.fit(dictionary, next_patch)
                for x, y in iter_patch_empty(picture.pixels, *next_pixel, self.patch_size):
                    next_pixel_value = self.predict(x - next_pixel[0] + (self.patch_size // 2),
                                                    y - next_pixel[1] + (self.patch_size // 2),
                                                    dictionary)
                    picture.pixels[x, y] = next_pixel_value
                    progress_bar.update(progress_bar.value + 1)
            return picture

    def fit(self, dictionary, patch):
        datax_hue, datax_saturation, datax_value, datay_hue, datay_saturation, datay_value = \
            self._preprocess_training_data(patch, dictionary)

        self._classifier_hue.fit(datax_hue, datay_hue)
        self._classifier_saturation.fit(datax_saturation, datay_saturation)
        self._classifier_value.fit(datax_value, datay_value)

    def predict(self, x, y, dictionary):
        hue = self._classifier_hue.predict(dictionary[:, x, y, 0].reshape(1, -1))
        saturation = self._classifier_saturation.predict(dictionary[:, x, y, 1].reshape(1, -1))
        value = self._classifier_value.predict(dictionary[:, x, y, 2].reshape(1, -1))
        return np.hstack((hue, saturation, value))

    def _get_next_patch(self, picture: Picture, size: int, value_out_of_bounds: np.ndarray = VALUE_OUT_OF_BOUNDS,
                        value_missing_pixel: np.ndarray = VALUE_MISSING_PIXEL) -> Tuple[int, int]:
        # TODO: Implement this heuristic
        # pixels_confidence = _get_pixels_confidence(picture.pixels, value_missing_pixel)
        # patches_priorities = {(x, y): patch_priority(picture.pixels, pixels_confidence, x, y, size,
        #                                              value_out_of_bounds, value_missing_pixel) \
        #                                              for (x, y) in picture.get_patches()}
        # return max(patches_priorities.keys(), key=lambda k: patches_priorities[k])

        missing_pixels_x, missing_pixels_y, *_ = np.where(picture.pixels == self.value_missing_pixel)
        return zip(missing_pixels_x, missing_pixels_y).__next__()

    def _preprocess_training_data(self, patch, dictionary):
        datax_hue, datax_saturation, datax_value, datay_hue, datay_saturation, datay_value = [], [], [], [], [], []

        # Iterate trough each pixels of the patch to inpaint
        for x in range(self.patch_size):
            for y in range(self.patch_size):
                # Ignore missing-pixels in the patch, we cannot learn from them
                if np.all(patch[x, y] != self.value_missing_pixel) and np.all(patch[x, y] != self.value_out_of_bounds):
                    datax_hue.append(dictionary[:, x, y, 0])
                    datax_saturation.append(dictionary[:, x, y, 1])
                    datax_value.append(dictionary[:, x, y, 2])
                    datay_hue.append(patch[x, y, 0])
                    datay_saturation.append(patch[x, y, 1])
                    datay_value.append(patch[x, y, 2])

        return np.array(datax_hue), np.array(datax_saturation), \
            np.array(datax_value), np.array(datay_hue), \
            np.array(datay_saturation), np.array(datay_value)


def patch_priority(pixels: np.ndarray, pixels_confidence: np.ndarray, x: int, y: int, size: int,
                   value_out_of_bounds: np.ndarray = VALUE_OUT_OF_BOUNDS,
                   value_missing_pixel: np.ndarray = VALUE_MISSING_PIXEL) -> float:
    confidence = _get_patch_confidence(pixels, pixels_confidence, x, y, size, value_out_of_bounds, value_missing_pixel)
    return confidence


def _get_pixels_confidence(pixels: np.ndarray, value_missing_pixels: np.ndarray = VALUE_MISSING_PIXEL) -> np.ndarray:
    confidence = np.copy(pixels)
    confidence[confidence != value_missing_pixels] = 1
    confidence[confidence == value_missing_pixels] = 0
    return confidence[:, :, 0]


def _get_patch_confidence(pixels: np.ndarray, confidence: np.ndarray, x: int, y: int, size: int,
                          value_out_of_bounds: np.ndarray = VALUE_OUT_OF_BOUNDS,
                          value_missing_pixel: np.ndarray = VALUE_MISSING_PIXEL) -> float:
    patch_pixels = get_patch(pixels, x, y, size, value_out_of_bounds)
    patch_confidence = get_patch(confidence, x, y, size, np.nan)
    return np.nansum(patch_confidence[(patch_pixels != value_missing_pixel)[:, :, 0]]) / \
           np.count_nonzero(~np.isnan(patch_confidence))


if __name__ == "__main__":
    pass
