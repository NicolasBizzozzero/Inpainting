import numpy as np
from numpy import uint8
import matplotlib.pyplot as plt

from src.picture_tools.codage import Codage, change_codage
from src.picture_tools.picture import VALUE_MISSING_PIXEL
from src.common import normalize


def get_center(pixels: np.ndarray) -> np.ndarray:
    """ Return the pixel at the center of the pixels. """
    return pixels[(pixels.shape[0] - 1) // 2, (pixels.shape[1] - 1) // 2]


def show_patch(patch: np.ndarray, codage: Codage = Codage.RGB, show: bool = True):
    """ Plot le patch sur matplotlib et l'affiche sur demande.
    :param: patch, le patch à afficher.
    :param: codage, le codage utilisé pour le patch.
    :param: show, waut `True` si on affiche le patch après l'avoir plottée.
    """
    # Remove missing values
    new_patch = np.copy(patch)
    new_patch[patch == VALUE_MISSING_PIXEL] = np.random.uniform(low=-1, high=1)

    new_patch = change_codage(new_patch, codage, Codage.RGB)
    new_patch = normalize(new_patch, 0, 255, -1, 1).astype(uint8)
    plt.imshow(new_patch)
    if show:
        plt.show()


def flatten(patch: np.ndarray) -> np.ndarray:
    """ Flatten le patch! """
    return patch.flatten(order='C')



if __name__ == "__main__":
    pass
