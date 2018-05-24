# -*- coding: utf-8 -*-
""" Ce module contient toutes les méthodes et fonctions nécessaires pour
pouvoir manipuler facilement des images.
"""
from typing import Tuple
import os.path
import random

import numpy as np
from matplotlib import pyplot as plt
from numpy import uint8

from src.picture_tools.codage import Codage, change_codage
from src.common.math import normalize

VALUE_MISSING_PIXEL = np.ones((3,)) * -100
VALUE_OUT_OF_BOUNDS = np.ones((3,)) * -1000
VALUE_SHOWING_MISSING_PIXEL = np.array([-1])  # np.random.uniform(low=-1, high=1)


class Picture:
    """ Permet de manipuler très facilement une image tout en supportant
    plusieurs encodages.
    Attributs :
        - picture_path : str, le chemin vers l'image.
        - codage : Codage, le codage des pixels de l'image (par défaut, HSV).
        - pixels : np.ndarray, les pixels (le contenu) de l'image. Quel que
        soit le codage de l'image, les valeurs des pixels sont normalisées
        entre -1 et 1.
        - hauteur : int, la hauteur de l'image.
        - largeur : int, la largeur de l'image.
    """

    def __init__(self, picture_path: str, codage: Codage = Codage.HSV):
        self.picture_path = picture_path
        self.codage = codage
        pixels, self.hauteur, self.largeur = _load_pixels(picture_path)
        self.pixels = change_codage(pixels, Codage.RGB, self.codage)

    def show(self, show: bool = True) -> None:
        """ Plot l'image sur matplotlib et l'affiche sur demande.
        :param: show, waut `True` si on affiche l'image après l'avoir plottée.
        """
        picture = self._get_showable_picture()
        plt.axis("off")
        plt.imshow(picture)
        if show:
            plt.show()

    def save(self, picture_path: str = None) -> None:
        """ Sauvegarde l'image sur le disque.
        Si aucun nom n'est passé en paramètre, sauvegarde l'image dans le repertoire courant avec le basename utilisé à
        l'ouverture.
        :param: picture_path, le chemin de destination.
        """
        if picture_path is None:
            picture_path = os.path.basename(self.picture_path)

        # Remove missing values
        picture = np.copy(self.pixels)
        picture[picture == VALUE_MISSING_PIXEL] = VALUE_SHOWING_MISSING_PIXEL
        picture[picture == VALUE_OUT_OF_BOUNDS] = VALUE_SHOWING_MISSING_PIXEL

        picture = change_codage(self.pixels, self.codage, Codage.RGB)
        picture = normalize(picture, 0, 255, -1, 1).astype(uint8)
        plt.imshow(picture)
        plt.savefig(picture_path)

    def copy(self):
        new_picture = Picture.__new__(Picture)
        new_picture.picture_path = self.picture_path
        new_picture.codage = self.codage
        new_picture.hauteur, new_picture.largeur = self.hauteur, self.largeur
        new_picture.pixels = np.copy(self.pixels)
        return new_picture

    def add_noise(self, threshold: float = 0.05):
        """ Ajoute aléatoirement du bruit dans l'image.
        :param: threshold, seuil en dessous duquel on bruite le pixel.
        """
        for x in range(self.hauteur):
            for y in range(self.largeur):
                self.pixels[x, y] = VALUE_MISSING_PIXEL if random.random(
                ) < threshold else self.pixels[x, y]

    def add_rectangle(self, x: int, y: int, hauteur: int, largeur: int) -> None:
        """ Ajoute aléatoirement un rectangle de bruit dans l'image.
        :param: x, Le pixel d'abscisse x du coint haut gauche du rectangle
        :param: y, Le pixel d'ordonnée y du coint haut gauche du rectangle
        :param: hauteur, La hauteur du rectangle
        :param: largeur, La largeur du rectangle
        """
        self.pixels[x:x + hauteur, y:y + largeur] = VALUE_MISSING_PIXEL

    def get_pixel(self, x: int, y: int) -> np.ndarray:
        """ Retourne le pixel de l'image aux indexes (x, y).
        :param: x, l'index de la colonne.
        :param: y, l'index de la ligne.
        :return: Le contenu du pixel demandé.
        """
        return self.pixels[x, y]

    def get_patch(self, x: int, y: int, size: int) -> np.ndarray:
        """ Retourne le patch de l'image centré aux indexes (x, y).
        :param: x, l'index de la colonne.
        :param: y, l'index de la ligne.
        :param: size, la longueur du patch.
        :return: Le contenu du patch demandé.
        """
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
        """ Retourne tous les indices des centres des patches de l'image contenant des pixels manquants.
        :return: Une list de patchs contenant des pixels manquants.
        """
        return np.array([np.array([x, y]) for x in range(self.hauteur) for y in range(self.largeur) \
                         if (self.get_pixel(x, y) == VALUE_MISSING_PIXEL).all()])

    def get_dictionnaire(self, size: int, step: int = 1, max_missing_pixel: int = 0) -> np.ndarray:
        """ Retourne tous les patches de l'image ne contenant pas de pixels manquants.
        :param: size, la taille de chaque patch.
        :param: step, la taille du pas d'itération.
        :param: max_missing_pixel, le nombre de pixels manquants à prendre en compte pour retourner ce patch.
        :return: Une list de patchs ne contenant pas de pixels manquants.
        """
        result = []
        for x in range(0, self.hauteur, step):
            for y in range(0, self.largeur, step):
                if not self.out_of_bounds_patch(x, y, size):
                    patch = self.get_patch(x, y, size)
                    if len(patch[patch == VALUE_MISSING_PIXEL]) // 3 <= max_missing_pixel:
                        result.append(patch)
        return np.array(result)

    def out_of_bounds(self, x: int, y: int) -> bool:
        """ Check if the pixel located at (x, y) is out of the bounds of the picture.
        >>> picture = Picture(LENA_COLOR_512, codage=Codage.RGB)
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
        return not (0 <= x < self.hauteur and 0 <= y < self.largeur)

    def out_of_bounds_patch(self, x: int, y: int, size: int) -> bool:
        return (x - (size // 2) < 0) or \
               (x + (size // 2) + 1 >= self.hauteur) or \
               (y - (size // 2) < 0) or \
               (y + (size // 2) + 1 >= self.largeur)

    def _get_showable_picture(self) -> np.ndarray:
        """ Return the picture in a showable format (as in a format which can be plotted by invocating `plt.imshow`on
        it.
        """
        pixels = np.copy(self.pixels)

        # Fill the missing pixels with an interpretable value
        pixels[pixels == VALUE_MISSING_PIXEL] = VALUE_SHOWING_MISSING_PIXEL
        pixels[pixels == VALUE_OUT_OF_BOUNDS] = VALUE_SHOWING_MISSING_PIXEL

        # Change the format of the picture to the RGB format (for better visibility)
        pixels = change_codage(pixels, self.codage, Codage.RGB)

        # Normalise the values of the pixel from [-1, 1] to [0, 255]
        pixels = normalize(pixels, 0, 255, -1, 1).astype(uint8)

        return pixels


def get_center(pixels: np.ndarray) -> np.ndarray:
    """ Return the pixel at the center of the pixels. """
    return pixels[(pixels.shape[0] - 1) // 2, (pixels.shape[1] - 1) // 2]


def show_patch(patch: np.ndarray, codage: Codage = Codage.HSV, show: bool = True):
    """ Plot le patch sur matplotlib et l'affiche sur demande.
    :param: patch, le patch à afficher.
    :param: codage, le codage utilisé pour le patch.
    :param: show, waut `True` si on affiche le patch après l'avoir plottée.
    """
    # Remove missing values
    new_patch = np.copy(patch)
    new_patch[patch == VALUE_MISSING_PIXEL] = VALUE_SHOWING_MISSING_PIXEL
    new_patch[patch == VALUE_OUT_OF_BOUNDS] = VALUE_SHOWING_MISSING_PIXEL

    new_patch = change_codage(new_patch, codage, Codage.RGB)
    new_patch = normalize(new_patch, 0, 255, -1, 1).astype(uint8)
    plt.imshow(new_patch)
    if show:
        plt.show()


def flatten(patch: np.ndarray) -> np.ndarray:
    """ Convertit un patch en un vecteur. """
    return np.copy(patch).reshape(-1)


def unflatten(vector: np.ndarray, size_patch: int) -> np.ndarray:
    """ Convertit un vecteur en un patch. """
    return np.copy(vector).reshape(size_patch, size_patch, 3)


def out_of_bounds(pixels: np.ndarray, x: int, y: int) -> bool:
    """ Check if the pixel located at (x, y) is out of the bounds of the picture. """
    return not (0 <= x < pixels.shape[0] and 0 <= y < pixels.shape[1])


def out_of_bounds_patch(pixels: np.ndarray, x: int, y: int, size: int) -> bool:
    return (x - (size // 2) <= 0) or \
           (x + (size // 2) + 1 < pixels.shape[0]) or \
           (y - (size // 2) <= 0) or \
           (y + (size // 2) + 1 < pixels.shape[1])


def get_patch(pixels: np.ndarray, x: int, y: int, size: int,
              value_out_of_bounds: np.ndarray = VALUE_OUT_OF_BOUNDS) -> np.ndarray:
    """ Retourne le patch de l'image centré aux indexes (x, y).
    :param: x, l'index de la colonne.
    :param: y, l'index de la ligne.
    :param: size, la longueur du patch.
    :return: Le contenu du patch demandé.
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
    for index_y in range(y - (size // 2), y + (size // 2) + 1):
        for index_x in range(x - (size // 2), x + (size // 2) + 1):
            yield index_x, index_y


def iter_patch_empty(pixels: np.ndarray, x: int, y: int, size: int):
    for index_x, index_y in iter_patch(x, y, size):
        if not out_of_bounds(pixels, index_x, index_y):
            if all(pixels[index_x, index_y] == VALUE_MISSING_PIXEL):
                yield index_x, index_y


def _load_pixels(picture_path: str) -> Tuple[np.ndarray, int, int]:
    """ Charge les pixels d'une image depuis le disque et retourne son contenu.
    Si l'image est en couleur, retourne ses pixels au format RGB en ignorant le canal alpha.
    Si l'image est en noir et blanc, retourne ses pixels de la même intensité dans chaque couleur (elle sera donc
    toujours en noir et blanc, mais au format RGB).
    :param: picture_path, Le chemin contenant l'image à charger.
    :return: Un tuple contenant les pixels de l'image, sa hauteur et sa largeur.
    """
    picture = plt.imread(picture_path)

    if len(picture.shape) == 3:
        # L'image comporte trois composantes : RGB
        # Elle est donc en couleurs.
        # Elle peut contenir une composante Alpha, mais elle sera ignorée.
        pixels = picture[:, :, :3]
    else:
        # L'image comporte une seule composante : l'intensité des pixels.
        # Elle est donc en noir et blanc. On la convertie au format RGB.
        pixels = np.dstack([picture] * 3)

    image_hauteur, image_largeur, _ = pixels.shape

    # Normalisation des pixels dans l'intervalle -1, 1
    pixels = normalize(pixels, -1, 1, 0, 255)

    return pixels, image_hauteur, image_largeur


if __name__ == "__main__":
    pass
