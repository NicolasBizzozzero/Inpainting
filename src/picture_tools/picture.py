# -*- coding: utf-8 -*-
""" Ce module contient toutes les méthodes et fonctions nécessaires pour
pouvoir manipuler facilement des images.
"""
from typing import Tuple
import os.path

import matplotlib.pyplot as plt
import numpy as np

from src.picture_tools.codage import Codage, change_codage


VALUE_MISSING_PIXEL = -100


class Picture:
    """ Permet de manipuler très facilement une image tout en supportant plusieurs encodages.
    Attributs :
        - picture_path : str, le chemin vers l'image.
        - codage : Codage, le codage des pixels de l'image (par défaut, HSV).
        - pixels : np.ndarray, les pixels (le contenu) de l'image.
        - hauteur : int, la hauteur de l'image.
        - largeur : int, la largeur de l'image.
    """
    def __init__(self, picture_path: str, codage: Codage = Codage.HSV):
        self.picture_path = picture_path
        self.codage = codage
        pixels, self.hauteur, self.largeur = _load_pixels(picture_path)
        self.pixels = change_codage(pixels, Codage.RGB, self.codage)

    def show(self, show=True) -> None:
        """ Plot l'image sur matplotlib et l'affiche sur demande.
        :param: show, waut `True` si on affiche l'image après l'avoir plottée.
        """
        picture = _pixels_to_picture(self.pixels, self.hauteur, self.largeur)
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
        image = _pixels_to_picture(self.pixels, self.hauteur, self.largeur)
        plt.imshow(image)
        plt.savefig(picture_path)


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
        # L'image comporte trois composantes : RGBK
        # Elle est donc en couleurs.K
        # Elle peut contenir une composante Alpha, mais elle sera ignorée.
        picture = picture[:, :, :3]
    else:
        # L'image comporte une seule composante : l'intensité des pixels.
        # Elle est donc en noir et blanc. On la converti au format RGB.
        picture = np.dstack([picture] * 3)

    # Transformation en matrice `n * 3`, avec `n` le nombre de pixels
    image_hauteur, image_largeur, _ = picture.shape
    pixels = _picture_to_pixels(picture, image_hauteur, image_largeur)

    return pixels, image_hauteur, image_largeur


def _picture_to_pixels(picture: np.ndarray, image_hauteur: int, image_largeur: int) -> np.ndarray:
    """ Récupère les pixels d'une image dans un format plus facilement utilisable.
    :param: image, l'image chargée par matplotlib.
    :param: image_hauteur, la hauteur de l'image.
    :param: image_largeur, la largeur de l'image.
    """
    return picture.reshape((image_hauteur * image_largeur, 3))


def _pixels_to_picture(pixels: np.ndarray, image_hauteur: int, image_largeur: int) -> np.ndarray:
    """ Récupère les pixels d'une image dans un format plus facilement utilisable.
    :param: pixels, Les pixels de l'image chargée par matplotlib.
    :param: image_hauteur, la hauteur de l'image.
    :param: image_largeur, la largeur de l'image.
    """
    return pixels.reshape((image_hauteur, image_largeur , 3))


if __name__ == "__main__":
    pass
