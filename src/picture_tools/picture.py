# -*- coding: utf-8 -*-
""" Ce module contient toutes les méthodes et fonctions nécessaires pour
pouvoir manipuler facilement des images.
"""
from typing import Tuple
import os.path
import random

import matplotlib.pyplot as plt
import numpy as np
from numpy import uint8

from src.picture_tools.codage import Codage, change_codage
from src.common import normalize


VALUE_MISSING_PIXEL = -100


class Picture:
    """ Permet de manipuler très facilement une image tout en supportant plusieurs encodages.
    Attributs :
        - picture_path : str, le chemin vers l'image.
        - codage : Codage, le codage des pixels de l'image (par défaut, HSV).
        - pixels : np.ndarray, les pixels (le contenu) de l'image. Quel que soit le codage de l'image, les valeurs des
        pixels sont normalisées entre -1 et 1.
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
        picture = change_codage(self.pixels, self.codage, Codage.RGB)
        print("to RGB", picture.shape, picture.min(), picture.max(), picture.argmin(), picture.argmax())
        picture = normalize(picture, 0, 255).astype(uint8)
        print("normalized", picture.shape, picture.min(), picture.max(), picture.argmin(), picture.argmax())
        print(picture[0, 0])
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
        picture = change_codage(self.pixels, self.codage, Codage.RGB)
        picture = normalize(picture, 0, 255)
        plt.imshow(picture)
        plt.savefig(picture_path)

    def add_noise(self, threshold: float = 0.2):
        """ Ajoute aléatoirement du bruit dans l'image.
        :param: threshold, seuil en dessous duquel on bruite le pixel.
        """
        for x in range(self.largeur):
            for y in range(self.hauteur):
                self.pixels[x, y] = np.random.randint(low=-1, high=1, size=(3,)) \
                    if random.random() < threshold else self.pixels[x, y]


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
    print(pixels[0, 0], pixels.dtype)
    pixels = normalize(pixels, -1, 1)

    return pixels, image_hauteur, image_largeur


if __name__ == "__main__":
    pass
