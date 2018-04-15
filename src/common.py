# -*- coding: utf-8 -*-
""" Ce module contient toutes les méthodes et fonctions n'ayant pas pu être rangées dans un module précis. Il est
souvent réutilisé dans plusieurs modules différents, et doit donc être en bout de chaîne pour éviter les dépendances
cycliques.
"""

import numpy as np
from typing import Union

Number = Union[int, float]


def normalize(array: np.ndarray, floor: Number = 0, ceil: Number = 1):
    """ Normalise un tableau dans un seuil minimal et maximal.
    :param: array, le tableau à normaliser.
    :param: floor, le seuil bas de la normalisation.
    :param: ceil, le seuil haut de la normalisation.
    :return: Le tableau normalisé.
    """
    array_min, array_max = array.min(), array.max()
    upper = (ceil - floor) / (array_max - array_min)
    lower = (ceil - upper * array_max)  # Or (floor - upper * array_min)
    return upper * array + lower


if __name__ == "__main__":
    pass
