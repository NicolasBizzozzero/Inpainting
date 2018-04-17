# -*- coding: utf-8 -*-
""" Ce module contient multitudes de fonctions de projection, permettant
d'envoyer les données dans un espace de dimension plus grande.
Cette projection permet alors de trouver un frontière linéairement séparable
dans ce plus grand espace.
"""

import numpy as np

from data_generation import make_grid


def identite(datax):
    """ Retourne les données en elles-mêmes. Sert de projection par défaut. """
    return datax


def polynomiale(datax):
    """ Projette les données dans un espace de dimension polynomial (vu en TD).
    ATTENTION : Cette projection ne fonctionne que pour des données en 2D.
    # TODO: Possibilité de généraliser aux données en ND. Le faire.
    """
    x1, x2 = datax[:, 0], datax[:, 1]
    return np.column_stack((x1, x2, x1 * x2, np.power(x1, 2), np.power(x2, 2)))


def gaussienne(datax, mu=0, sigma=1):
    # TODO: fixme
    return np.array([[np.exp(-(np.linalg.norm(datax[i] - mu) ** 2) / (2 * sigma ** 2))] for i in range(len(datax))])


if __name__ == '__main__':
    pass
