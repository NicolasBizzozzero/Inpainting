# -*- coding: utf-8 -*-
""" Ce module contient toutes les méthodes et fonctions nécessaires pour
pouvoir initaliser des vecteurs et des matrices.
"""

import numpy as np


def init_zeros(lines, columns):
    return np.zeros((lines, columns))


def init_random(lines, columns):
    return np.random.random((lines, columns))


def init_grid(im_h, im_l, dimension=2):
    X, Y = np.meshgrid(np.arange(0, im_l, 1), np.arange(0, im_h, 1))
    return np.array([X.flatten(), Y.flatten()]).T


if __name__ == '__main__':
    pass