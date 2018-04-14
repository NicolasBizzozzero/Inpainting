# -*- coding: utf-8 -*-
""" Ce module contient toutes les méthodes et fonctions nécessaires pour
pouvoir utiliser des mesures de distance
"""

import numpy as np


def euclidean(v1, v2):
    return np.linalg.norm(v1 - v2)


if __name__ == '__main__':
    pass
