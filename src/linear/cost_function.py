# -*- coding: utf-8 -*-
""" Ce module contient mulitudes de fonctions de coût, accompagnées de leur
gradient. Elles sont utilisées durant l'apprentissage du Perceptron en
effectuant une descente de gradient.
Chaque fonction doit être accompagnée de son gradient, et ce dernier doit avoir
le même nom que celle-ci, suffixé par `_g`.
De plus, chaque coût et son gradient doivent être décorés par
`_decorateur_vec`, leur permettant d'être utilisée quelle que soit les
dimensions d'entrées.
"""

import numpy as np

from src.common import norme_1, norme_2


def _decorator_vec(fonc):
    def vecfonc(datax, datay, w, *args, **kwargs):
        if not hasattr(datay, "__len__"):
            datay = np.array([datay])
        datax, datay, w = datax.reshape(
            len(datay), -1), datay.reshape(-1, 1), w.reshape((1, -1))
        return fonc(datax, datay, w, *args, **kwargs)
    return vecfonc


@_decorator_vec
def mse(datax, datay, w):
    """ Retourne la moyenne de l'erreur aux moindres carres """
    prediction = np.dot(datax, w.T)
    real_labels = datay
    squared_error = np.power(prediction - real_labels, 2)
    return np.mean(squared_error) * (1 / 2)  # Pour dériver plus facilement


@_decorator_vec
def mse_g(datax, datay, w):
    """ Retourne le gradient de l'erreur au moindres carres """
    prediction = np.dot(datax, w.T)
    real_labels = datay
    gradient_squared_error = (prediction - real_labels) * datax
    return np.mean(gradient_squared_error)


@_decorator_vec
def l1(datax, datay, w, alpha):
    """ Retourne l'erreur de la régularisation L1. """
    return mse(datax, datay, w) + (alpha * norme_1(w))


@_decorator_vec
def l1_g(datax, datay, w, alpha):
    """ Retourne le gradient de l'erreur de la régularisation L1. """
    return mse_g(datax, datay, w) + (alpha * np.sign(w))

@_decorator_vec
def l2(datax, datay, w, alpha):
    """ Retourne l'erreur de la régularisation L2. """
    return mse(datax, datay, w) + (alpha * norme_2(w))


@_decorator_vec
def l2_g(datax, datay, w, alpha):
    """ Retourne le gradient de l'erreur de la régularisation L2. """
    return mse(datax, datay, w) + (2 * alpha * w)


@_decorator_vec
def hinge(datax, datay, w, alpha=0):
    """ Retourne la moyenne de l'erreur hinge """
    prediction = np.dot(datax, w.T)
    real_labels = datay
    hinge_loss = np.maximum(0, alpha - real_labels * prediction)
    return np.mean(hinge_loss)


@_decorator_vec
def hinge_g(datax, datay, w, alpha=0, activation=np.sign):
    """ Retourne le gradient de l'erreur hinge """
    cost = -activation(hinge(datax, datay, w, alpha)) * datax * datay
    return (np.sum(cost, axis=0) / len(datax))  # Normalisation


@_decorator_vec
def hinge_penality(datax, datay, w, alpha=0, lbda=1):
    return hinge(datax, datay, w, alpha=0) + \
        lbda * np.power(np.linalg.norm(w), 2)


@_decorator_vec
def hinge_penality_g(datax, datay, w, alpha=0, lbda=1, activation=np.sign):
    cost = -activation(hinge_penality(datax, datay, w, alpha, lbda)) * datax * datay + \
        2 * lbda * w
    return (np.sum(cost, axis=0) / len(datax))  # Normalisation


if __name__ == '__main__':
    pass
