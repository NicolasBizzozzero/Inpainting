# -*- coding: utf-8 -*-
""" Ce module contient toutes les fonctions nécessaires pour appliquer une
descente de gradient.
"""

import enum

import numpy as np


@enum.unique
class DescenteDeGradient(enum.IntEnum):
    STOCHASTIQUE = 0
    MINI_BATCH = 1
    BATCH = 2


class UnknownDescenteDeGradient(Exception):
    def __init__(self, descente: str):
        Exception.__init__(self, "La descente de gradient : \"{descente}\" "
                                 "n'existe pas".format(descente=descente))


def gradient_descent(datax, datay, type_descente, w_init, loss_g,
                     max_iter=10000, pas=0.01, batch_size=None,
                     **kwaargs):
    """ Effectue une descente de gradient pour optimiser la valeur d'un
    vecteur de poids d'un perceptron.
    :param: datax, L'ensemble de données d'apprentissage.
    :param: datay, L'ensemble des labels des données d'apprentissage.
    :param: type_descente, Le type de descente de gradient utilisé.
    :param: w_init, La valeur initiale de la fonction à optimiser.
    :param: loss_g, La fonction retournant le gradient du coût. Elle doit
    pouvoir prendre en paramètre (datax, datay, w).
    :param: max_iter, Le seuil d'itération maximal.
    :param: pas, Le pas d'apprentissage.
    :param: batch_size, La taille d'un sous-ensemble d'exemples tiré
    aléatoirement pour le type de descente `MINI_BATCH`.
    :return: w, La valeur optimale de la fonction.
    """
    if type_descente == DescenteDeGradient.BATCH:
        return batch(datax, datay, w_init, max_iter, pas, loss_g, **kwaargs)
    elif type_descente == DescenteDeGradient.MINI_BATCH:
        return mini_batch(datax, datay, w_init, max_iter, pas, loss_g,
                          batch_size)
    elif type_descente == DescenteDeGradient.STOCHASTIQUE:
        return stochastic(datax, datay, w_init, max_iter, pas, loss_g)
    else:
        raise UnknownDescenteDeGradient(type_descente)


def batch(datax, datay, w_init, max_iter, pas, loss_g, **kwaargs):
    """ Effectue une descente de gradient de type batch (par lot).
    Une descente de gradient de type batch calcule, à chaque itération, le
    gradient sur toutes les données.
    """
    w = w_init
    for _ in range(max_iter):
        w = w - pas * loss_g(datax, datay, w, **kwaargs)
    return w


def mini_batch(datax, datay, w_init, max_iter, pas,
               loss_g, batch_size):
    """ Effectue une descente de gradient de type mini-batch (par mini-lot).
    Une descente de gradient de type mini-batch calcule, à chaque itération,
    le gradient sur un sous-ensemble des données tiré aléatoirement.
    """
    w = w_init
    for _ in range(max_iter):
        # Choix de `batch_size` exemples aléatoirement
        indexes = list(range(len(datay)))
        np.random.shuffle(indexes)
        indexes = indexes[:batch_size]
        datax_bis, datay_bis = datax[indexes], datay[indexes]

        w = w - pas * loss_g(datax_bis, datay_bis, w)
    return w


def stochastic(datax, datay, w_init, max_iter, pas, loss_g):
    """ Effectue une descente de gradient de type stochastique.
    Une descente de gradient de type stochastique calcule, à chaque
    itération, le gradient sur un singleton des données tiré aléatoirement.
    """
    w = w_init
    for _ in range(max_iter):
        # Choix d'un exemple aléatoirement
        index = np.random.randint(len(datay))
        datax_bis, datay_bis = datax[index], datay[index]

        w = w - pas * loss_g(datax_bis, datay_bis, w)
    return w


def gradient_descent_h(datax, datay, w_init, loss, loss_g, max_iter=10000,
                       pas=0.01):
    """ Effectue une descente de gradient pour optimiser la valeur d'un
    vecteur de poids d'un perceptron.
    Cette fonction garde en mémoire l'historique de tous les calculs
    effectués et les retourne. Elle permet ainsi une visualisation de la
    progression de l'optimisation.
    :param: datax, L'ensemble de données d'apprentissage.
    :param: datay, L'ensemble des labels des données d'apprentissage.
    :param: w_init, La valeur initiale de la fonction à optimiser.
    :param: loss, La fonction retournant le  coût. Elle doit pouvoir prendre
    en paramètre (datax, datay, w).
    :param: loss_g, La fonction retournant le gradient du coût. Elle doit
    pouvoir prendre en paramètre (datax, datay, w).
    :param: max_iter, Le seuil d'itération maximal.
    :param: pas, Le pas d'apprentissage.
    :return: (w_histo, f_histo, grad_histo), L'historique des valeurs
    calculées.
    """
    # Initialisation des historiques
    w_histo, f_histo, grad_histo = w_init,\
        np.array([loss(datax, datay, w_init)]),\
        np.array([loss_g(datax, datay, w_init)])

    for _ in range(max_iter):
        # Calcul du gradient
        w = w_histo[-1] - (pas * grad_histo[-1])

        # Mise à jour des historiques
        w_histo = np.vstack((w_histo, w))
        f_histo = np.vstack((f_histo, loss(datax, datay, w)))
        grad_histo = np.vstack((grad_histo, loss_g(datax, datay, w)))

    return w_histo, f_histo, grad_histo


def optimize(dfonc, x_init, pas=0.001, max_iter=1000):
    """ Effectue une descente de gradient de type batch (par lot) pour
    optimiser la valeur d'une fonction.
    :param: dfonc, La fonction retournant le gradient de la fonction à
    optimiser.
    :param: x_init, La valeur initiale de la fonction à optimiser.
    :param: pas, Le pas d'apprentissage.
    :param: max_iter, Le seuil d'itération maximal.
    :return: (x_histo, f_histo, grad_histo), L'historique des valeurs
    calculées.
    """
    x = x_init
    for _ in range(max_iter):
        x = x - (pas * dfonc(*x.T))
    return x


def optimize_h(fonc, dfonc, x_init, pas=0.001, max_iter=1000):
    """ Effectue une descente de gradient de type batch (par lot) pour
    optimiser la valeur d'une fonction.
    Cette fonction garde en mémoire l'historique de tous les calculs
    effectués et les retourne. Elle permet ainsi une visualisation de la
    progression de l'optimisation.
    :param: fonc, La fonction à optimiser.
    :param: dfonc, La fonction retournant le gradient de la fonction à
    optimiser.
    :param: x_init, La valeur initiale de la fonction à optimiser.
    :param: pas, Le pas d'apprentissage.
    :param: max_iter, Le seuil d'itération maximal.
    :return: (x_histo, f_histo, grad_histo), L'historique des valeurs
    calculées.
    """
    # Initialisation des historiques
    x_histo, f_histo, grad_histo = np.array([x_init]),\
        np.array([fonc(*x_init)]),\
        np.array([np.array(dfonc(*x_init))])

    for _ in range(max_iter):
        # Calcul du gradient
        x = x_histo[-1] - (pas * grad_histo[-1])

        # Mise à jour des historiques
        x_histo = np.vstack((x_histo, x))
        f_histo = np.vstack((f_histo, fonc(*x.T)))
        grad_histo = np.vstack((grad_histo, np.array([dfonc(*x.T)])))

    return x_histo, f_histo, grad_histo


if __name__ == '__main__':
    pass
