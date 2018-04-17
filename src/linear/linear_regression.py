# -*- coding: utf-8 -*-
""" Ce module contient toutes les fonctions nécessaires pour utiliser
une régression linéaire.
"""

import enum

import numpy as np

from src.linear.cost_function import hinge, hinge_g, hinge_penality, hinge_penality_g, mse, mse_g
from src.linear.projection import gaussienne, identite, polynomiale
from src.linear.gradient_descent import DescenteDeGradient, gradient_descent, gradient_descent_h


@enum.unique
class Initialisation(enum.IntEnum):
    ZERO = 0
    RANDOM = 1


class UnknownInitialisation(Exception):
    def __init__(self, initialisation: str):
        Exception.__init__(self, "L'initialisation : \"{initialisation}\" "
                                 "n'existe pas"
                                 "".format(initialisation=initialisation))


class LinearRegression:
    def __init__(self, loss=hinge, loss_g=hinge_g, max_iter=10000, eps=0.01, biais=True, activation=np.sign,
                 type_descente=DescenteDeGradient.MINI_BATCH, taille_batch=50, initialisation=Initialisation.RANDOM,
                 projection=identite):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
            :biais: Utilise-t'on un biais ?
            :activation: Fonction d'activation utilisée.
            :type_descente: Le type de descente de gradient utilisé pour
                            l'apprentissage.
            :taille_batch: Taille d'un lot d'exemple prit à chaque itération
                           pour la descente de gradient en mini-batch.
            :initialisation: Stratégie de l'initialisation du vecteur de
                             poids.
            :projection: Fonction de projection des données dans un espace de
                         dimension supérieure.
        """
        self.max_iter = max_iter
        self.eps = eps
        self.biais = biais
        self.activation = activation
        self.loss = loss
        self.loss_g = loss_g
        self.type_descente = type_descente
        self.taille_batch = taille_batch
        self.initialisation = initialisation
        self.projection = projection

    def fit(self, datax, datay):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        datay = datay.reshape(-1, 1)
        datax = datax.reshape(len(datay), -1)

        # Ajout de la projection
        datax = self.projection(datax)

        # Ajout du biais
        if self.biais:
            datax = np.c_[datax, np.ones(len(datax))]

        # Initialisation du vecteur de poids
        self._init_w(taille=datax.shape[1])

        # Optimisation du vecteur de poids
        self.w = gradient_descent(datax, datay,
                                  type_descente=self.type_descente,
                                  w_init=self.w,
                                  loss_g=self.loss_g,
                                  max_iter=self.max_iter,
                                  pas=self.eps,
                                  batch_size=self.taille_batch)

    def fit_h(self, datax, datay):
        """ Apprends le perceptron tout en gardant en mémoire l'historique de
        la descente de gradient effectuée.
        Retourne le tuple (w_histo, f_histo, grad_histo)
        """
        datay = datay.reshape(-1, 1)
        datax = datax.reshape(len(datay), -1)

        # Ajout de la projection
        datax = self.projection(datax)

        # Ajout du biais
        if self.biais:
            datax = np.c_[datax, np.ones(len(datax))]

        # Initialisation du vecteur de poids
        self._init_w(taille=datax.shape[1])

        # Optimisation du vecteur de poids
        w_histo, f_histo, grad_histo = gradient_descent_h(datax, datay,
                                                          w_init=self.w,
                                                          loss=self.loss,
                                                          loss_g=self.loss_g,
                                                          max_iter=self.max_iter,
                                                          pas=self.eps)
        self.w = w_histo[-1]
        return w_histo, f_histo, grad_histo

    def predict(self, datax):
        if len(datax.shape) == 1:
            datax = datax.reshape(1, -1)

        # Ajout de la projection
        datax = self.projection(datax)

        # Ajout du biais
        if self.biais:
            datax = np.c_[datax, np.ones(len(datax))]

        return self.activation(np.dot(datax, self.w.T))

    def score(self, datax, datay):
        return np.array([np.equal(self.predict(x), y)
                         for x, y in zip(datax, datay)]).sum() / len(datax)

    def _init_w(self, taille):
        """ Initialise le vecteur w selon une stratégie spécifique. """
        if self.initialisation == Initialisation.ZERO:
            self.w = np.zeros((1, taille))
        elif self.initialisation == Initialisation.RANDOM:
            self.w = np.random.random((1, taille))
        else:
            raise UnknownInitialisation(self.initialisation)


if __name__ == '__main__':
    pass
