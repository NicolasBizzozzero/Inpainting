# -*- coding: utf-8 -*-
""" Ce module contient toutes les méthodes et fonctions nécessaires pour
pouvoir manipuler des images.
"""

import matplotlib.pyplot as plt



def load_picture(picture_path):
    # Chargement des 3 premières composantes de l'image (la 4ème étant la
    # composante alpha, la transparence).
    image = plt.imread(picture_path)[:, :, :3]
    im_h, im_l, _ = image.shape

    # Transformation en matrice `n * 3`, avec `n` le nombre de pixels
    pixels = _picture_to_pixels(image, im_h, im_l)

    return pixels, im_h, im_l


def show_picture(pixels, im_h, im_l, show=True):
    plt.imshow(pixels.reshape((im_h, im_l , 3)))
    if show:
        plt.show()


def save_picture(pixels, im_h, im_l, picture_path):
    image = _pixels_to_picture(pixels, im_h, im_l)
    plt.imshow(image)
    plt.savefig(picture_path)


def _pixels_to_picture(pixels, im_h, im_l):
    return pixels.reshape((im_h, im_l , 3))


def _picture_to_pixels(image, im_h, im_l):
    return image.reshape((im_h * im_l, 3))


if __name__ == '__main__':
    pass
