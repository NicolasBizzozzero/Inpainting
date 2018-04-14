""" ARF - TME8 : Algorithme k-means 
"""

import numpy as np
import matplotlib.pyplot as plt

from picture_tools import load_picture, show_picture, save_picture
from clustering import k_means
from initialisation import init_grid


PATH_PICTURE_SRC = "res/couleurs.jpg"
PATH_PICTURE_DEST = "out.png"


def main():
    pixels, im_h, im_l = load_picture(PATH_PICTURE_SRC)
    datax = init_grid(im_h, im_l)
    print(datax)

    # show_picture(pixels, im_h, im_l)


if __name__ == '__main__':
    main()
