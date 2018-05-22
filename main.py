from src.picture_tools.examples import CAMERAMAN, HOUSE, JETPLANE, LAKE, LENA_COLOR_256, LENA_COLOR_512, LENA_GRAY_256, \
    LENA_GRAY_512, LIVINGROOM, MANDRIL_COLOR, MANDRIL_GRAY, PEPPERS_COLOR, PEPPERS_GRAY, PIRATE, WALKBRIDGE, \
    WOMAN_BLONDE, WOMAN_DARKHAIR, CASTLE, OUTDOOR
from src.picture_tools.codage import Codage
from src.picture_tools.picture import Picture, show_patch
from src.usps_tools import test_all_usps_1_vs_all, test_all_usps
from src.linear.linear_regression import LinearRegression, identite, mse_g, l1, l1_g, l2, l2_g, DescenteDeGradient
from src.inpainting import InPainting


PATCH_SIZE = 21
STEP = PATCH_SIZE
PICTURE_PATH = LENA_COLOR_512
CODAGE = Codage.RGB


def main():
    # Chargement de l'image
    picture = Picture(PICTURE_PATH, codage=CODAGE)
    picture.show()

    # Ajout du bruit
    picture.add_rectangle(250, 190, 40, 50)
    # picture.add_noise(0.01)
    picture.show()

    # On inpaint l'image !
    inpainting = InPainting(PATCH_SIZE)
    picture = inpainting.inpaint(picture)
    picture.show()


def main_all_vs_all():
    print("TEST ALL VS ALL")
    print("MSE")
    # test_all_usps(classifieur=LinearRegression,
    #               loss_g=mse_g,
    #               type_descente=DescenteDeGradient.BATCH,
    #               alpha=0)
    for loss_g in (l2_g, l1_g):
        for alpha in (0, 0.25, 0.5, 0.75, 1):
            print(loss_g.__name__, "alpha=" + str(alpha))
            test_all_usps(classifieur=LinearRegression,
                          loss_g=loss_g,
                          type_descente=DescenteDeGradient.BATCH,
                          alpha=alpha)


def main_1_vs_all():
    print("TEST 1 VS ALL")
    print("MSE")
    test_all_usps_1_vs_all(classifieur=LinearRegression,
                           loss_g=mse_g,
                           type_descente=DescenteDeGradient.BATCH,
                           alpha=0)
    for loss_g in (l1_g, l2_g):
        for alpha in (0, 0.25, 0.5, 0.75, 1):
            print(loss_g.__name__, "alpha=" + str(alpha))
            test_all_usps_1_vs_all(classifieur=LinearRegression,
                                   loss_g=loss_g,
                                   type_descente=DescenteDeGradient.BATCH,
                                   alpha=alpha)


if __name__ == "__main__":
    main()
