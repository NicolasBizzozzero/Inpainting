from src.picture_tools.examples import CAMERAMAN, HOUSE, JETPLANE, LAKE, LENA_COLOR_256, LENA_COLOR_512, LENA_GRAY_256, \
    LENA_GRAY_512, LIVINGROOM, MANDRIL_COLOR, MANDRIL_GRAY, PEPPERS_COLOR, PEPPERS_GRAY, PIRATE, WALKBRIDGE, \
    WOMAN_BLONDE, WOMAN_DARKHAIR
from src.picture_tools.codage import Codage
from src.picture_tools.picture import Picture, show_patch


PATH_DIR_USPS = "../res/USPS"

PATCH_SIZE = 50
STEP = PATCH_SIZE
PICTURE_PATH = LENA_COLOR_512
CODAGE = Codage.RGB


def main():
    # Chargement de l'image
    picture = Picture(PICTURE_PATH, codage=CODAGE)
    picture.show()

    # Ajout du bruit
    picture.add_rectangle(45, 45, 50, 80)
    picture.add_noise(0.0001)
    picture.show()

    # Récuperation des patchs et du dictionnaire
    patches = picture.get_patches(size=PATCH_SIZE, step=STEP)
    dictionnaire = picture.get_dictionnaire(size=PATCH_SIZE, step=STEP)

    print("Affichage d'un patch avec pixels manquant")
    show_patch(patches[0], codage=CODAGE)
    print("Affichage d'un patch sans pixel manquant")
    show_patch(dictionnaire[0], codage=CODAGE)


if __name__ == "__main__":
    main()
