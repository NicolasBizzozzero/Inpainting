from src.picture_tools.examples import CAMERAMAN, HOUSE, JETPLANE, LAKE, LENA_COLOR_256, LENA_COLOR_512, LENA_GRAY_256, \
    LENA_GRAY_512, LIVINGROOM, MANDRIL_COLOR, MANDRIL_GRAY, PEPPERS_COLOR, PEPPERS_GRAY, PIRATE, WALKBRIDGE, \
    WOMAN_BLONDE, WOMAN_DARKHAIR
from src.picture_tools.codage import Codage
from src.picture_tools.picture import Picture


PATH_DIR_USPS = "../res/USPS"

PICTURE_PATH = LENA_COLOR_512


def main():
    picture = Picture(PICTURE_PATH, codage=Codage.HSV)
    picture.add_noise()
    picture.show()


if __name__ == "__main__":
    main()
