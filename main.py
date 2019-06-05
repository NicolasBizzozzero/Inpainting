from src.picture_tools.colormodel import ColorModel
from src.picture_tools.picture import Picture, VALUE_MISSING_PIXEL, VALUE_OUT_OF_BOUNDS
from src.inpainting import InPainting
from src.common.matplotlib import show_pictures
from src.picture_tools.examples import CAMERAMAN, HOUSE, JETPLANE, LAKE, LENA_COLOR_256, LENA_COLOR_512,\
    LENA_GRAY_256, LENA_GRAY_512, LIVINGROOM, MANDRIL_COLOR, MANDRIL_GRAY, PEPPERS_COLOR, PEPPERS_GRAY, PIRATE,\
    WALKBRIDGE, WOMAN_BLONDE, WOMAN_DARKHAIR, CASTLE, OUTDOOR


PATCH_SIZE = 101
STEP = PATCH_SIZE // 4
ALPHA = 1e-5
MAX_ITERATIONS = 1e+5
TOLERANCE = 1e-4
COLOR_MODEL = ColorModel.RGB


def main_lena():
    # Load picture
    original_picture = Picture(picture_path=LENA_COLOR_512, color_model=COLOR_MODEL)

    # Add noise
    noisy_picture = original_picture.copy()
    noisy_picture.add_noise(0.005)

    main_inpainting(original_picture, noisy_picture)


def main_castle():
    # Load picture
    original_picture = Picture(picture_path=CASTLE, color_model=COLOR_MODEL)

    # Add a rectangle of noise
    noisy_picture = original_picture.copy()
    noisy_picture.add_rectangle(400, 380, 50, 20)

    main_inpainting(original_picture, noisy_picture)


def main_outdoor():
    # Load picture
    original_picture = Picture(picture_path=OUTDOOR, color_model=COLOR_MODEL)

    # Add a rectangle of noise
    noisy_picture = original_picture.copy()
    noisy_picture.add_rectangle(288, 497, 190, 80)

    main_inpainting(original_picture, noisy_picture)


def main_inpainting(original_picture, noisy_picture, patch_size=PATCH_SIZE, step=STEP, alpha=ALPHA,
                    max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE, value_missing_pixel=VALUE_MISSING_PIXEL,
                    value_out_of_bounds=VALUE_OUT_OF_BOUNDS, save_path=None):
    inpainting = InPainting(patch_size=patch_size, step=step, alpha=alpha, max_iterations=max_iterations,
                            tolerance=tolerance, value_missing_pixel=value_missing_pixel,
                            value_out_of_bounds=value_out_of_bounds)
    inpainted_picture = inpainting.inpaint(noisy_picture)

    # Show resulting pictures
    show_pictures(pictures=[original_picture._get_showable_picture(), noisy_picture._get_showable_picture(),
                            inpainted_picture._get_showable_picture()],
                  titles=["Original picture", "Noisy picture", "InPainted picture"],
                  save_path=save_path)


if __name__ == "__main__":
    main_outdoor()
