from typing import Union

import numpy as np

Number = Union[int, float]


def normalize(array: np.ndarray, floor: Number = 0, ceil: Number = 1, origin_floor: Number = None,
              origin_ceil: Number = None):
    """ Min-max normalisation of a 2D array
    :param: array, 2D array to normalise.
    :param: floor, the minimal value wanted after normalisation.
    :param: ceil, the maximal value wanted after normalisation.
    :param: origin_floor, the minimal value of the array's domain before normalisation (if not provided, array.min()).
    :param: origin_ceil, the maximal value of the array's domain before normalisation (if not provided, array.max()).
    :return: The normalized array.
    """
    array_min = origin_floor if origin_floor is not None else array.min()
    array_max = origin_ceil if origin_ceil is not None else array.max()
    upper = (ceil - floor) / (array_max - array_min)
    lower = (ceil - upper * array_max)  # or (floor - upper * array_min)
    return upper * array + lower


if __name__ == "__main__":
    pass
