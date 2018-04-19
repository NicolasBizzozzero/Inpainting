from typing import List

import numpy as np

from src.picture_tools.picture import Picture


class InPainting:
    def __init__(self, classifier = None):
        pass

    def fit(self, picture: Picture):
        pass

    def predict(self, picture: Picture) -> Picture:
        pass

    def _get_weights(self, patch: np.ndarray, dictionnary: List[np.ndarray]) -> np.ndarray:
        pass


if __name__ == "__main__":
    pass
