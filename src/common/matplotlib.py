from typing import List

import numpy as np
from matplotlib import pyplot as plt


def show_pictures(pictures: iter, titles: List[str] = None, columns: int = 1, save_path: str = None):
    """ Display a list of images in a single figure with matplotlib.

    :param pictures: List of np.arrays compatible with plt.imshow
    :param titles: List of titles corresponding to each image. Must have the same length as titles.
    :param columns: Number of columns in figure (number of rows is set to np.ceil(n_images / float(cols)))
    :param save_path: If provided, where to save the final figure.

    Source :
        https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    """
    assert ((titles is None) or (len(pictures) == len(titles)))

    # Set default title names for each picture
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, len(pictures) + 1)]

    fig = plt.figure()
    for index_picture, (picture, title) in enumerate(zip(pictures, titles)):
        sub_figure = fig.add_subplot(columns, np.ceil(len(pictures) / float(columns)), index_picture + 1)
        if picture.ndim == 2:
            plt.gray()
        plt.imshow(picture)
        sub_figure.set_title(title)
        sub_figure.axis("off")
    fig.set_size_inches(np.array(fig.get_size_inches()) * len(pictures))

    if save_path is not None:
        plt.savefig(save_path)
        plt.imread(save_path)
    plt.show()


if __name__ == "__main__":
    pass
