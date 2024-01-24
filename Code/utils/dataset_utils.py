from os import path
from torchvision.datasets import Caltech101


def initialize_dataset():
    """
    Returns a wrapper around the dataset. Downloads if not present. Stores in
    /Code/caltech101/. Ignored by git.
    """
    print("Downloading dataset if not present.")

    return Caltech101(root=path.dirname(path.dirname( __file__ )), download=True)


def get_image_with_label(dataset, image_id):
    """
    Return a tuple of PIL image and the corresponding label for a given image ID.
    """
    return dataset[image_id][0], dataset.categories[dataset[image_id][1]]