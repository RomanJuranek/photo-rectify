"""
Accessor classes for various datasets

Dataset classes provide unified interface for reading data simply by indexing
the class instance.

Example
-------

d = datasets.YorkUrbanDataset(path_to_data)

image_dict = d[0]  # get first item
image_dict = random.choice(d)  # get random item
for image_dict in d:  # iterate through the dataset
    # ... whatever
    pass

Infinite random iterator can be implemented as:

def random_iter(dataset):
    while True:
        yield random.choice(dataset)

The image_dict contains at least the key "image" with the image array.
Ground truth annotation (a horizon line) is in keys "A", "B" with coordinates
of two points on the horizon.

There might be other keys like "filename", "pp", etc. that can be emplyed
for horizon detecion or evaluation purposes.
"""

from pathlib import Path

from skimage.io import imread

from .ecd import EurasianCities
from .gsw import GoogleDataset
from .hlw2 import HorizonLinesInTheWildDataset
from .yud import YorkUrbanDataset


class ImageDirDataset:
    """
    Image directory as a dataset with no annotations
    """
    def __init__(self, path: Path, suffixes=None):
        self.fs = list(filter(lambda f: f.is_file or (suffixes and f.suffix not in suffixes), Path(path).glob("*")))  # one liner!

    def __len__(self):
        return len(self.fs)

    def __getitem__(self, idx):
        return dict(image=imread(self.fs[idx]))
