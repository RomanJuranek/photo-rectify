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

from itertools import count
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
from skimage.io import imread
from skimage.transform import resize

from .ecd import EurasianCities
from .gp3k import GeoPose3KDataset
from .gsw import GoogleDataset
from .hlw2 import HorizonLinesInTheWildDataset
from .yud import YorkUrbanDataset
from .zoner import ZPSHorizons


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



class DataWithCachedMasks:
    def __init__(self, dataset, redis, cache_target, key_prefix):
        self.dataset = dataset
        self.redis = redis
        self.cache_target = cache_target
        self.key_prefix = key_prefix

    def __getitem__(self, idx):
        image_dict:dict = self.dataset[idx]
        key = f"{self.key_prefix}:{idx}"
        identifier = self.redis.get(key).decode()
        #print(self.cache_target, identifier)
        data = np.load(self.cache_target / identifier)
        arr = [x[...,None] for x in [data["arr_0"], data["arr_1"], data["arr_2"]]]
        masks = np.concatenate(arr, axis=-1)
        masks = resize(masks, image_dict["image"].shape[:2] + (masks.shape[-1],), preserve_range=True).astype("f")
        image_dict.update(masks=masks)
        return image_dict

    def __len__(self):
        return len(self.dataset)


def random_iterator(seq:Sequence[Any], max_len=None) -> Any:
    for k in count():
        if k > max_len:
            return
        yield random.choice(seq)