import os
from pathlib import Path
import random
import itertools

from skimage.io import imread
import numpy as np


def read_metadata(f):
    with open(f,"r") as f:
        metadata = {}
        for line in f:
            data = line.strip().split(",")
            filename = data[0]
            h,w = tuple(map(float,data[1:3]))
            lx,ly,rx,ry = np.array(tuple(map(float,data[3:])),"f")
            A, B =(w/2+lx, h/2-ly), (w/2+rx, h/2-ry)
            metadata[filename] = A, B, (h,w)
        return metadata


def read_list(f):
    with open(f,"r") as f:
        return [l.strip() for l in f]


class HorizonLinesInTheWildDataset:

    def __init__(self, path, split="train", load_images=True):
        self.path = Path(path)
        self.load_images = load_images

        if split not in ["train", "test", "val"]:
            raise KeyError("Wrong split")

        metadata = read_metadata(self.path / "metadata.csv")
        split_list_name = (self.path / "split" / split).with_suffix(".txt")
        image_list = read_list(split_list_name)

        self.data = [
            (x, metadata[x]) for x in image_list
        ]

    def __getitem__(self, idx):
        filename, (A, B, shape) = self.data[idx]
        if self.load_images:
            image = imread(os.fspath(self.path / "images" / filename))
            image_dict = dict(image=image, A=np.array(A), B=np.array(B), shape=shape, filename=filename)
        else:
            image_dict = dict(A=np.array(A), B=np.array(B), shape=shape, filename=filename)
        return image_dict

    def __len__(self):
        return len(self.data)
