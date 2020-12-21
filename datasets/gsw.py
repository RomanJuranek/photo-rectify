import os
import random
import re
from pathlib import Path

import numpy as np
from skimage.io import imread


def params_from_name(f):
    x = re.split('_|\\.',f)
    fov = float(x[4])/180*np.pi
    pitch = float(x[6])/180*np.pi
    return fov, pitch


class GoogleDataset:
    def __init__(self, path):
        self.base_path = Path(path)
        self.files = self.base_path.glob("*.jpg")
        self.data = []
        for f in self.files:
            try:
                fov,pitch = params_from_name(os.path.basename(f))
                self.data.append( (f,fov,pitch) )
            except:
                pass

    def __getitem__(self, index):
        filename,fov,pitch = self.data[index]
        image = imread(filename)
        h = (image.shape[0] * np.tan(pitch)) / (2 * np.tan(fov / 2))
        h += image.shape[0]/2
        A = np.array([0,h])
        B = np.array([image.shape[1],h])
        return dict(image=image, A=A, B=B, filename=filename)

    def __len__(self):
        return len(self.data)