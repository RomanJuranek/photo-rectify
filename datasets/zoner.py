from pathlib import Path

import numpy as np
from skimage.io import imread


class ZPSHorizons:
    """Access to data provided by Zoner"""
    def __init__(self, datapath:Path):
        self.path = Path(datapath)
        self.data = list()
        with open(self.path/"Horizont.csv") as f:
            for k,line in enumerate(f.readlines()):
                if k < 1: continue
                parts = line.split(";")
                x1,y1,x2,y2 = map(float,parts[1:-1])
                self.data.append( (parts[0], np.array([x1,y1]), np.array([x2,y2])) )
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        name,A,B = self.data[idx]
        fn = Path(self.path/name).with_suffix(".jpg")
        image = imread(fn)
        return dict(image=image, A=A, B=B, filename=fn)
