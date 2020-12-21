import os.path
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from skimage.io import imread


class EurasianCities:
    def __init__(self, data_path):
        self.path = Path(data_path)

        image_names = [f.stem for f in self.path.glob("*.jpg")]
        self.data = []
        for name in image_names:
            #print(name)
            im_file_name = self.path / f"{name}.jpg"
            gt_file_name = self.path / f"{name}hor.mat"

            self.data.append((im_file_name, gt_file_name))

    def load_data(self, im_file, gt_file):

        horizon = loadmat(gt_file)['horizon'].flatten()
        image = imread(im_file)

        A = np.cross(horizon, np.array([1, 0, 0]))
        B = np.cross(horizon, np.array([1, 0, -image.shape[1]]))

        A = A[0:2] / A[2]
        B = B[0:2] / B[2]

        return image, A, B

    def __getitem__(self, idx):
        image_file, gt_file = self.data[idx]
        image, A, B = self.load_data(image_file, gt_file)
        return dict(image=image, A=A, B=B, filename=image_file, shape=image.shape[0:2])

    def __len__(self):
        return len(self.data)

