from pathlib import Path

import numpy as np
from scipy.io import loadmat
from skimage.io import imread


class YorkUrbanDataset:
    def __init__(self, data_path, split=None):
        self.path = Path(data_path)
        camera = loadmat(self.path / "cameraParameters.mat")
        self.focal = camera['focal'][0, 0]
        self.pixel_size = camera['pixelSize'][0, 0]
        self.pp = camera['pp'][0,:] - [1, 1]            #Matlab <> python indexing correction

        splits = loadmat(self.path / "ECCV_TrainingAndTestImageNumbers.mat")
        train_idx = splits["trainingSetIndex"].flatten() - 1
        test_idx = splits["testSetIndex"].flatten() - 1
        image_names = loadmat(self.path / "Manhattan_Image_DB_Names.mat")

        if split not in ["train", "test", None]:
            raise KeyError("Split must be train, test or None")
        
        if split is None:
            idx = slice(None)
        elif split == "train":
            idx = train_idx
        else:
            idx = test_idx
        
        image_names = [str(x[0][0][:-1]) for x in image_names["Manhattan_Image_DB_Names"][idx]]
        self.data = []
        for name in image_names:
            #print(name)
            im_file_name = self.path/name/f"{name}.jpg"
            gt_file_name = self.path/name/f"{name}GroundTruthVP_CamParams.mat"
            self.data.append( (im_file_name, gt_file_name) )
        
    def load_data(self, im_file, gt_file):
        image = imread(im_file)
        gt = loadmat(gt_file)
        vps = gt['vp'].T
        vp1 = [self.focal * vps[0, 0] / (vps[0, 2] * self.pixel_size) + self.pp[0], -self.focal * vps[0, 1] / (vps[0, 2] * self.pixel_size) + self.pp[1],1]
        vp2 = [self.focal * vps[2, 0] / (vps[2, 2] * self.pixel_size) + self.pp[0], -self.focal * vps[2, 1] / (vps[2, 2] * self.pixel_size) + self.pp[1],1]

        vp_z = [self.focal * vps[1, 0] / (vps[1, 2] * self.pixel_size) + self.pp[0], -self.focal * vps[1, 1] / (vps[1, 2] * self.pixel_size) + self.pp[1],1]

        horizon = np.cross(vp1,vp2)
        A = np.cross(horizon, np.array([1, 0, 0]))
        B = np.cross(horizon, np.array([1, 0, -image.shape[1]]))
        A = A[0:2] / A[2]
        B = B[0:2] / B[2]

        return image, A, B, vp_z

    def __getitem__(self, idx):
        image_file, gt_file = self.data[idx]
        image, A, B, vp_z = self.load_data(image_file, gt_file)
        return dict(image=image, A=A, B=B, filename=image_file, pp=self.pp, shape=image.shape[0:2], zenith_vp=vp_z)

    def __len__(self):
        return len(self.data)


