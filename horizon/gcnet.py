from math import floor

import numpy as np
from skimage.transform import SimilarityTransform, downscale_local_mean, warp
from tensorflow.keras.models import load_model

from .sord import rho_from_soft_label, theta_from_soft_label
from image_geometry.line_segments import LineSegments, find_line_segments_ff
from image_geometry.line_groups_ransac import find_line_groups_ransac_only


class GCNet:
    """
    Wrapper that provides theta/rho values from CNN
    """

    def __init__(self, net_path):
        self.gcnet = load_model(net_path, compile=False)
        self.gcnet_shape = 224, 224  # Can be obtained from NN as the shape if the input tensor

    def downscale(self, image):
        h,w = image.shape[:2]
        u,v = self.gcnet_shape
        s = floor(min(h/u, w/v))   # 2000/200, 1000/200 -> 10, 5
        return downscale_local_mean(image, (s,s,1))

    def __call__(self, image):
        """
        """
        downscaled = self.downscale(image) / 256
        h,w = downscaled.shape[:2]
        u,v = self.gcnet_shape
        half_size = (min(h,w)-1) // 2

        scale = min(image.shape[:2]) / u

        #print(half_size, u, v, h, w, scale)
        src = np.array([
            (w/2-half_size,h/2-half_size),  # A
            (w/2+half_size,h/2-half_size),  # B
            (w/2-half_size,h/2+half_size),  # C
        ])
        #print(src)
        dst = np.array([
            (0,0),  # A
            (u,0),  # B
            (0,v)   # C
        ])

        T = SimilarityTransform()
        T.estimate(src, dst)
        X = warp(downscaled, T.inverse, output_shape=self.gcnet_shape, order=2).astype(np.float32)

        T_pred, R_pred = self.gcnet.predict(X[None,...])

        return theta_from_soft_label(T_pred), scale * rho_from_soft_label(R_pred, K_range=100)

from image_geometry.line_groups_ransac import fit_vanishing_point


def get_zenith_line(lines: LineSegments, groups:np.ndarray, image_shape, zenith_prior:np.ndarray=None) -> np.ndarray:
    h,w = image_shape
    scale = max(h,w)
    pp = shift = (w/2, h/2)
    num_g = np.min([4, groups.max()+1])

    group_directions = []
    group_vps = []

    for i in range(num_g):
        vp = fit_vanishing_point(lines.gather(groups == i).scale_homogenous(scale, shift))
        vp[0:2] = vp[0:2]*scale + shift
        dir = vp[0:2]-pp
        dir = dir/np.linalg.norm(dir)
        group_directions.append(dir)
        group_vps.append(vp)

    if zenith_prior is None:
        zenith_prior = np.array([1,0])

    print(zenith_prior)

    A = np.abs(np.array(group_directions) @ np.atleast_2d(zenith_prior).T)  # cosine distance of directions to zenit guess
    print(A)

    k = np.argmax(A)

    if A[k] > 0.9: # cos(max_allowed_distance)
        d = group_directions[k]
    else:
        d = zenith_prior

    zenith_line = np.array([-d[1],d[0], d[1]*pp[0]-d[0]*pp[1]])
    print(zenith_line)

    return zenith_line, group_vps[k], k



class HorizonPredictor:
    """
    This predicts the horizon with the prior from CNN using line segments
    """
    def __init__(self, net_path):
        self.gcnet = GCNet(net_path)
    
    def __call__(self, image, lines:LineSegments, groups):
        """
        image:
            Image array (H,W,3)
        lines:
            Line segments with pixel coordinates (i.e. not normalized or scaled)
        """

        h,w = image.shape[:2]
        pp = (w/2, h/2)

        # Obtain prior from the cnn
        theta, rho = self.gcnet(image)
        theta = theta[0]

        # Get zenith prior (initial guess) from theta
        z_dir = np.array([np.cos(theta), np.sin(theta)])

        # Get zenith line using theta as prior
        zenith_line, zenith_point, zenith_group = get_zenith_line(lines, groups, image.shape[:2], z_dir)

        # Filter lines - remove lines supporting the zenith VP
        filtered_lines = lines.gather(groups != zenith_group)
        filtered_groups = groups[groups != zenith_group]

        # Generate voting pairs
        from .baseline import create_samples_from_groups, accumulate_on_zenith
        s = create_samples_from_groups(filtered_groups, 10000)
        
        # Vote on the zenith line with weights from rho
        x = accumulate_on_zenith(zenith_line, filtered_lines, s, pp)

        # Find best rho

        # Compose output - homogeneous coords of the horizon
        horizon_line = np.array([-zenith_line[1], zenith_line[0], -x])

        return zenith_point, horizon_line
