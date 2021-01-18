from math import floor

import numpy as np
from image_geometry.line_segments import LineSegments
from skimage.transform import SimilarityTransform, downscale_local_mean, warp
from tensorflow.keras.models import load_model

from .baseline import accumulate_on_zenith, create_samples_from_groups
from .sord import rho_from_soft_label, theta_from_soft_label


class GCNet:
    """
    Wrapper that provides theta/rho values from CNN
    """

    def __init__(self, net_path):
        self.gcnet = load_model(net_path, compile=False)
        self.gcnet_shape = 224, 224  # Can be obtained from NN as the shape of the input tensor

    def downscale(self, image):
        h,w = image.shape[:2]
        u,v = self.gcnet_shape
        s = floor(min(h/u, w/v))
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


def get_zenith_line(lines:LineSegments, zenith_prior:np.ndarray=None) -> np.ndarray:
    groups = lines.get_field("group")
    num_g = min(4, groups.max()+1)

    vps = []

    for i in range(num_g):
        vp = fit_vanishing_point(lines[groups == i].homogeneous())
        vps.append(vp)

    vps = np.array(vps)  # (N,3), Vanishing points
    vp_dist = np.linalg.norm(vps[:,:2], axis=1, keepdims=True)
    dirs = vps[:,:2] / vp_dist  # Directions to vps

    if zenith_prior is None:
        zenith_prior = np.array([0,1])

    print("Zenith prior\n", zenith_prior)
    print("Directions\n", dirs)

    A = np.abs(np.array(dirs) @ np.atleast_2d(zenith_prior).T)  # cosine distance of directions to zenit guess
    print(A)

    valid = np.logical_and(A > 0.9, vp_dist[:] > 0.5)

    if not np.any(valid):
        return zenith_prior, None
    else:
        k = np.nonzero(valid)[0]
        k = k[0]
        return dirs[k], k


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def find_optimal_rho(
    zenith_direction,
    lines:LineSegments,
    pairs:np.ndarray,
    prior):

    l = lines.homogeneous()
    ln = lines.length().flatten()
    d = lines.direction()

    a, b = pairs.T

    H = np.cross(l[a], l[b])  # horizon candidate points
    W = np.maximum(ln[a], ln[b])

    # Normalize and keep only real points (remove those in infinity)
    k = np.abs(H[:,2]) > 1e-6
    H = H[k,:2] / H[k,2].reshape(-1, 1)
    W = W[k]

    d = (H @ np.atleast_2d(zenith_direction).T).flatten()  # Location along the zenith line

    print(W.shape, d.shape)
    mu, sigma = prior
    w = W * gaussian(d, mu, sigma) 

    return np.average(d, weights=w)


class HorizonPredictor:
    """
    This predicts the horizon with the prior from CNN using line segments
    """
    def __init__(self, net_path):
        self.gcnet = GCNet(net_path)
    
    def __call__(self, image_dict):
        """
        image_dict: dict
            Dict with image and its metadata
        """

        image = image_dict["image"]
        h,w = image.shape[:2]
        pp = (w/2, h/2)
        scale = max(h, w)
        shift = pp
        
        # Obtain prior from the cnn
        theta, rho = self.gcnet(image)
        theta = theta[0]
        rho = rho[0] / scale   # relative to image center so just scaling
        # Get zenith prior (initial guess) from theta
        z_prior = np.array([np.cos(theta), np.sin(theta)])

        # Image lines - coords relative to image center and scaled
        lines = image_dict["lines"].normalized(scale, shift)
        groups = lines.get_field("group")

        # Get zenith line using theta as prior
        zenith_direction, zenith_group = get_zenith_line(lines, z_prior)

        # Filter lines - remove lines supporting the zenith VP
        if zenith_group is not None:
            lines = lines[groups != zenith_group]
            groups = lines.get_field("group")

        # Generate voting pairs
        pairs = create_samples_from_groups(groups, 10000)
        
        # Vote on the zenith line with weights from rho
        rho = find_optimal_rho(zenith_direction, lines, pairs, (rho, 0.5))

        return zenith_direction, rho * scale