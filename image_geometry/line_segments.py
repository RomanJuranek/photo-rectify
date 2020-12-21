"""
Gradient based line segment detection

Author: Roman Juranek <ijuranek@fit.vutbr.cz>
"""


import logging
import pickle
from itertools import compress
from math import floor

import numpy as np
import scipy.ndimage as ni
import skimage.morphology as mo
from skimage.filters import scharr_h, scharr_v
from skimage.measure import block_reduce
from skimage.segmentation import flood

from .geometry import homogenous, inclination, wpca, scale_homogenous


def triangle_kernel(size=1):
    """ Triangle kernel with 2*size+1 width """
    H = np.array(np.concatenate( [np.r_[1:size+2], np.r_[size:0:-1]]),"f")
    H /= H.sum()
    return H


def smooth_image(image, size=1):
    """ Separable image smooth with triangle kernel """
    smoothed = np.empty_like(image, "f")
    H = triangle_kernel(size)
    ni.convolve1d(image, H, output=smoothed, axis=0)
    ni.convolve1d(smoothed, H, output=smoothed, axis=1)
    return smoothed


def gauss_deriv_kernel(size, sigma=1, phase=(0,0), direction="x"):
    assert direction in {"x","y"}, "Direction must be 'x' or 'y'"
    kx = np.arange(-size,size+1,1)-phase[0]
    ky = np.arange(-size,size+1,1)-phase[1]
    x, y = np.meshgrid(kx,ky)
    z = x if direction=="x" else y
    return (z / (2*np.pi*sigma**4)) * np.exp(-(x**2+y**2)/(2*sigma**2)).astype("f")


def fit_pca(X, weights=None):
    """ Fit line parameters to points using weighted PCA """
    if weights is None:
        weights = np.ones(X.shape[0])
    A = np.mean(X*weights.reshape(-1,1), axis=0) / weights.mean()  # A - anchor point
    U, E = wpca(X-A, weights/weights.sum())
    return A, U, E


class LineSegments:
    """ Line segments and their parameters """
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    @staticmethod
    def from_points(iterable):
        params = []
        for X,W in iterable:
            A, U, E = fit_pca(X, W)
            D = U[np.argmax(E)]  # D - direction vector |D|=1
            N = U[np.argmin(E)]  # N - normal vector |N|=1
            t = np.dot(A-X, D)   # projection of X to direction vector - A + t*D is the projection coordinates
            e = np.abs(np.dot(A-X, N))
            params.append(
                (A, D, E, (t.min(),t.max()), e.mean(), W.mean())
            )
        anchor,direction,eigens,endpt,err,weight = zip(*params)
        return LineSegments(
            anchor=np.array(anchor,"f"),
            direction=np.array(direction,"f"),
            endpt=np.array(endpt,"f"),
            eigens=np.array(eigens,"f"),
            err=np.array(err,"f"),
            weight=np.array(weight,"f"))

    @staticmethod
    def load(filename:str):
        with open(filename, "rb") as f:
            a,d,e,eig,err,w = pickle.load(f)
            return LineSegments(
                anchor=a,
                direction=d,
                endpt=e,
                eigens=eig,
                err=err,
                weight=w)

    @staticmethod
    def load_from_dict(line_dict):
        w = line_dict['weight']
        err = line_dict['error']
        endpoints = line_dict['line']
        endpoints = endpoints[:,(1,0,3,2)]
        d = endpoints[:,2:4] - endpoints[:,0:2]
        a = endpoints[:,0:2] + d/2
        d /= np.linalg.norm(d,axis=1).reshape(-1,1)
        e1 = np.linalg.norm(endpoints[:,0:2] - a, axis=1)

        group = line_dict['group']
        return LineSegments(
            anchor=np.array(a,"f"),
            direction=np.array(d,"f"),
            endpt= np.vstack([-1*e1,e1]).T,
            err=np.array(err,"f"),
            weight=np.array(w,"f")), group

    @staticmethod
    def load_from_array(line_array):
        w = np.ones(line_array.shape[0])
        err = np.zeros(line_array.shape[0])
        endpoints = line_array
        d = endpoints[:, 2:4] - endpoints[:, 0:2]
        a = endpoints[:, 0:2] + d / 2
        d /= np.linalg.norm(d,axis=1).reshape(-1,1)

        e1 = np.linalg.norm(endpoints[:,0:2] - a, axis=1)

        return LineSegments(
            anchor=np.array(a,"f"),
            direction=np.array(d,"f"),
            endpt= np.vstack([-1*e1,e1]).T,
            err=np.array(err,"f"),
            weight=np.array(w,"f"))


    @staticmethod
    def concatenate(ls):
        anchor = np.concatenate([l.anchor for l in ls])
        direction = np.concatenate([l.direction for l in ls])
        endpt = np.concatenate([l.endpt for l in ls])
        err = np.concatenate([l.err for l in ls])
        weight = np.concatenate([l.weight for l in ls])
        return LineSegments(anchor=anchor, direction=direction,endpt=endpt,err=err, weight=weight)

    def switch_axes(self):
        self.anchor =  self.anchor[:,::-1]
        self.direction = self.direction[:,::-1]

    def save(self, filename:str):
        with open(filename, "wb") as f:
            pickle.dump((self.anchor, self.direction, self.endpt, self.eigens, self.err, self.weight), f)

    @property
    def size(self):
        return self.anchor.shape[0]

    def anchor_point(self):
        return self.anchor

    def direction_vector(self):
        return self.direction

    def coordinates(self):
        e0  = self.anchor - self.direction * self.endpt[:,0,None]
        e1  = self.anchor - self.direction * self.endpt[:,1,None]
        return np.hstack([e0,e1])

    def length(self):
        return np.abs(self.endpt[:,1] - self.endpt[:,0])

    def max_eigen(self):
        return np.max(self.eigens, axis=1)

    def min_eigen(self):
        return np.fmax(np.min(self.eigens, axis=1), 1e-3)

    def reprojection_error(self):
        return self.err

    def line_weight(self):
        return self.weight

    def normal_vector(self):
        a,b = np.split(self.direction, 2, axis=1)
        return np.concatenate([-b,a], axis=1)

    def normalized_direction_vector(self):
        d = self.direction
        d /= np.linalg.norm(d, axis=1).reshape(-1, 1)
        return d

    def normalized_normal_vector(self):
        n = self.normal_vector()
        n /= np.linalg.norm(n,axis=1).reshape(-1, 1)
        return n

    def homogenous(self):
        return homogenous(self.anchor, self.normal_vector())

    def scale_homogenous(self, scale, shift):
        return scale_homogenous(self.anchor, self.normal_vector(), scale, shift)

    def inclination(self, p):
        return inclination(self.anchor_point(), self.normal_vector(), p)

    def gather(self, mask):
        """ Return subset of lines """
        assert mask.dtype == np.bool, "Vole!"
        return LineSegments(
            anchor=self.anchor[mask,...],
            direction=self.direction[mask,...],
            endpt=self.endpt[mask,...],
#            eigens=self.eigens[mask],
            err=self.err[mask],
            weight=self.weight[mask])


def fit_line_segments(components, mag, min_size=50, max_size=10000, scale=1):
    """ Fit line segments on image components and return instance of LineSegments """
    def line_support():
        for c in components:
            X = c["X"]
            if min_size < X.shape[0] < max_size:
                yield X*scale, mag[X[:,0], X[:,1]]
    return LineSegments.from_points(line_support())


def mask_borders(image, b, value):
    """ Set image borders to a value """
    if b > 0:
        image[:,:b] = value
        image[:,-b:] = value
        image[:b,:] = value
        image[-b:,:] = value


def find_line_segments_ff(image,
                          seed_radius=7,
                          border_size=4,
                          n_bins=8,
                          mag_ratio = 0.9,
                          mag_tol=0.3,
                          return_internals=False):
    """ Detect line segments

    The algotithm processds in this steps:
    * Calculate gradients and gradient magnitude and optionally downsample
    * Get seed points as local maximas of magnitude
    * Assign each seed point to an orientation bin based on its orientation
    * For each bin:
      * Calculate gradient magnitude for the bin orientation
      * From each seed point trace pixels using flood function
      * Fit line to the traced pixels and calculate endpoints

    Input
    -----
    image : ndarray
        The input image with shape (H,W). Image is converted to float and
        normalized to 0-1 range.
    block_size : int
        Aggregation factor - image gradients and magnitude is downscaled
        by this factor before detecting lines in oder to increaseprocessing
        speed and reduce possible noise. Higher values lead to less precise
        lines.
    n_bins : int
        Number of orientation bins in which lines are traced.
    mag_ratio : float
        Ratio of 
    mag_tol : float
    return_internals : bool
        Instead of line segments, return internal variables - gradients, etc.

    Output
    ------


    Example
    -------

    """
    image = image.astype("f") / image.max()

    logging.debug(f"Calculating gradients and edge magnitudes")
    KX = gauss_deriv_kernel(2,1,direction="x")
    KY = gauss_deriv_kernel(2,1,direction="y")
    dx = ni.correlate(image, KX)
    dy = ni.correlate(image, KY)
    mask_borders(dx, border_size, 0)
    mask_borders(dy, border_size, 0)
    mag = np.sqrt(dx*dx + dy*dy)
    
    # if block_size > 1:
    #     block_size = (block_size,)*2  # scalar x -> tuple (x,x)
    #     logging.debug(f"Downsampling image")
    #     dx = block_reduce(dx, block_size, np.mean)
    #     dy = block_reduce(dy, block_size, np.mean)
    #     mag = block_reduce(mag, block_size, np.max)

    logging.debug(f"Calculating oriented gradient magnitude (nbins={n_bins})")
    theta = np.linspace(0, np.pi, n_bins, endpoint=False)   
    # seed_dir = np.array([dx[r,c], dy[r,c]]).T
    grad_dir = np.array([np.sin(theta), np.cos(theta)])
    # affinity = np.abs(seed_dir @ grad_dir)
    # grad_class = np.argmax(affinity, axis=1)    
    grad_mag_arr = np.array([np.abs(dx*wx + dy*wy) for wx,wy in grad_dir.T], "f")
    grad_mag_ind = np.argmax(grad_mag_arr, axis=0)

    logging.debug(f"Searching for seed points")
    mag_top = ni.maximum_filter(mag, seed_radius)
    seed_mask = (mag == mag_top)
    r,c = np.nonzero(seed_mask)
    seed_mag = mag[r,c]
    grad_class = grad_mag_ind[r,c]
    
    mag_threshold = (1-mag_ratio) * seed_mag.max()
    seed_mask = seed_mag > mag_threshold
    logging.debug(f"{seed_mask.sum()} seed points found (total {r.size})")
    r = r[seed_mask]
    c = c[seed_mask]
    seed_mag = seed_mag[seed_mask]
    grad_class = grad_class[seed_mask]

    logging.debug(f"Sorting seed points")
    seed_order = np.argsort(seed_mag)[::-1]
    r = r[seed_order]
    c = c[seed_order]
    seed_mag = seed_mag[seed_order]
    grad_class = grad_class[seed_order]

    logging.debug("Tracing lines")
    found = np.zeros_like(mag,"i")
    components = []

    grad_images = []

    for g, grad_mag in enumerate(grad_mag_arr):
        bin_mask = ni.binary_dilation(grad_mag_ind==g, iterations=2)
        grad_image = grad_mag * bin_mask
        grad_images.append(grad_image)
        #seed_idx = grad_class == g

    #for i,seed in enumerate(zip(r[seed_idx],c[seed_idx]),start=found.max()+1):
    for i,seed in enumerate(zip(r,c,grad_class,seed_mag)):
        a,b,seed_bin,seed_mag = seed
        if found[a,b]:
            continue
        tol = seed_mag * mag_tol
        component = flood(grad_images[seed_bin], (a,b), tolerance=tol, selem=np.ones( (3,3) ) )
        found[component] = i
        component_points = np.array(np.nonzero(component)).T  # (N,2)
        components.append({"X": component_points, "seed_point": seed})

    logging.debug("Calculating segment parameters")
    segments = fit_line_segments(components, mag, min_size=10, scale=1)
    
    if return_internals:
        return {
            "dx":dx, "dy":dy, "mag":mag, "mag_ind": grad_mag_ind,
            "seed_points": [r,c],
            "seed_grad_class": grad_class,
            "grad_dir": grad_dir,
            "segment_labels": found,
            "components": components,
            "segments" : segments,
        }
    else:
        return segments
