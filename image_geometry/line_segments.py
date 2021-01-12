"""
Gradient based line segment detection

Author: Roman Juranek <ijuranek@fit.vutbr.cz>
"""


import logging

import numpy as np
import scipy.ndimage as ni
from skimage.segmentation import flood

from .geometry import inclination, wpca


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
    """
    A set of line segments defined by their coordinates
    """
    def __init__(self, C, **kwargs):
        if not isinstance(C,np.ndarray):
            raise TypeError("Coordinates must be a numpy array")
        if C.ndim != 2 or C.shape[1] != 4:
            raise ValueError("Coordinates must be a matrix with 4 columns")
        self.C = np.atleast_2d(C.copy())
        self.fields = dict()
        for field,value in kwargs.items():
            self.set_field(field, value)
    @staticmethod
    def fit_segment(X:np.ndarray, W:np.ndarray):
        """Fit a single line segment to points"""
        A, U, E = fit_pca(X, W)
        D = U[np.argmax(E)]  # D - direction vector |D|=1
        N = U[np.argmin(E)]  # N - normal vector |N|=1
        t = np.dot(A-X, D)   # projection of X to direction vector - A + t*D is the projection coordinates
        e = np.dot(A-X, N)
        x1,y1  = A - D * t.min()
        x2,y2  = A - D * t.max()
        return [x1,y1,x2,y2], np.abs(e).max(), W.mean()
    @staticmethod
    def fit(iterable):
        """Fit multiple line segments and return instance of LineSegments.

        Exceptions may be raised when nonconforming tuple item,
        or wrong shapes of arrays is encountered.

        Input
        -----
        iterable:
            An iterable object providing (X, W) tuples where X is a
            numpy array with pointw and W array with point weights.

        Output
        ------
        L : LineSegments
            New instance of LineSegments with line segments fitted to
            the points in the input iterable. It contains fields "width"
            and "weight".

        See also
        --------
        LineSegments.fit_segment - used internally fot segment fitting
        """
        coords, width, weight = zip(*(LineSegments.fit_segment(X, W) for X,W in iterable))
        L = LineSegments(np.array(coords), width=np.array(width), weight=np.array(weight))
        return L
    @classmethod
    def from_dict(line_dict:dict):
        L = LineSegments(line_dict["coordinates"])
        for field, val in line_dict.items():
            if field is not "coordinates":
                L.set_field(field, val)
    def to_dict(self) -> dict:
        pass
    def __len__(self) -> int:
        return self.C.shape[0]
    def __getitem__(self, indices):
        L = LineSegments(self.C[indices])
        for field, val in self.fields.items():
            L.set_field(field, val[indices])
        return L
    def cat(self, other):
        """Concatenate two sets of line segments, keeping only common fields"""
        new_keys = set(self.get_fields()).intersection(other.get_fields())
        L = LineSegments(np.concatenate([self.coordinates(), other.coordinates()], axis=0))
        for k in new_keys:
            val_a = self.get_field(k)
            val_b = other.get_field(k)
            L.set_field(k, np.concatenate([val_a, val_b], axis=0))
        return L
    def normalized(self, scale=1, shift=(0,0)):
        # TODO: validate shift 2-Tuple or np.array of size 2, 1x2 shape
        shift = np.tile(np.atleast_2d(shift), 2)
        L = LineSegments(C = (self.C - shift) / scale)
        for field, val in self.fields.items():
            L.set_field(field, val)
        return L
    def coordinates(self) -> np.ndarray:
        return self.C
    def endpoints(self, homogeneous=False):
        A, B = np.split(self.C, 2, axis=1)
        if homogeneous:
            ones = np.ones((A.shape[0],1),"f")
            A = np.hstack([A, ones])
            B = np.hstack([B, ones])
        return A, B
    # anchor
    def anchor(self) -> np.ndarray:
        A, B = self.endpoints()
        return (A + B)/2
    # length
    def length(self) -> np.ndarray:
        A, B = self.endpoints()
        return np.linalg.norm(B-A, axis=1)
    # normal
    def normal(self) -> np.ndarray:
        direction = self.direction()
        u, v = np.split(direction, 2, axis=-1)
        return np.hstack([-v, u])
    # direction
    def direction(self) -> np.ndarray:
        A, B = self.endpoints()
        direction = (B - A)
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        return direction
    # homogeneous
    def homogeneous(self, normalized=True) -> np.ndarray:
        A, B = self.endpoints(homogeneous=True)
        h = np.cross(A, B)
        if normalized:
            h /= np.linalg.norm(h, axis=-1, keepdims=True)
        return h
    # inclination
    def inclination(self, p) -> np.ndarray:
        return inclination(self.anchor(), self.normal(), p)
    #
    def _validate_field(self, v:np.ndarray) -> bool:
        if not isinstance(v, np.ndarray):
            raise TypeError("Only numpy arrays are supported for fields")
        if v.shape[0] != len(self):
            raise ValueError(f"Expected {len(self)} items, {v.shape[0]} passed")
    # get_field
    def get_field(self, field):
        return self.fields[field]
    # set_field
    def set_field(self, field, value, overwrite=True):
        self._validate_field(value)
        if overwrite and field in self.fields:
            raise KeyError(f"Field {field} already present")
        self.fields[field] = value.copy()
    def has_field(self, field) -> bool:
        return field in self.fields
    def get_fields(self):
        return self.fields.keys()


def fit_line_segments(components, mag, min_size=50, max_size=10000, scale=1):
    """ Fit line segments on image components and return instance of LineSegments """
    def line_support():
        for c in components:
            X = c["X"]
            if min_size < X.shape[0] < max_size:
                yield X*scale, mag[X[:,1], X[:,0]]
    return LineSegments.fit(line_support())


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
    lines: An instance of LineSegments or, if return_internals=True, a dict with
        internal stuff of the line traing algorithm

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
        inds = np.nonzero(component)
        component_points = np.array((inds[1], inds[0]),"i").T
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
