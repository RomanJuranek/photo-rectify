"""
Author: Roman Juranek <ijuranek@fit.vutbr.cz>

Development of this software was funded by
TACR project TH04010394, Progressive Image Processing Algorithms.
"""


from functools import reduce

import numpy as np

from .geometry import *
from .line_segments import LineSegments,find_line_segments_ff


def _reduce_logical_and(*iterable):
    return reduce(np.logical_and, iterable)


def reliable_lines(lines:LineSegments, max_err=1, min_length=10, min_eigen_ratio=5) -> LineSegments:
    eigen_ratio = lines.max_eigen() / lines.min_eigen()
    mask = _reduce_logical_and(
        lines.reprojection_error() < max_err,
        lines.length() > min_length,
        eigen_ratio > min_eigen_ratio)
    return lines.gather(mask)


def fit_vanishing_point_ransac(lines:LineSegments, weights, max_iter=1000):

    best_fit, V = 0, None

    order = np.argsort(weights)[::-1] # Order of lines

    l = lines.homogenous()
    x = lines.anchor_point()
    n = lines.normal_vector()
    length = lines.length()

    for n_iters in range(max_iter):
        # Select two from initial group
        group_size = min(n_iters//20+4, len(order))
        i,j = np.random.choice(order[:group_size], 2,replace=False)
        # Calc intersection point
        v = np.cross(l[i], l[j])  # vanishing point hypothesis
        # Eval weighted fit function
        fit = (length * (inclination(x, n, v) > 0.9998)).sum()
        # if better, record the solution (intersection pt)
        if fit > best_fit:
            #print(f"{n_iters}: fit={fit:.2f}, V={v}")
            V,best_fit = v,fit
    return V, best_fit


def find_line_groups_ransac_only(lines:LineSegments):
    #n = lines.size    
    groups = []
    while lines.size > 2:
        V1,_ = fit_vanishing_point_ransac(lines, W, max_iter=4000)
        W1 = lines.inclination(V1)
        inlier_mask = W1>0.9998
        groups.append( lines.gather(inlier_mask) )
        lines = lines.gather(W1 < 0.95)
    return groups


def preprocess_image(image, max_size=1000):
    w = max(image.shape[:2])
    f = min(max_size / w, 1)
    if f < 1:
        image = ni.zoom(image, f)
    return image.astype(np.float32) / image.max(), f


def vp_direction(vps, shape):
    c = np.array(shape,"f") / 2  # Center point

    v_dir = []

    for vp in vps:
        if vp[2] == 0:
            v = vp[:2]
        else:
            v = vp[:2] - c
        v_dir.append(v)
    
    v_dir = np.array(v_dir,"f")
    v_dir /= np.linalg.norm(v_dir, axis=1, keepdims=True)
    return v_dir


def vertical_group(v_dir):
    """ indetify most likely vertical group - the one closest to [1,0] vector """
    g = np.argmax( np.abs(v_dir @ [[1],[0]]) )
    return g


def horizontal_group(v_dir):
    """ indetify most likely vertical group - the one closest to [0,1] vector """
    g = np.argmax( np.abs(v_dir @ [[0],[1]]) )
    return g


def compute_rectification_transform(
    image,
    block_size = 1,
    border_size = 0,
    min_line_length = 10):
    
    lines = find_line_segments_ff(
        image,
        seed_radius=5,
        border_size=border_size,
        mag_ratio=0.95,
        mag_tol=0.3)

    lines = reliable_lines(lines, max_err=block_size, min_length=min_line_length)

    groups = find_line_groups_ransac_only(lines)

    vps = [(closest_point_lsq(ls.homogenous()),ls) for ls in groups]

    d = vp_direction(vps)
    v_group = vertical_group(d)
    h_group = horizontal_group(d)

    


