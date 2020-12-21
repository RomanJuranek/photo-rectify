""" Various geometry related functions
"""

import numpy as np


def wpca(X, w):
    """ Weighted PCA """
    U,E,_ = np.linalg.svd(X.T @ np.diag(w) @ X) # '@' is matrix multiplication
    return U, E


def homogenous(x, n):
    x = np.atleast_2d(x)
    n = np.atleast_2d(n)
    """ Homogenous coordinates of lines - ax+by+c = 0 """
    assert x.shape[0] == n.shape[0], "Number of rows of 'x' and 'n' must match"
    assert x.shape[1] == 2 and n.shape[1] == 2, "'x' and 'n' must be 2 column matrices"
    c = (x * n).sum(axis=1)
    return np.concatenate([n, -c.reshape(-1,1)], axis=1)

def scale_homogenous(x, n, scale, shift):

    x = (np.atleast_2d(x) - np.atleast_2d(shift))/scale
    n = np.atleast_2d(n)
    """ Homogenous coordinates of lines - ax+by+c = 0 """
    assert x.shape[0] == n.shape[0], "Number of rows of 'x' and 'n' must match"
    assert x.shape[1] == 2 and n.shape[1] == 2, "'x' and 'n' must be 2 column matrices"
    c = (x * n).sum(axis=1)
    return np.concatenate([n, -c.reshape(-1,1)], axis=1)

def inclination(x, n, point):
    """ Returns cos(angle) between lines (x, n) and point p
    """
    assert x.shape[1] in {2,3}, "x must have 2 or 3 columns"
    assert n.shape[1] == 2, "n must have 2 columns"
    assert x.shape[0] == n.shape[0], "Number of rows of x and n must match"
    n_lines = x.shape[0]
    if x.shape[1] == 2:
        x = np.hstack([x, np.ones((n_lines,1))])
    cross_lines = np.cross(x, point)

    norm = np.linalg.norm(cross_lines[:,:2], axis=1)

    cross_lines = cross_lines / norm.reshape(-1,1)
    return np.abs( (n*cross_lines[:,:2]).sum(axis=1) )


def closest_point_lsq(lines, weights=None):
    """ Find weighted closest point to a set of lines in homogenous coordinates """
    if weights is None:
        weights = np.ones(lines.shape[0])
    scale = lines[:,2].mean()
    lines[:,2] /= scale
    lines = lines / np.linalg.norm(lines, axis=1, keepdims=True)
    v,w = wpca(lines, weights)
    p = v[:, np.argmin(w)]
    if p[2] != 0:
        p /= p[2]
    p[:2] *= scale
    return p


def image_point_to_world_coordinates(V, focal, pp, pixel_size):
    M = np.diag([-1,1,1])
    V = V @ M
    world_coord = (V - [-pp[0],pp[1],1-focal/pixel_size])*pixel_size
    #world_coord = np.array([-V[0] + pp[0], V[1] - pp[1],focal/pixel_size])*pixel_size
    world_coord = world_coord/np.linalg.norm(world_coord, axis=1).reshape(-1, 1)
    return world_coord

def world_point_to_image_coordinates(W, focal, pp, pixel_size):
    image_coord = [-focal*W[0]/(W[2]*pixel_size) + pp[0],focal*W[1]/(W[2]*pixel_size) + pp[1]]
    return image_coord
