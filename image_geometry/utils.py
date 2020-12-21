import numpy as np

from image_geometry import line_segments as ls
from image_geometry import line_groups_ransac as lgr
from skimage.filters import gaussian
from skimage.io import imread
##
def plot_line_segments(ax, ls, color, line_width=1, scale=1, zorder=None):
    segments = ls.coordinates()*scale
    y0,x0,y1,x1 = np.split(segments,4,axis=1)
    X = np.concatenate([x0,x1], axis=1)
    Y = np.concatenate([y0,y1], axis=1)
    for x,y in zip(X,Y):
        ax.plot(x,y,"-",c = color, lw=line_width)

## detect line segments
def detect_segments_in_set_with_py(image_set, path):

    segments = dict()
    groups = dict()
    for i, file in enumerate(image_set):
        print(file)
        gray = imread(path + file, as_gray=True)
        lines = ls.find_line_segments_ff(gray)
        scale = max(gray.shape)
        shift = np.array([gray.shape[0] / 2, gray.shape[1] / 2])# [h / 2, w / 2]
        l,g = lgr.find_line_groups_ransac_only(lines, scale, shift)
        segments[file] = l
        groups[file] = g

    return segments, groups

## detect line segments
def detect_segments_in_set_with_dll(image_set, path):
    segments = dict()
    groups = dict()
    for i, file in enumerate(image_set):
        print(file)
        gray = imread(path + file, as_gray=True)
        l_dir = librectify.detect_line_segments(gaussian(gray))
        lines,g = ls.LineSegments().load_from_dict(l_dir)

        segments[file] = lines
        groups[file] = g

    return segments,groups
##
def fit_vanishing_point(lines):

    #line normalization
    lines = lines / np.linalg.norm(lines, axis=1).reshape(-1, 1)

    #covariance
    covariance = np.matmul(lines.T, lines)

    #eigen values
    w, v = np.linalg.eig(covariance)

    #vanishing point
    vanishing_point = v[:, np.argmin(w)]
    vanishing_point = vanishing_point / vanishing_point[2]

    return vanishing_point

##

def inclination_from_horizon_points(A,B):
    direction = A - B
    angle = np.arctan2(direction[1],-direction[0])
    if angle > np.pi/2:
        angle -= np.pi
    if angle < -np.pi/2:
        angle += np.pi

    return angle/np.pi*180
##

def inclination_from_horizon(h):
    angle = np.arctan2(h[0], h[1])
    if angle > np.pi / 2:
        angle -= np.pi
    if angle < -np.pi / 2:
        angle += np.pi

    return angle/np.pi*180

##
def horizon_intersections(horizon, image_width):
    A = np.cross(horizon, np.array([1, 0, 0]))
    B = np.cross(horizon, np.array([1, 0, -image_width]))

    return A / A[2], B / B[2]