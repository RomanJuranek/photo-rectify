import numpy as np

from .line_segments import LineSegments
from .geometry import inclination
from .utils import *

##
def fit_best_vanishing_point(lines:LineSegments, weights, max_iter=1000, scale = 1, shift = [0,0]):

    best_fit, V = 0, None

    order = np.argsort(weights)[::-1] # Order of lines

    l = lines.scale_homogenous(scale, shift)
    x = (lines.anchor_point() - shift)/scale
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

    if np.abs(V[2]) > 1e-6:
        V = V/V[2]
        V[0:2] = V[0:2] * scale + shift
    return V, best_fit

##
def find_line_groups_ransac_only(lines:LineSegments, scale = 1, shift = [0,0]):
    #n = lines.size
    line_groups = []

    while lines.size > 2 and len(line_groups) < 5:
        V1,_ = fit_best_vanishing_point(lines, lines.line_weight(), 10000, scale, shift)
        W1 = lines.inclination(V1)
        inlier_mask = W1>np.cos(np.pi*2/180) # 2deg tolerance
        line_groups.append(lines.gather(inlier_mask) )
        lines = lines.gather(W1 < np.cos(np.pi*4/180)) # 4deg tolerance

    lines = LineSegments.concatenate(line_groups)
    groups = np.concatenate([np.full(l.size, i) for i, l in enumerate(line_groups)])
    return lines, groups
##
def find_line_groups_and_vps(lines:LineSegments, scale = 1, shift = [0,0]):
    #n = lines.size
    line_groups = []
    V = []

    while lines.size > 2 and len(line_groups) < 5:
        V1,_ = fit_best_vanishing_point(lines, lines.line_weight(), 10000, scale, shift)
        W1 = lines.inclination(V1)
        hard_inliers = W1 > np.cos(np.pi*2/180) # 2deg tolerance
        soft_inliers = W1 > np.cos(np.pi*4/180) # 4deg tolerance

        vp = fit_vanishing_point(lines.gather(hard_inliers).scale_homogenous(scale, shift))
        vp[0:2] = vp[::-1][1:3] * scale + shift[::-1]
        V.append(vp)
        line_groups.append(lines.gather(soft_inliers))
        lines = lines.gather(~soft_inliers)

    lines = LineSegments.concatenate(line_groups)
    groups = np.concatenate([np.full(l.size, i) for i, l in enumerate(line_groups)])
    return lines, groups, V
