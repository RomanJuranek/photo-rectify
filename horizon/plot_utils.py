import numpy as np
import matplotlib.pyplot as plt
##
def zenith_intersections(img_size, A, B, pp = None):
    horizont_direction = A[0:2] - B[0:2]
    img_h = img_size[0]
    img_w = img_size[1]

    if pp is None:
        mid_point = [img_w/2, img_h/2]
    else:
        mid_point = pp

    horizont_direction = horizont_direction/np.linalg.norm(horizont_direction)
    zenith_line = [horizont_direction[0],horizont_direction[1], -mid_point[0]*horizont_direction[0] - mid_point[1]*horizont_direction[1]]

    C = np.cross(zenith_line, np.array([0, 1, 0]))
    D = np.cross(zenith_line, np.array([0, 1, -img_h]))

    C = C / C[2]
    D = D / D[2]
    return C, D
##
def draw_line_segments_array(coords, color='red'):
    x0, y0, x1, y1 = np.split(coords, 4, axis=1)
    X = np.concatenate([x0,x1], axis=1)
    Y = np.concatenate([y0,y1], axis=1)

    for x,y in zip(X,Y):
        plt.plot(x,y,"-",c=color, zorder=1, lw=2)
##
def draw_colored_line_segments_array(coords, group):
    colors = ['red','green','blue','yellow','orange','pink','violet']
    x0, y0, x1, y1 = np.split(coords, 4, axis=1)
    #X = np.concatenate([x0, x1], axis=1)
    #Y = np.concatenate([y0, y1], axis=1)

    for i in range(group.max() + 1):
        X = np.concatenate([x0[group==i], x1[group==i]], axis=1)
        Y = np.concatenate([y0[group==i], y1[group==i]], axis=1)
        for x,y in zip(X,Y):
            plt.plot(x, y, "-", c=colors[i], zorder=1, lw=1)

##
def draw_line_to_zenith(pp, zenith_point, img_height):
    zenith_line = np.cross(np.hstack([pp,1]), zenith_point)

    C = np.cross(zenith_line, np.array([0, 1, 0]))
    D = np.cross(zenith_line, np.array([0, 1, -img_height]))

    C = C / C[2]
    D = D / D[2]

    plt.plot(np.array([C[0], D[0]]), np.array([C[1],D[1]]), ':', color='blue')

    return
##
def draw_line_as_horizon(horizon_line, img_width):
    A = np.cross(horizon_line, np.array([1, 0, 0]))
    B = np.cross(horizon_line, np.array([1, 0, -img_width]))

    A = A / A[2]
    B = B / B[2]

    plt.plot(np.array([A[0], B[0]]), np.array([A[1], B[1]]), ':', color='blue')
    return
##
def draw_cross(img_data, A = None, B = None, horizon_line = None, zenith_point = None, lc = 'red'):
    img_h, img_w = img_data['shape']
    pp = img_data['pp']

    if horizon_line is not None:
        A = np.cross(horizon_line, np.array([1, 0, 0]))
        B = np.cross(horizon_line, np.array([1, 0, -img_w]))
        A = A / A[2]
        B = B / B[2]

    if zenith_point is None:
        horizont_direction = A[0:2] - B[0:2]
        horizont_direction = horizont_direction / np.linalg.norm(horizont_direction)
        zenith_line = [horizont_direction[0], horizont_direction[1], -pp[0] * horizont_direction[0] - pp[1] * horizont_direction[1]]
    else:
        zenith_line = np.cross(np.hstack([pp, 1]), zenith_point)

    C = np.cross(zenith_line, np.array([0, 1, 0]))
    D = np.cross(zenith_line, np.array([0, 1, -img_h]))

    C = C / C[2]
    D = D / D[2]

    plt.plot(np.array([A[0], B[0]]), np.array([A[1], B[1]]), '--', color=lc)
    plt.plot(np.array([C[0], D[0]]), np.array([C[1], D[1]]), '--', color=lc)


