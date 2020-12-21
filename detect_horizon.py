import sys
import argparse
from skimage.io import imread
import image_geometry as ig
from horizon import baseline
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    image = imread(sys.argv[1], as_gray=True)
    
    lines = ig.find_line_segments_ff(image)
    scale = max(image.shape[:2])
    shift = np.array([image.shape[0] / 2, image.shape[1] / 2]) # [h / 2, w / 2]

    l,g = ig.find_line_groups_ransac_only(lines, scale, shift)
    l.switch_axes()  # A bit hacky!
    img_h, img_w = image.shape[:2]
    image_dict = dict(
        shape = image.shape[:2],
        pp = np.array([img_w / 2, img_h / 2]),
        lines = l,
        groups = g
    )
    zenith_point, horizon_line = baseline.detect_horizon(image_dict, picks=5000)

    A = np.cross(horizon_line, np.array([1, 0, 0]))
    B = np.cross(horizon_line, np.array([1, 0, -img_w]))

    A,B = A / A[2], B / B[2]

    plt.imshow(image, cmap="gray")
    plt.plot(np.array([A[0], B[0]]), np.array([A[1], B[1]]), '--', color=(1,1,0), lw=4)

    plt.savefig("output.png")