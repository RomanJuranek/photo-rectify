import logging
from typing import Mapping

import numpy as np
import sklearn.metrics


def calc_auc(error_array, cutoff=0.25):
    error_array = np.atleast_1d(error_array.flatten())
    error_array = np.sort(error_array)
    num_values = error_array.shape[0]

    plot_points = np.zeros((num_values, 2))

    midfraction = 1.

    for i in range(num_values):
        fraction = (i + 1) * 1.0 / num_values
        value = error_array[i]
        plot_points[i, 1] = fraction
        plot_points[i, 0] = value
        if i > 0:
            lastvalue = error_array[i - 1]
            if lastvalue < cutoff < value:
                midfraction = (lastvalue * plot_points[i - 1, 1] + value * fraction) / (value + lastvalue)

    if plot_points[-1, 0] < cutoff:
        plot_points = np.vstack([plot_points, np.array([cutoff, 1])])
    else:
        plot_points = np.vstack([plot_points, np.array([cutoff, midfraction])])

    sorting = np.argsort(plot_points[:, 0])
    plot_points = plot_points[sorting, :]

    auc = sklearn.metrics.auc(plot_points[plot_points[:, 0] <= cutoff, 0],
                              plot_points[plot_points[:, 0] <= cutoff, 1])
    auc = auc / cutoff

    return auc, plot_points


def inclination_from_horizon_points(A,B):
    direction = np.array(A) - np.array(B)
    angle = np.arctan2(direction[1],-direction[0])
    if angle > np.pi/2:
        angle -= np.pi
    if angle < -np.pi/2:
        angle += np.pi
    return angle/np.pi*180


def line_from_endpoints(A, B) -> np.ndarray:
    x1,y1 = A
    x2,y2 = B
    return np.cross(np.array([x1,y1,1], np.float64), np.array([x2,y2,1], np.float64))


def get_line_info(A, B, shape):
    h,w = shape
    left_border = np.array([1,0,0],np.float64)
    right_border = np.array([1,0,-w],np.float64)
    
    angle = inclination_from_horizon_points(A, B)
    h = line_from_endpoints(A, B)
    _,y1,z1 = np.cross(h, left_border)
    _,y2,z2 = np.cross(h, right_border)
    return y1/z1, y2/z2, angle


def get_keys(mapping:Mapping, *keys):
    yield from (mapping[k] for k in keys)


class Evaluator:
    def __init__(self):
        self.eval_data = dict()
        self.needs_eval = True

    def add_groundtruth(self, idx, A, B, shape):
        self.needs_eval = True
        if idx not in self.eval_data:
            self.eval_data[idx] = dict()
        y1, y2, angle = get_line_info(A, B, shape)
        gt_dict = dict(shape=shape, gt_l=y1, gt_r=y2, gt_angle=angle)
        self.eval_data[idx].update(gt_dict)

    def add_detection(self, idx, A, B):
        self.needs_eval = True
        if idx not in self.eval_data:
            raise RuntimeError("Call add_groundtruth for idx={idx} first")
        y1, y2, angle = get_line_info(A, B, self.eval_data[idx]["shape"])
        gt_dict = dict(dt_l=y1, dt_r=y2, dt_angle=angle)
        self.eval_data[idx].update(gt_dict)
        
    def evaluate(self):
        if not self.eval_data:
            raise RuntimeError("Nothing to evaluate. use add_groundtruth and add_detection first.")
        # Get results for individual images
        self.h_err_list = []
        self.a_err_list = []
        for img_idx, img_data in self.eval_data.items():
            has_gt = "gt_l" in img_data
            has_dt = "dt_l" in img_data
            if not has_gt or not has_dt:
                logging.warning(f"No groundtruth or detection for image {img_idx}. Skipping.")
                continue
            height,_ = img_data["shape"]
            gt_l, gt_r, dt_l, dt_r = get_keys(img_data, "gt_l", "gt_r", "dt_l", "dt_r")
            h_err = max(np.abs(gt_l-dt_l), np.abs(gt_r-dt_r)) / height
            gt_a, dt_a = get_keys(img_data, "gt_angle", "dt_angle")
            a_err = np.abs(gt_a - dt_a)
            self.h_err_list.append(h_err)
            self.a_err_list.append(a_err)

        logging.info(f"Evaluated {len(self.h_err_list)} samples")
        self.needs_eval = False
        return len(self.h_err_list)

    def horizon_error_auc(self, cutoff=0.25):
        if self.needs_eval:
            self.evaluate()
        return calc_auc(np.array(self.h_err_list), cutoff)

    def angular_error_auc(self, cutoff=5):
        if self.needs_eval:
            self.evaluate()
        return calc_auc(np.array(self.a_err_list), cutoff)
    
    def angular_error_ratio(self, limit=2):
        """Return the ratio of samples with angular error lower than the limit (in degrees) """
        if self.needs_eval:
            self.evaluate()
        return (np.array(self.a_err_list) < limit).sum() / len(self.a_err_list)

    def get_indices_by_horizon_error(self):
        return np.argsort(self.h_err_list)

    def get_indices_by_angular_error(self):
        return np.argsort(self.a_err_list)
