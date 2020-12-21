import json
import os
import random
from collections import deque
from itertools import cycle, chain, islice, repeat, tee
from pathlib import Path

import cv2
import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables import Batch, KeypointsOnImage
from imgaug.augmenters import *
from .utils import Normalizer


def center_crop(output_shape=(224,224)):
    h,w = output_shape
    return Sequential( [
        Resize({"shorter-side":max(h,w), "longer-side":"keep-aspect-ratio"},interpolation=cv2.INTER_AREA),
        CropToFixedSize(width=w,height=h,position="center")
    ], random_order=False)


def random_crop(output_shape):
    height, width = output_shape
    size = min(output_shape)
    return Sequential([
        Sequential(
            [
                GammaContrast( (0.5,2) ),
                Affine(rotate=(-30,30), mode="reflect", order=3),
                Sometimes(0.3, Sharpen( (0.1,0.5) ) ),
                Sometimes(0.3, GaussianBlur( (0.5,1) ) ),
            ], random_order=True),
        Flipud(0.2),
        #Resize({"shorter-side":(size, 2*size), "longer-side":"keep-aspect-ratio"},interpolation="nearest"),
        CropToFixedSize(width=width,height=height),
        Sometimes(0.3, AdditiveGaussianNoise((2,10)) ),
        Sometimes(0.1, Grayscale()),
        Sometimes(0.1, Cutout(size=(0.1,0.2), nb_iterations=(1,3), fill_mode="gaussian", fill_per_channel=True)),
    ],
    random_order=False)

     

def batch_generator(iterator, augmenter, batch_size=16):
    try:
        while True:
            imgs,kps = zip(*(item for item in islice(iterator, batch_size)))
            yield augmenter.augment_batch_(Batch(images=imgs, keypoints=kps))
    except StopIteration:
        return


def homogenous(kps, center=(0,0)):
    A,B = kps.keypoints
    u,v = center
    h = np.cross([A.x-u,A.y-v,1],[B.x-u,B.y-v,1])
    return h


def one_hot(x, n_bins, sigma=1):
    y = np.zeros((x.size, n_bins),"f")
    for i,b in enumerate(x):
        y[i,b] = 1
    y /= y.sum(axis=-1,keepdims=True)
    return y

def anchor_normal(kps):
    A,B = kps.keypoints
    h = np.cross([A.x-112,A.y-112,1],[B.x-112,B.y-112,1])
    hp = np.array([-h[1],h[0],0])
    a = np.cross(h,hp)
    a = a[:2] / a[2]
    n = h[:2] / np.linalg.norm(h[:2])
    return a/112, n  

from tensorflow.keras.utils import to_categorical
from scipy.ndimage import gaussian_filter1d

def categorical_theta_rho(seq, t_bins = 256, r_bins = 128, center=(0,0)):
    t_bin_map = Normalizer((0,np.pi), (0, t_bins))
    d_bin_map = Normalizer((-np.pi/2,np.pi/2), (0, r_bins))
    rho_scale = 400
    for batch in seq:
        # Images
        X = np.array(batch.images_aug, np.float32) / 256   # (N,H,W,3)

        # Homogenous coords of horizon
        h = np.array([homogenous(kps, center=center) for kps in batch.keypoints_aug])
        h /= np.linalg.norm(h[:,:2], axis=1, keepdims=True)
        neg_b = h[:,1:2] < 0
        np.multiply(h, -1,  where=neg_b, out=h)

        # Orientation bin
        theta = np.arctan2(h[:,1], h[:,0])
        theta_bin = np.floor(t_bin_map(theta)).astype("i")
        #print(theta_bin)

        # Distance bin
        dist = np.arctan(h[:,2] / rho_scale)
        dist_bin = np.floor(d_bin_map(dist)).astype("i")

        theta_bin = to_categorical(theta_bin, t_bins)
        gaussian_filter1d(theta_bin, sigma=1, output=theta_bin)
        theta_bin /= theta_bin.max(axis=1, keepdims=True)

        dist_bin = to_categorical(dist_bin, r_bins)
        gaussian_filter1d(dist_bin, sigma=1, output=dist_bin)
        dist_bin /= dist_bin.max(axis=1, keepdims=True)

        yield X.astype(np.float32), [theta_bin, dist_bin]


def line_segments_from_homogeneous(lines, bbox):
    x,y,w,h = bbox
    
    # Corner points
    A = np.array([x,y,1])
    B = np.array([x+w,y,1])
    C = np.array([x+w,y+h,1])
    D = np.array([x,y+h,1])

    # Cross product of pairs of corner points
    edges = [
        np.cross(a,b) for a,b in [[A,B],[B,C],[C,D],[D,A]]
    ]

    # Cross product of line params with edges
    intersections = [
        np.cross(lines, e) for e in edges
    ]

    # Normalize
    normalized = [
        p[:,:2] / p[:,-1].reshape(-1,1) for p in intersections
    ]

    X = []
    Y = []
    
    for p in zip(*normalized):
        P = []
        for (u,v) in p:
            if (x <= u <= x+w) and (y <= v <= y+h):
                P.append( (u,v) )
        if len(P) == 2:
            (x0,y0), (x1,y1) = P
            X.append( (x0,x1) )
            Y.append( (y0,y1) )
        else:
            X.append(None)
            Y.append(None)

    return X, Y