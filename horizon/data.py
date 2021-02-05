import random
from collections import deque
from itertools import count, islice
from queue import deque
from typing import Any, Sequence

import numpy as np
from imgaug.augmentables import Keypoint, KeypointsOnImage, HeatmapsOnImage, heatmaps
from imgaug.augmenters import *
from skimage.transform import rescale

from .sord import soft_label_rho, soft_label_theta
from .utils import Normalizer


def center_crop(output_shape=(224,224)):
    h,w = output_shape
    return Sequential( [
        Resize({"shorter-side":max(h,w), "longer-side":"keep-aspect-ratio"},interpolation=cv2.INTER_AREA),
        CropToFixedSize(width=w,height=h,position="center")
    ], random_order=False)


def random_crop(output_shape=(224,224)):
    height, width = output_shape
    size = min(output_shape)
    return Sequential([
        #Resize({"shorter-side":(size, 1.5*size), "longer-side":"keep-aspect-ratio"},interpolation="nearest"),
        Sequential(
            [
                GammaContrast( (0.5,2) ),
                Rotate(mode="reflect"),
                #Affine(rotate=(-30,30), mode="reflect", order=2),
                Sometimes(0.3, Sharpen( (0.1,0.5) ) ),
                Sometimes(0.3, GaussianBlur( (0.5,1) ) ),
            ], random_order=True),
        Fliplr(0.5),
        CropToFixedSize(width=width,height=height),
        Sometimes(0.3, AdditiveGaussianNoise((1,5)) ),
        Sometimes(0.1, Grayscale()),
        Sometimes(0.1, Cutout(size=(0.1,0.2), nb_iterations=(1,3), fill_mode="gaussian", fill_per_channel=True)),
    ],
    random_order=False)



def homogenous(kps, center=(0,0)):
    A,B = kps.keypoints
    u,v = center
    h = np.cross([A.x-u,A.y-v,1],[B.x-u,B.y-v,1])
    return h


def prescale_image(new_image, size_range):
     min_size, max_size = size_range
     img = new_image["image"]
     A,B = new_image["A"], new_image["B"]
     size = min(img.shape[:2])
     new_size = np.random.uniform(min_size, max_size)
     scale = new_size / size
     img = rescale(img, scale, anti_aliasing=True, preserve_range=True, multichannel=True).astype("u1")
     out_dict = dict(image=img, A=A*scale, B=B*scale)
     if "masks" in new_image:
         masks = rescale(new_image["masks"], scale, anti_aliasing=False, preserve_range=True, multichannel=True)
         out_dict.update(masks=masks)
     return out_dict


def get_image_and_keypoints(image_dict):
    (x1,y1), (x2,y2) = image_dict["A"], image_dict["B"]
    image = image_dict["image"]
    shape = image.shape[:2]
    kps = KeypointsOnImage([Keypoint(x=x1,y=y1), Keypoint(x=x2,y=y2)], shape=shape)
    heatmaps = None
    if "masks" in image_dict: # Optionally return heatmeps
        heatmaps = HeatmapsOnImage(image_dict["masks"], shape)
    return image, kps, heatmaps


def batch_from_dicts(image_dicts, augmenter):
    image_data = (get_image_and_keypoints(x) for x in image_dicts)
    images, kps, heatmaps = zip(*image_data)

    images_aug, kps_aug, heatmaps_aug = augmenter.augment(images=images, keypoints=kps, heatmaps=heatmaps)

    horizons = np.array([homogenous(k, center=(112,112)) for k in kps_aug])   # (B, 3)
    norm = np.linalg.norm(horizons[:,:2], axis=-1, keepdims=True)
    horizons /= norm
    v = horizons[:,1] < 0
    horizons[v,:] *= -1
    theta = np.arctan2(-horizons[:,0],horizons[:,1])
    
    rho = horizons[:,2]
    
    soft_theta = soft_label_theta(theta, n_bins=100, K=4)                    
    soft_rho = soft_label_rho(rho, n_bins=100, K=0.05, K_range=100)
    x_images = np.array(images_aug,"f")/256
    
    x_heatmaps = np.array([x.arr_0to1 for x in heatmaps_aug],"f")

    return np.concatenate([x_images, x_heatmaps],axis=-1), [soft_theta, soft_rho]


def batch_generator(reader, augmenter, batch_size=16, stream_window=16, batches_per_window=32):
    img_queue = deque(maxlen=stream_window)
    for img_dict in reader:
        img_dict = prescale_image(img_dict, (230,260))
        #print(f"Adding new image, {img_dict['image'].shape}")
        img_queue.append(img_dict)  # add new image to the queue
        for k in range(batches_per_window):  # generate N batches with the images in the queue without reading new one
            image_dicts = (random.choice(img_queue) for _ in range(batch_size))
            yield batch_from_dicts(image_dicts, augmenter)

### Deprecated

from scipy.ndimage import gaussian_filter1d
from tensorflow.keras.utils import to_categorical


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
