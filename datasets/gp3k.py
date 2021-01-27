from typing import Sequence
import logging
from pathlib import Path

import numpy as np
from skimage.io import imread


from xml.dom import minidom


def _parse_rotation_matrix(xml):
    doc = minidom.parse(xml.open())
    els = (doc.getElementsByTagName(elname)[0] for elname in ["C0","C1","C2"])
    data = []
    for el in els:
        data.append(np.atleast_2d(list(map(float, (el.attributes[nm].value for nm in ["x0","x1","x2"])))))
    return np.concatenate(data, axis=0)


def _parse_info(lines:Sequence[str]) -> dict:
    """Parse info.txt and return dictionary with relevant data"""
    auto = lines[0] == "AUTO"
    yaw, pitch, roll = map(float, lines[1].split(" "))
    fov = float(lines[5])
    return dict(
        auto=auto,
        fov=fov,
        yaw=yaw,
        pitch=pitch,
        roll=roll
    )

def get_normalized_point(v, focal):
    a,b,c = v
    if np.abs(c) < 1e-6:
        x = -a, b, 0
    else:
        s = c/focal
        x = -a/s, b/s, 1
    x = np.array(x)
    return x

class GeoPose3KDataset:
    def __init__(self, path):
        self.items = []
        for item in Path(path).glob("*"):
            if item.is_dir():
                photo = list(item.glob("photo*"))
                if not photo:
                    continue
                
                item_dict = dict()
                item_dict["info"] = item / "info.txt"
                item_dict["G2C"] = item / "rotationG2C.xml"
                item_dict["filename"] = photo[0]
                item_dict["cached"] = False
                
                self.items.append(item_dict)

    def _cache_info(self, idx):
        item = self.items[idx]
        if not item["cached"]:
            with open(item["info"],"rt") as f:
                item.update(_parse_info(f.read().splitlines()))
            item["R"] = _parse_rotation_matrix(item["G2C"])
            item["cached"] = True

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        self._cache_info(idx)
        item_dict = self.items[idx]
        image = imread(item_dict["filename"])

        h,w = image.shape[:2]
        pp = w/2,h/2
        focal = (w/2) / np.tan(item_dict["fov"]/2)
        u,v,_ = item_dict["R"]
        u = get_normalized_point(u, focal)
        v = get_normalized_point(v, focal)

        h = np.cross(u, v)

        A = np.cross(h, [1,0, w/2])
        B = np.cross(h, [1,0,-w/2])

        A = (A[:2] / A[2]) + pp
        B = (B[:2] / B[2]) + pp

        return dict(image=image, A=A, B=B, filename=item_dict["filename"])