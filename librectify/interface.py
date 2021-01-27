import ctypes
from logging import error
import platform
import numpy as np
from pathlib import Path
from skimage.transform import rescale
from skimage.filters import gaussian


dll_name = {
    "Windows": "librectify.dll",
    "Linux": "librectify.so",
}

dll_path = Path(__file__).parent / dll_name[platform.system()]

dll = ctypes.cdll.LoadLibrary(dll_path.as_posix())
find_lines = dll.find_line_segment_groups


class LineSegment(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("weight", ctypes.c_float),
        ("err", ctypes.c_float),
        ("group_id", ctypes.c_int)]


c_int32_p = ctypes.POINTER(ctypes.c_int32)
c_float_p = ctypes.POINTER(ctypes.c_float)
linesegments_p = ctypes.POINTER(LineSegment)

find_lines.restype = linesegments_p
find_lines.argtypes = [
    c_float_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_bool,  # min_length, refine
    ctypes.c_int,  # num threads
    c_int32_p
]

dll.release_line_segments.argtypes = [
    ctypes.POINTER(linesegments_p)
]

def detect_line_segments(image:np.ndarray, max_size=1200, smooth=0):
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    ndim = image.ndim
    if ndim not in [2,3]:
        raise ValueError("Image must be 2 or 3 dimensioanl")
    if ndim == 3:
        image = np.mean(image, axis=-1)

    #print(image.shape, image.max(), image.dtype)

    scale = max_size / float(max(image.shape))
    #print(scale)

    if scale < 1:
        image = rescale(image, scale, order=1, anti_aliasing=False, preserve_range=True)

    #print(image.shape, image.max(), image.dtype)

    if smooth > 0:
        image = gaussian(image, smooth)
    
    #print(image.shape, image.max(), image.dtype)

    image = np.ascontiguousarray(image, np.float32)

    #print(image.shape, image.max(), image.dtype)

    buffer = image.ctypes.data_as(c_float_p)
    h,w = image.shape
    stride = image.strides[0] // 4

    # print(h,w,stride)
    # run lib code
    n_lines = np.empty((1,),"i")
    lines_p = find_lines(buffer, w, h, stride, 10.0, False, 1, n_lines.ctypes.data_as(c_int32_p))
    
    # get results as list of tuples
    n = n_lines[0]
    #print(n_lines)

    lines = np.empty((n,4), np.float32)
    weights = np.empty(n, np.float32)
    errors = np.empty(n, np.float32)
    groups = np.empty(n, np.int32)

    for i in range(n):
        l = lines_p[i]
        lines[i,:] = (l.x1, l.y1, l.x2, l.y2)
        weights[i] = l.weight
        errors[i] = l.err
        groups[i] = l.group_id

    if scale < 1:
        lines /= scale

    #print(lines)

    dll.release_line_segments(ctypes.pointer(lines_p))
    
    return lines, dict(weight=weights, error=errors, group=groups)
