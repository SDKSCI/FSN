# merge_superpxiel.pyx
cimport numpy as np
from libcpp.vector cimport vector
import numpy as np

cdef extern from "merge.cpp":
    cdef struct RGB:
        unsigned char R
        unsigned char G
        unsigned char B

    vector[int] getlabel(RGB *rgb_data, int* label_ptr, int w, int h, int n)

def getsuperpixle(np.ndarray[np.uint8_t, ndim=3] img, np.ndarray[np.int32_t, ndim=1] label, int num):
    cdef int width = img.shape[1]
    cdef int height = img.shape[0]

    # 确保 img 是 C-contiguous
    assert img.flags.c_contiguous, "img must be C-contiguous"
    assert label.flags.c_contiguous, "label must be C-contiguous"
    assert img.shape[2] == 3, "img must have 3 channels"

    cdef RGB *lab_data = <RGB *> img.data
    cdef int* label_ptr = <int*> label.data

    cdef vector[int] out_label = getlabel(lab_data, label_ptr, width, height, num)

    return np.array(out_label).reshape(height,width)

