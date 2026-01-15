import numpy as np
cimport numpy as np
cimport cython
import torch
@cython.boundscheck(False)
@cython.wraparound(False)

def decode_segmap_cython(short[:, :] mask, unsigned char[:, :] palette):
    cdef int h = mask.shape[0]
    cdef int w = mask.shape[1]
    cdef int i, j, class_id

    # On crée l'image de sortie
    cdef np.ndarray[np.uint8_t, ndim=3] rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            class_id = mask[i, j]
            # On remplit les canaux R, G, B directement
            rgb_img[i, j, 0] = palette[class_id, 0]
            rgb_img[i, j, 1] = palette[class_id, 1]
            rgb_img[i, j, 2] = palette[class_id, 2]

    return rgb_img

