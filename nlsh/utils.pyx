import cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
cdef np.int16_t binarr_to_int(int[:] binarr) nogil:
    cdef np.int32_t out = 0
    cdef np.int32_t bit
    cdef int i
    cdef int n_bits = binarr.shape[0]
    for i in range(n_bits):
        bit = binarr[i]
        out = (out << 1) | bit
    return out


@cython.boundscheck(False)
def hash_codes(int[:, :, :] codes):
    cdef int i, j
    cdef list hash_results = []
    cdef set hashes
    cdef int n_codes = codes.shape[0]
    cdef int n_samples = codes.shape[1]
    cdef int[:] binarr
    for i in range(n_codes):
        hashes = set()
        for j in range(n_samples):
            binarr = codes[i, j, :]
            hashes.add(binarr_to_int(binarr))
        hash_results.append(hashes)
    return hash_results
