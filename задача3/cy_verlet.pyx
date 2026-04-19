# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel cimport prange

cdef double G = 39.478

cdef void get_acc(double[:, :] r, double[:] m, double[:, :] a, int n) nogil:
    cdef int i, j
    cdef double dx, dy, r2, dist, f
    
    for i in prange(n, nogil=True):
        a[i, 0] = 0.0
        a[i, 1] = 0.0
        for j in range(n):
            if i != j:
                dx = r[j, 0] - r[i, 0]
                dy = r[j, 1] - r[i, 1]
                r2 = dx*dx + dy*dy
                dist = sqrt(r2)
                f = (G * m[j]) / (r2 * dist)
                a[i, 0] += f * dx
                a[i, 1] += f * dy

def run_cython(double[:, :] r0, double[:, :] v0, double[:] m, double t0, double t1, double dt):
    cdef int n = r0.shape[0]
    cdef int steps = int((t1 - t0) / dt)
    cdef int s, i

    cdef double[:, :] r = np.copy(r0)
    cdef double[:, :] v = np.copy(v0)
    cdef double[:, :] a = np.zeros((n, 2), dtype=np.float64)
    cdef double[:, :] a_new = np.zeros((n, 2), dtype=np.float64)
    
    out = np.zeros((steps, n, 2), dtype=np.float64)
    cdef double[:, :, :] out_view = out

    get_acc(r, m, a, n)

    for s in range(steps):
        for i in prange(n, nogil=True):
            r[i, 0] += v[i, 0] * dt + 0.5 * a[i, 0] * dt * dt
            r[i, 1] += v[i, 1] * dt + 0.5 * a[i, 1] * dt * dt

        get_acc(r, m, a_new, n)

        for i in prange(n, nogil=True):
            v[i, 0] += 0.5 * (a[i, 0] + a_new[i, 0]) * dt
            v[i, 1] += 0.5 * (a[i, 1] + a_new[i, 1]) * dt
            
            a[i, 0] = a_new[i, 0]
            a[i, 1] = a_new[i, 1]
            
            out_view[s, i, 0] = r[i, 0]
            out_view[s, i, 1] = r[i, 1]

    return out