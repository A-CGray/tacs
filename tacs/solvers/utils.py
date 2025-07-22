import numpy as np


def lagrangeInterp(xKnown, yKnown, xQuery, yQuery):
    """Interpolate an array using lagrange polynomials

    Parameters
    ----------
    xKnown : iterable of length n
        scalar x values of known points
    yKnown : iterable of n np.ndarrays
        arrays at known points
    xQuery : float
        x value to interpolate at
    yQuery : np.ndarray
        array to store interpolated result in
    """
    numPoints = len(xKnown)
    yQuery[:] = 0.0
    yTemp = np.zeros_like(yQuery)
    for jj in range(numPoints):
        yTemp[:] = 1.0
        for mm in range(numPoints):
            if mm != jj:
                yTemp[:] = yTemp[:] * (xQuery - xKnown[mm]) / (xKnown[jj] - xKnown[mm])
        yQuery[:] = yQuery[:] + yKnown[jj] * yTemp
