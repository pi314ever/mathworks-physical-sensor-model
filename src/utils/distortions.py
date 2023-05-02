### Util library for distorting via polynomials

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from .typing import paramType

def get_distorted_location(X: ArrayLike, Y: ArrayLike, K: paramType, P: paramType, x0: float = 0, y0: float = 0) -> Tuple[ArrayLike, ArrayLike]:
    """
    Generates distortion location for each X, Y location

    Args:
        X (ArrayLike): X coordinates of each pixel
        Y (ArrayLike): Y coordinates of each pixel
        K (Iterable[float]): Radial distortion coefficients
        P (Iterable[float]): Tangential distortion coefficients, must be of length at least 2
        x0 (float, optional): Center of distortion (x). Defaults to 0.
        y0 (float, optional): Center of distortion (y). Defaults to 0.

    Returns:
        Tuple[ArrayLike, ArrayLike]: Two arrays of distorted X and Y coordinates
    """
    X_r, Y_r = distort_radial(X, Y, K)
    X_dist, Y_dist = distort_tangential(X_r, Y_r, P)
    return X_dist, Y_dist

def distort_radial(X: ArrayLike, Y: ArrayLike, K: paramType, x0: float=0, y0: float=0) -> Tuple[ArrayLike, ArrayLike]:
    """
    Generates radial distortion for each X, Y location. Each coordinate must be bounded by 0 <= x-x0,y-y0 <= 1

    Args:
        X (ArrayLike): X coordinates of each pixel
        Y (ArrayLike): Y coordinates of each pixel
        K (Iterable[float]): Radial distortion coefficients
        x0 (float, optional): Center of distortion (x). Defaults to 0.
        y0 (float, optional): Center of distortion (y). Defaults to 0.

    Returns:
        Tuple[ArrayLike, ArrayLike]: Two arrays of distorted X and Y coordinates
    """
    radial, radial_max = 1, 1
    X_til, Y_til = X - x0, Y - y0 # type: ignore
    R2 = X_til** 2 + Y_til** 2
    # Radial distortion
    for i, k in enumerate(K):
        radial += k * R2**(i + 1)
        radial_max += k * np.sqrt(2) ** (i + 1)
    X_radial = radial * X_til * (1 / radial_max)
    Y_radial = radial * Y_til * (1 / radial_max)
    return X_radial, Y_radial

def distort_tangential(X: ArrayLike, Y: ArrayLike, P: Tuple[float, float], x0: float=0, y0: float=0) -> Tuple[ArrayLike, ArrayLike]:
    """
    Converts X, Y points using a tangential distortion function. Only works for 2 tangential parameters and 0 <= x-x0,y-y0 <= 1

    Args:
        X (ArrayLike): X coordinates of each pixel
        Y (ArrayLike): Y coordinates of each pixel
        P (paramType): Tangential distortion parameters
        x0 (float, optional): Optical center, x. Defaults to 0.
        y0 (float, optional): Optical center, y. Defaults to 0.

    Returns:
        Tuple[ArrayLike, ArrayLike]: X, Y distorted points
    """
    tangential = 1
    x_scale = 1 + (2 * P[0] + 4 * P[1])
    y_scale = 1 + (4 * P[0] + 2 * P[1])
    X_til, Y_til = X - x0, Y - y0 # type: ignore
    R2 = X_til **2 + Y**2 # type: ignore
    X_tangential = (X_til + (2 * P[0] * X_til * Y_til + P[1] * (R2 + 2 * X_til **2) )) / x_scale
    Y_tangential = (Y_til + (P[0] * (R2 + 2 * Y_til **2) + 2 * P[1] * X_til * Y_til)) / y_scale
    return X_tangential, Y_tangential