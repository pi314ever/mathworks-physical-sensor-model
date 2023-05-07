### Util library for distorting via polynomials

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.typing import paramType


def get_distorted_location(
    X: ArrayLike, Y: ArrayLike, K: paramType, P: paramType, x0: float = 0, y0: float = 0
) -> Tuple[ArrayLike, ArrayLike]:
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


def distort_radial(
    X: ArrayLike, Y: ArrayLike, K: paramType, x0: float = 0, y0: float = 0
) -> Tuple[ArrayLike, ArrayLike]:
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
    X_til, Y_til = X - x0, Y - y0  # type: ignore
    R2 = X_til**2 + Y_til**2
    # Radial distortion
    for i, k in enumerate(K):
        radial += k * R2 ** (i + 1)
        radial_max += k * np.sqrt(2) ** (i + 1)
    X_radial = radial * X_til * (1 / radial_max)
    Y_radial = radial * Y_til * (1 / radial_max)
    return X_radial, Y_radial


def distort_tangential(
    X: ArrayLike, Y: ArrayLike, P: Tuple[float, float], x0: float = 0, y0: float = 0
) -> Tuple[ArrayLike, ArrayLike]:
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
    X_til, Y_til = X - x0, Y - y0  # type: ignore
    R2 = X_til**2 + Y**2  # type: ignore
    X_tangential = (
        X_til + (2 * P[0] * X_til * Y_til + P[1] * (R2 + 2 * X_til**2))
    ) / x_scale
    Y_tangential = (
        Y_til + (P[0] * (R2 + 2 * Y_til**2) + 2 * P[1] * X_til * Y_til)
    ) / y_scale
    return X_tangential, Y_tangential


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate grid of points
    X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    X, Y = X.flatten(), Y.flatten()
    idx = list(range(len(X)))
    np.random.default_rng(100).shuffle(idx)
    print(idx[:10])
    X, Y = X[idx], Y[idx]

    # Generate distortion parameters
    K = (2.0, 0, 0)
    P = (0, 0)

    # Distort points
    X_dist, Y_dist = get_distorted_location(X, Y, K, P)
    X_corners, Y_corners = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])
    X_corners_dist, Y_corners_dist = get_distorted_location(X_corners, Y_corners, K, P)

    # Plot
    fig = plt.figure()
    plt.scatter(X_dist[:500], Y_dist[:500], c="r")
    plt.scatter(X_corners_dist, Y_corners_dist, c=["b", "g", "y", "k"])
    plt.show()
