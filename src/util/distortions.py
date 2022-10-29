### Util library for distorting via polynomials

import numpy as np
from typing import List, Tuple, Union
from numpy.typing import NDArray

__all__ = ['get_distorted_location', 'paramType', 'get_param_encoding']

paramType = Union[List[float], Tuple[float,...]]

def get_distorted_location(X: NDArray, Y: NDArray, K: paramType, P: paramType, x0: float = 0, y0: float = 0) -> Tuple[NDArray, NDArray]:
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
    radial, tangential = 0, 1
    X_til, Y_til = X - x0, Y - y0
    R2 = X_til** 2 + Y_til** 2
    for i, k in enumerate(K):
        radial += k * R2**(i + 1)
    if len(P) > 2:
        for i, p in enumerate(P[2:]):
            tangential += p * R2 **(i + 1)
    X_distorted = X + radial * X_til
    X_distorted += tangential * (P[0] * (R2 + 2 * X_til** 2) + 2 * P[1] * X_til * Y_til)
    Y_distorted = Y + radial * Y_til
    Y_distorted += tangential * (P[1] * (R2 + 2 * Y_til** 2) + 2 * P[0] * X_til * Y_til)
    return X_distorted, Y_distorted

def get_param_encoding(params: tuple[float,...]) -> str:
    """
    Generates a string encoding from a set of distortion parameters

    Args:
        params (tuple[float]): Distortion parameters

    Returns:
        str: Filename
    """
    return str(abs(hash(params)))
    # return sha256(str(params).encode()).hexdigest()