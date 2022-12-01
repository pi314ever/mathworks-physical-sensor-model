### Util library for distorting via polynomials

import json
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from .paths import get_data_path
from .constants import TRAIN_VAL_TEST_SPLIT

__all__ = ['get_distorted_location', 'paramType', 'get_param_encoding', 'encodings_to_params', 'get_param_split', 'distort_radial', 'distort_tangential']

paramType = Tuple[float,...]

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
    X_r, Y_r = distort_radial(X, Y, K)
    X_dist, Y_dist = distort_tangential(X_r, Y_r, P)
    return X_dist, Y_dist

def distort_radial(X: NDArray, Y: NDArray, K: paramType, x0: float=0, y0: float=0) -> Tuple[NDArray, NDArray]:
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
    X_til, Y_til = X - x0, Y - y0
    R2 = X_til** 2 + Y_til** 2
    # Radial distortion
    for i, k in enumerate(K):
        radial += k * R2**(i + 1)
        radial_max += k * np.sqrt(2) ** (i + 1)
    X_radial = radial * X_til * (1 / radial_max)
    Y_radial = radial * Y_til * (1 / radial_max)
    return X_radial, Y_radial

def distort_tangential(X: NDArray, Y: NDArray, P: Tuple[float, float], x0: float=0, y0: float=0) -> Tuple[NDArray, NDArray]:
    """
    Converts X, Y points using a tangential distortion function. Only works for 2 tangential parameters and 0 <= x-x0,y-y0 <= 1

    Args:
        X (NDArray): X coordinates of each pixel
        Y (NDArray): Y coordinates of each pixel
        P (paramType): Tangential distortion parameters
        x0 (float, optional): Optical center, x. Defaults to 0.
        y0 (float, optional): Optical center, y. Defaults to 0.

    Returns:
        Tuple[NDArray, NDArray]: X, Y distorted points
    """
    tangential = 1
    x_scale = 1 + (2 * P[0] + 4 * P[1])
    y_scale = 1 + (4 * P[0] + 2 * P[1])
    X_til, Y_til = X - x0, Y - y0
    R2 = X_til **2 + Y**2
    X_tangential = (X_til + (2 * P[0] * X_til * Y_til + P[1] * (R2 + 2 * X_til **2) )) / x_scale
    Y_tangential = (Y_til + (P[0] * (R2 + 2 * Y_til **2) + 2 * P[1] * X_til * Y_til)) / y_scale
    return X_tangential, Y_tangential

def get_param_encoding(params: tuple[float,...]) -> str:
    """
    Generates a string encoding from a set of distortion parameters

    Args:
        params (tuple[float]): Distortion parameters

    Returns:
        str: Filename
    """
    return str(abs(hash(params)))

def encodings_to_params(encodings: Union[str, List[str]]) -> tuple[Union[paramType, list[paramType], None], Union[paramType, list[paramType], None]]:
    """
    Converts a string encoding or a list of string encodings into a set of distortion parameters

    Args:
        encodings (str | List[str]): Encodings for distortion parameters

    Returns:
        tuple[K, P]: K and P either a set of parameters or a list of sets of parameters, depending on the input
    """
    # Load encoding mapping
    with open(get_data_path('hash_to_params.json'), 'r') as f:
        try:
            hash_to_params = json.loads(f.read())
        except:
            hash_to_params = {}

    # Case: encodings is a list
    if isinstance(encodings, list):
        K, P = [], []
        for encoding in encodings:
            if encoding not in hash_to_params:
                print(f'Encoding {encoding} not found in hash_to_params.json')
                continue
            K.append(hash_to_params[encoding]['K'])
            P.append(hash_to_params[encoding]['P'])

    # Case: Single encoding
    if encodings not in hash_to_params:
        print(f'Encoding {encodings} not found in hash_to_params.json')
        return None, None
    return hash_to_params[encodings]['K'], hash_to_params[encodings]['P']

def get_param_split(params: paramType) -> str:
    return np.random.default_rng(seed=int(get_param_encoding(params))).choice(
            ['train', 'val', 'test'],
            p=TRAIN_VAL_TEST_SPLIT,
        )