### Util library for distorting via polynomials

import json
import os
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from .paths import get_data_path
from .constants import TRAIN_VAL_TEST_SPLIT

__all__ = ['get_distorted_location', 'paramType', 'get_param_encoding', 'encodings_to_params', 'get_param_split']

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

def get_param_split(params: paramType):
    return np.random.default_rng(seed=int(get_param_encoding(params))).choice(
            ['train', 'val', 'test'],
            p=TRAIN_VAL_TEST_SPLIT,
        )