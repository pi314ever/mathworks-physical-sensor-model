### Generates point mappings for a set of distortion parameters
# Usage:
# ```
# cd src/data
# python generate_point_maps.py
# ```


from __future__ import annotations

import itertools
import json
import os
import sys
from typing import Optional

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util import get_distorted_location, get_param_encoding

# ---------------------------------------------------------------------------- #
#                              TUNEABLE PARAMETERS                             #
# ---------------------------------------------------------------------------- #

DISTORTION_RANGES = (
    # (min, max, step)
    (-0.05, 0.05, 0.005), # k1
    (-0.05, 0.05, 0.005), # k2
    (-0.05, 0.05, 0.005), # k3
    (-0.05, 0.05, 0.005), # p1
    (-0.05, 0.05, 0.005), # p2
)

NUM_K = 3 # Must keep up to date with DISTORTION_RANGES

IMAGE_SIZE = (5, 5)

# ---------------------------------------------------------------------------- #
#                            END TUNEABLE PARAMETERS                           #
# ---------------------------------------------------------------------------- #

def _get_random_grid(minimum: float = 0, maximum: float = 1, seed: Optional[int] = None) -> np.ndarray:
    return np.random.default_rng(seed=seed).uniform(
        [minimum] * (2 * np.prod(IMAGE_SIZE)),
        [maximum] * (2 * np.prod(IMAGE_SIZE))
        ).reshape((2,) + tuple(dim for dim in IMAGE_SIZE))  # type: ignore

def _distortion_parameter_generator() -> itertools.product[tuple[float,...]]:
    return itertools.product(*[np.arange(*r) for r in DISTORTION_RANGES])

# function call
if __name__ == '__main__':
    print('Generating point mappings...')
    data_path = os.path.abspath(os.path.join(SCRIPT_DIR, '../../data'))
    with open(os.path.join(data_path, 'hash_to_params.json'), 'r') as f:
        try:
            hash_to_params = json.loads(f.read())
        except:
            hash_to_params = {}
    for params in _distortion_parameter_generator():
        # Generate random grid of distorted points given undistorted pixels
        encoding = get_param_encoding(params)
        X, Y = _get_random_grid(seed=int(encoding)) # Seed with parameters to get same outcome every time run on same grid.
        X_distorted, Y_distorted = get_distorted_location(X, Y, params[:NUM_K], params[NUM_K:])
        # Save to file
        print(f'Saving {params} to {encoding}')
        if encoding not in hash_to_params:
            hash_to_params[encoding] = {'K': params[:NUM_K], 'P': params[NUM_K:]}
        np.savetxt(os.path.join(data_path, 'point_maps', f'{encoding}.gz'), np.stack((X_distorted, Y_distorted, X, Y), axis=2).reshape((np.prod(IMAGE_SIZE), 4)))
    with open(os.path.join(data_path, 'hash_to_params.json'), 'w') as f:
        f.write(json.dumps(hash_to_params))

