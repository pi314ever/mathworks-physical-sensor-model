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

from utils import get_distorted_location, get_param_encoding, get_data_path, get_param_split

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

IMAGE_SIZE = (256, 256)

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
    with open(get_data_path('hash_to_params.json'), 'r') as f:
        try:
            hash_to_params = json.loads(f.read())
        except:
            hash_to_params = {}
    for params in _distortion_parameter_generator():
        # Generate random grid of distorted points given undistorted pixels
        encoding = get_param_encoding(params)
        X, Y = _get_random_grid(seed=int(encoding)) # Seed with parameters to get same outcome every time run on same grid.
        X_distorted, Y_distorted = get_distorted_location(X, Y, params[:NUM_K], params[NUM_K:])
        # Randomly determine if this set is in test/train/val set
        split = get_param_split(params)
        # Save to file
        print(f'Saving {params} to {encoding} as {split} set')
        if encoding not in hash_to_params:
            hash_to_params[encoding] = {'K': params[:NUM_K], 'P': params[NUM_K:], 'split': split}
        np.savetxt(get_data_path('point_maps', split, f'{encoding}.gz'), np.stack((X_distorted, Y_distorted, X, Y), axis=2).reshape((np.prod(IMAGE_SIZE), 4)))
    # Save hash to params dictionary
    with open(get_data_path('hash_to_params.json'), 'w') as f:
        f.write(json.dumps(hash_to_params))

