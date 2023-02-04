### Generates point mappings for a set of distortion parameters
# Usage:
# ```
# cd src/data
# python generate_point_maps.py
# ```


from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Optional

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import (get_data_path, get_distorted_location, find_data)

from data_util import (get_point_map_data, get_param_encoding, get_param_split, load_hashmap_data, write_hashmap_data)
# ---------------------------------------------------------------------------- #
#                              TUNEABLE PARAMETERS                             #
# ---------------------------------------------------------------------------- #

DISTORTION_RANGES = (
    # (min, max, step)
    (-0.04, 0.041, 0.01), # k1
    (-0.04, 0.041, 0.01), # k2
    (-0.04, 0.041, 0.01), # k3
    (-0.04, 0.041, 0.01), # p1
    (-0.04, 0.041, 0.01), # p2
)

NUM_K = 3 # Must keep up to date with DISTORTION_RANGES

IMAGE_SIZE = (50, 50)

# ---------------------------------------------------------------------------- #
#                            END TUNEABLE PARAMETERS                           #
# ---------------------------------------------------------------------------- #

def _get_random_grid(minimum: float = -1, maximum: float = 1, seed: Optional[int] = None) -> np.ndarray:
    return np.random.default_rng(seed=seed).uniform(
        [minimum] * (2 * np.prod(IMAGE_SIZE)),
        [maximum] * (2 * np.prod(IMAGE_SIZE))
        ).reshape((2,) + tuple(dim for dim in IMAGE_SIZE))  # type: ignore

def _distortion_parameter_generator() -> itertools.product[tuple[float,...]]:
    return itertools.product(*[np.arange(*r) for r in DISTORTION_RANGES])

def _create_and_save_point_maps(params, encoding, split, X, Y, X_distorted, Y_distorted):
    # Save to file
    print(f'Saving {params} to {encoding} as {split} set')
    np.savetxt(get_data_path('point_maps', split, f'{encoding}.gz'), np.stack((X_distorted, Y_distorted, X, Y), axis=2).reshape((np.prod(IMAGE_SIZE), 4)))

# function call
if __name__ == '__main__':
    print('Generating point mappings...')
    hash_to_params = load_hashmap_data()
    i = 0
    try:
        start = time.time()
        with mp.Pool(mp.cpu_count() - 1) as P:
            results = []
            for params in _distortion_parameter_generator():
                i += 1
                encoding = get_param_encoding(params)
                # Generate random grid of distorted points given undistorted pixels
                X, Y = _get_random_grid(seed=int(encoding)) # Seed with parameters to get same outcome every time run on same grid.
                X_distorted, Y_distorted = get_distorted_location(X, Y, params[:NUM_K], params[NUM_K:])
                # Randomly determine if this set is in test/train/val set
                split = get_param_split(params)
                results.append(P.apply_async(_create_and_save_point_maps, args=(params, encoding, split, X, Y, X_distorted, Y_distorted)))
                hash_to_params[encoding] = {'K': params[:NUM_K], 'P': params[NUM_K:], 'split': split}
            for r in results:
                r.get()
        print(f'{i} processes completed in {time.time() - start} seconds')
    except KeyboardInterrupt:
        print('Keyboard interrupt detected, saving hash_to_params.json...')
    # Save hash to params dictionary
    write_hashmap_data(hash_to_params)
    if len(hash_to_params) != i:
        print(f'WARNING: Hash collision detected. Missing {i - len(hash_to_params)} parameters.')

    # Check size of training + val data and see if it fits within 20GB = 20 * 1024 * 1024 * 1024 bytes
    train_val_size = 0
    for split in ['train', 'val']:
        for filename in find_data('point_maps', split):
            train_val_size += os.path.getsize(filename)
    print(f'Training and validation dataset takes up {train_val_size} bytes or {train_val_size / (1024 * 1024 * 1024)} GB')
    if train_val_size >= 20 * 1024 * 1024 * 1024:
        print('WARNING: Data surpasses 20 GB')
