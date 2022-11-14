### Utility functions for extracting data from data files
import json
from typing import TypedDict
import ntpath

import numpy as np
from numpy.typing import NDArray

from .paths import get_data_path, find_data

__all__ = ['load_hashmap_data', 'get_point_map_data', 'write_hashmap_data']


class HashDictType(TypedDict):
    K: tuple[float,...]
    P: tuple[float,...]
    split: str

def load_hashmap_data() -> dict[str, HashDictType]:
    with open(get_data_path('hash_to_params.json'), 'r') as f:
        try:
            hash_to_params: dict[str, HashDictType] = json.loads(f.read())
        except:
            hash_to_params = {}
    return hash_to_params

def write_hashmap_data(hash_to_params: dict[str, HashDictType]):
    with open(get_data_path('hash_to_params.json'), 'w') as f:
        f.write(json.dumps(hash_to_params))

def get_point_map_data(split: str) -> tuple[NDArray, NDArray]:
    """
    Gathers all point map data from given split.

    Args:
        split (str): 'test', 'train', or 'val'

    Returns:
        list[...]: list[(K, P, X_distorted, Y_distorted, X, Y)]
        NDArray, NDArray: Input, Label
    """
    # Load hashmap data
    if split not in ['test', 'train', 'val']:
        raise ValueError(f'Invalid split: {split}')
    hash_to_params = load_hashmap_data()
    # Load point map data
    data = []
    input = []
    label = []
    for file_path in find_data('point_maps', split):
        point_map_data = np.loadtxt(file_path)
        encoding = ntpath.basename(file_path).split('.')[0]
        params = hash_to_params[encoding]
        data.append((params['K'], params['P'], point_map_data[:, 0], point_map_data[:, 1], point_map_data[:, 2], point_map_data[:, 3]))
        N = point_map_data.shape[0]
        for i in range(N):
            input.append([
                point_map_data[i, 0],
                point_map_data[i, 1],
                params['K'][0],
                params['K'][1],
                params['K'][2],
                params['P'][0],
                params['P'][1]
            ])
            label.append([
                point_map_data[i, 2],
                point_map_data[i, 3]
            ])
    return np.array(input), np.array(label)

if __name__ == '__main__':
    print(get_point_map_data('test'))

