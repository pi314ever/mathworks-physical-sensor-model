### Utility functions for extracting data from data files
import json
from typing import TypedDict
import ntpath
import multiprocessing as mp

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
        NDArray, NDArray: Input, Label
    """
    # Load hashmap data
    if split not in ['test', 'train', 'val']:
        raise ValueError(f'Invalid split: {split}')
    hash_to_params = load_hashmap_data()
    # Load point map data
    input = []
    label = []
    results = []

    with mp.Pool(mp.cpu_count() - 1) as P:
        file_paths = []
        num_files = 0
        for file_path in find_data('point_maps', split):
            file_paths.append(file_path)
            results.append(P.apply_async(np.loadtxt, args=(file_path,)))
            num_files += 1
        # Load sizes from first filepath
        first_point_map_data = results[0].get()
        encoding = ntpath.basename(file_paths[0]).split('.')[0]
        params = hash_to_params[encoding]
        num_K, num_P = len(params['K']), len(params['P'])
        N = first_point_map_data.shape[0]
        input = np.ndarray((N * num_files, 2 + num_K + num_P))
        label = np.ndarray((N * num_files, 2))
        for i, r in enumerate(results):
            point_map_data = r.get()
            encoding = ntpath.basename(file_paths[i]).split('.')[0]
            params = hash_to_params[encoding]
            input[N*i:N*(i+1), :2] = point_map_data[:, :2]
            input[N*i:N*(i+1), 2:] = [k for k in params['K']] + [p for p in params['P']]
            label[N*i:N*(i+1), :] = point_map_data[:, 2:]
    return input, label

if __name__ == '__main__':
    print(get_point_map_data('test'))

