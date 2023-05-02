import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils._paths import find_data
from data_util import write_hashmap_data


DELETE_HASH = True
DELETE_DATA = True

SPLITS = ['test','train','valid']
MODEL_TYPES = ['radial', 'tangential', 'combined']

if __name__ == '__main__':
    # Delete hash_to_params
    if DELETE_HASH and input('Delete ALL hash_to_params? (y/N) ') == 'y':
        write_hashmap_data()
    if DELETE_DATA and input('Delete ALL point map data? (y/N) ') == 'y':
        for split in SPLITS:
            for model_type in MODEL_TYPES:
                for filename in find_data(f'{model_type}_point_maps', split):
                    print(f'Removing {filename}')
                    os.remove(filename)
