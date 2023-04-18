import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils._paths import find_data
from data_util import write_hashmap_data


DELETE_HASH = True
DELETE_DATA = True

SPLITS = ['test','train','val']

if __name__ == '__main__':
    # Delete hash_to_params
    if DELETE_HASH:
        write_hashmap_data({})
    if DELETE_DATA:
        for split in SPLITS:
            for filename in find_data('point_maps', split):
                print(f'Removing {filename}')
                os.remove(filename)
