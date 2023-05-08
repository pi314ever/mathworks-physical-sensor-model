import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from data_util import load_hashmap_data, write_hashmap_data
from utils._paths import find_data

SPLITS = ["test", "train", "valid"]
MODEL_TYPES = ["radial", "tangential", "combined"]

if __name__ == "__main__":
    # Delete hash_to_params
    hash_to_params = load_hashmap_data()
    print(f"Found {len(hash_to_params)} hash_to_params")
    if input("Delete ALL hash_to_params? (y/N) ") == "y":
        write_hashmap_data()
        print("Overwritten hash_to_params.json with {}")
    if input("Delete ALL point map data? (y/N) ") == "y":
        for split in SPLITS:
            for model_type in MODEL_TYPES:
                for filename in find_data(f"{model_type}_point_maps", split):
                    print(f"Removing {filename}")
                    os.remove(filename)
