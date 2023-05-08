### Generates point mappings for a set of distortion parameters
# Usage:
# ```
# cd src/data
# python generate_point_maps.py
# ```
from __future__ import annotations

import multiprocessing as mp
import os
import sys
from argparse import ArgumentParser

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import tensorflow as tf
from data_util import (
    RESOLUTION,
    get_param_encoding,
    get_param_split,
    load_hashmap_data,
    write_hashmap_data,
)
from tqdm import tqdm
from utils import distort_radial, distort_tangential, get_data_path
from utils.tensors import write_tensor
from utils.typing import distortionType

# ---------------------------------------------------------------------------- #
#                              TUNEABLE PARAMETERS                             #
# ---------------------------------------------------------------------------- #

XY_RANGE = (-1, 1)
PARAM_RANGE = (-0.1, 0.1)
NUM_K = 3
NUM_P = 2

# ---------------------------------------------------------------------------- #
#                            END TUNEABLE PARAMETERS                           #
# ---------------------------------------------------------------------------- #


def generate_random_point_map(distortion: distortionType, params, random):
    # Get random x, y points
    x, y = random.uniform((2, RESOLUTION), *XY_RANGE)
    if distortion == "radial":
        xd, yd = distort_radial(x, y, params)
    elif distortion == "tangential":
        xd, yd = distort_tangential(x, y, params)
    elif distortion == "combined":
        xd, yd = distort_radial(x, y, params[:NUM_K])
        xd, yd = distort_tangential(xd, yd, params[NUM_K:])
    original = tf.squeeze(
        tf.concat(
            (tf.reshape(x, (RESOLUTION, 1)), tf.reshape(y, (RESOLUTION, 1))), axis=1
        )
    )
    distorted = tf.squeeze(
        tf.concat(
            (tf.reshape(xd, (RESOLUTION, 1)), tf.reshape(yd, (RESOLUTION, 1))), axis=1
        )
    )
    return original, distorted


def generate_save(distortion: distortionType, model_path, i):
    seed = int.from_bytes(bytes(f"{i}_{distortion}_{model_path}", "utf-8"), "little")
    random = tf.random.Generator.from_seed(seed)
    if distortion == "radial":
        params = random.uniform((NUM_K,), *PARAM_RANGE)  # type: ignore
    elif distortion == "tangential":
        params = random.uniform((NUM_P,), *PARAM_RANGE)  # type: ignore
    elif distortion == "combined":
        params = random.uniform((NUM_K + NUM_P,), *PARAM_RANGE)  # type: ignore
    params = tuple(float(p) for p in params)
    split = get_param_split(params, distortion)
    encoding = get_param_encoding(params, distortion)
    filename = os.path.join(model_path, split, f"{encoding}.dat")
    out = encoding, dict(params=params, distortion=distortion, split=split)
    if os.path.exists(filename):
        return out
    XY, XYd = generate_random_point_map(distortion, params, random)
    stack = tf.concat((XY, XYd), axis=1)
    write_tensor(filename, stack)
    return out


def main(args):
    # Initialize
    hash_to_params = load_hashmap_data()
    i = 0

    # Generate point maps loop
    try:
        with mp.Pool(args.num_workers) as P:
            results = []
            for i in range(args.num_samples):
                results.append(
                    P.apply_async(
                        generate_save, args=(args.model_type, args.model_path, i)
                    )
                )
            # Evaluate all results
            for r in tqdm(results, desc=f"{args.model_type} point maps"):
                encoding, hash_dict_entry = r.get()
                hash_to_params[encoding] = hash_dict_entry
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
    # Save hash to params dictionary
    write_hashmap_data(hash_to_params)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "model_type", type=str, choices=["combined", "radial", "tangential"]
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=2000,
        help="Number of samples to generate (in total)",
    )
    parser.add_argument("-w", "--num_workers", type=int, default=mp.cpu_count() - 1)
    args = parser.parse_args()

    args.model_path = get_data_path(f"{args.model_type}_point_maps")
    os.makedirs(args.model_path, exist_ok=True)

    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(args.model_path, split), exist_ok=True)
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
