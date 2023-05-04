### Utility functions for extracting data from data files
import json
import os
import sys
from typing import List, Union, Optional
from hashlib import md5
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray
from utils import TRAIN_VAL_TEST_SPLIT, find_data, get_data_path
from utils.arg_checks import validate_split, validate_model_type
from utils.distortions import distort_radial, distort_tangential
from utils.typing import HashDictType, paramType
from utils.tensors import read_tensor, write_tensor

# Random data generator
random = np.random.default_rng(10000)

RESOLUTION = 1920 * 1080


def get_param_encoding(params: tuple[float, ...], distortion: str) -> str:
    """
    Generates a string encoding from a set of distortion parameters

    Args:
        params (tuple[float]): Distortion parameters

    Returns:
        str: Filename
    """
    return str(int(md5((str(params) + distortion).encode("utf-8")).hexdigest(), 16))


def encodings_to_params(
    encodings: Union[str, List[str]]
) -> tuple[
    Union[paramType, list[paramType], None], Union[paramType, list[paramType], None]
]:
    """
    Converts a string encoding or a list of string encodings into a set of distortion parameters

    Args:
        encodings (str | List[str]): Encodings for distortion parameters

    Returns:
        tuple[K, P]: K and P either a set of parameters or a list of sets of parameters, depending on the input
    """
    # Load encoding mapping
    with open(get_data_path("hash_to_params.json"), "r") as f:
        try:
            hash_to_params = json.loads(f.read())
        except:
            hash_to_params = {}

    # Case: encodings is a list
    if isinstance(encodings, list):
        K, P = [], []
        for encoding in encodings:
            if encoding not in hash_to_params:
                print(f"Encoding {encoding} not found in hash_to_params.json")
                continue
            K.append(hash_to_params[encoding]["K"])
            P.append(hash_to_params[encoding]["P"])

    # Case: Single encoding
    if encodings not in hash_to_params:
        print(f"Encoding {encodings} not found in hash_to_params.json")
        return None, None
    return hash_to_params[encodings]["K"], hash_to_params[encodings]["P"]


def get_param_split(params: paramType, distortion: str) -> str:
    return np.random.default_rng(
        seed=int(get_param_encoding(params, distortion))
    ).choice(
        ["train", "valid", "test"],
        p=TRAIN_VAL_TEST_SPLIT,
    )


def load_hashmap_data():
    try:
        hash_to_params: dict[str, HashDictType] = json.load(
            open(get_data_path("hash_to_params.json"), "r")
        )
    except:
        hash_to_params = {}
    return hash_to_params


def write_hashmap_data(hash_to_params: Union[dict[str, HashDictType], None] = None):
    if hash_to_params is None:
        hash_to_params = {}
    with open(get_data_path("hash_to_params.json"), "w") as f:
        f.write(json.dumps(hash_to_params))


# Dataset


@validate_split
@validate_model_type
def create_dataset(
    split: str, model_type: str, n_params: int, n_samples: Optional[int] = None
):
    """
    Creates a dataset of point maps using pre-generated point map data
    Returns a tuple of (ds, num_files)
    """
    hash_to_params = load_hashmap_data()
    file_paths = list(find_data(f"{model_type}_point_maps", split))
    params_list = [
        hash_to_params[f.split(os.path.sep)[-1].split(".")[0]]["params"]
        for f in file_paths
    ]

    OUTPUT_SIGNATURE = (
        tf.TensorSpec(shape=(RESOLUTION, 2), dtype=tf.float32),  # type: ignore
        tf.TensorSpec(shape=(RESOLUTION, 2), dtype=tf.float32),  # type: ignore
        tf.TensorSpec(shape=(n_params,), dtype=tf.float32),  # type: ignore
    )
    ds = tf.data.Dataset.from_generator(
        _generator,
        output_signature=OUTPUT_SIGNATURE,
        args=(file_paths, params_list),
    )
    if n_samples is not None:
        ds = ds.take(n_samples)
    ds = (
        ds.batch(16)
        .map(process_inputs, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .unbatch()
    )
    samples = n_samples if n_samples is not None else len(file_paths)
    return ds, samples


def _generator(file_paths, params_list):
    for file_path, params in zip(file_paths, params_list):
        file_path = file_path.decode()
        try:
            data = read_tensor(file_path)
        except:
            print(f"Error reading {file_path}")
            continue
        yield data[:, :-2], data[:, -2:], params


def process_inputs(XY, XYd, params):
    input = tf.concat(
        (XYd, tf.repeat(tf.reshape(params, (-1, 1, params.shape[1])), XYd.shape[1], 1)),
        axis=2,
    )
    output = XY
    return input, output


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ds, n = create_dataset(
        split="valid", model_type="combined", n_params=5
    )
    for i, batch in tqdm(enumerate(ds), desc="Dataset", total=n):
        if i == 0:
            print(batch[0])
        plt.figure()
        plt.scatter(batch[0][:, 0], batch[0][:, 1])
        plt.show()
        plt.close()
