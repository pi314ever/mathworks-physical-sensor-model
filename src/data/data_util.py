### Utility functions for extracting data from data files
import json
import os
import random
import sys
from hashlib import md5
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from fwr13y.d9m.tensorflow import enable_determinism

# Remove randomness
SEED = 1234
enable_determinism()
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from numpy.typing import NDArray
from utils import TRAIN_VAL_TEST_SPLIT, find_data, get_data_path
from utils.arg_checks import validate_model_type, validate_split
from utils.distortions import distort_radial, distort_tangential
from utils.tensors import read_tensor, write_tensor
from utils.typing import HashDictType, paramType

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
    file_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    params_ds = tf.data.Dataset.from_tensor_slices(params_list)

    OUTPUT_SIGNATURE = (
        tf.TensorSpec(shape=(RESOLUTION, 2), dtype=tf.float32),  # type: ignore
        tf.TensorSpec(shape=(RESOLUTION, 2), dtype=tf.float32),  # type: ignore
        tf.TensorSpec(shape=(n_params,), dtype=tf.float32),  # type: ignore
    )
    ds = tf.data.Dataset.zip((file_ds, params_ds))
    if n_samples is not None:
        ds = ds.take(n_samples)
    ds = (
        ds.shuffle(128)
        .batch(4)
        .prefetch(tf.data.AUTOTUNE)
        .interleave(
            lambda f, p: tf.data.Dataset.from_generator(
                _generator,
                output_signature=OUTPUT_SIGNATURE,
                args=(f, p),
            ).map(
                lambda XY, XYd, params: (process_inputs(XYd, params), XY),
                num_parallel_calls=tf.data.AUTOTUNE,
            ),
            # .prefetch(tf.data.AUTOTUNE),
            num_parallel_calls=tf.data.AUTOTUNE,
            cycle_length=8,
            block_length=1,
            deterministic=True,
        )
    )
    samples = n_samples if n_samples is not None else len(file_paths)
    return ds, samples


def extract_data(file_path, params):
    # print(file_path, params)
    # print(file_path.shape, file_path.dtype)
    # print(params[0])
    # print(params.shape)
    file_path = file_path.decode()
    try:
        data = read_tensor(file_path)
    except:
        print(f"Error reading {file_path}")
        return None
    return data[:, :-2], data[:, -2:], params


def _generator(file_paths, params_list):
    for file_path, params in zip(file_paths, params_list):
        yield extract_data(file_path, params)


def process_inputs(XYd, params):
    input = tf.concat(
        (
            tf.squeeze(XYd),
            tf.squeeze(
                tf.repeat(
                    tf.reshape(params, (-1, 1, params.shape[-1])), XYd.shape[-2], 1
                )
            ),
        ),
        axis=-1,
    )
    return input


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    ds, n = create_dataset(split="valid", model_type="combined", n_params=5)
    for i, batch in tqdm(enumerate(ds), desc="Dataset", total=n):
        if i == 0:
            print(batch[0])
        # plt.figure()
        # plt.scatter(batch[0][:, 0], batch[0][:, 1])
        # plt.title(f"Batch {i}: {batch[0][0, 2:]}")
        # plt.show()
        # plt.close()
