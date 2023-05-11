import tensorflow as tf
import numpy as np
from typing import Any


def write_tensor(filename, tensor):
    tf.io.write_file(filename, tf.io.serialize_tensor(tensor))


def read_tensor(filename, dtype=tf.float32) -> Any:
    return tf.io.parse_tensor(tf.io.read_file(filename), dtype)


if __name__ == "__main__":
    tensor = tf.convert_to_tensor(range(1000000))
