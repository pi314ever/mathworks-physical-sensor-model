import argparse
import os
import sys
from typing import Any
import numpy as np
import random
import pickle
import datetime
import time

# Remove TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from fwr13y.d9m.tensorflow import enable_determinism
import tensorflow as tf

# Remove randomness
SEED = 1234
enable_determinism()
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Make utils visible
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from data.data_util import create_dataset, process_inputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a neural network to predict point maps"
    )
    parser.add_argument(
        "model_type",
        help="Model type",
        type=str,
        choices=["combined", "radial", "tangential"],
    )
    parser.add_argument(
        "-s",
        "--layer_size",
        help="Layer size (for all hidden layers)",
        type=int,
        default=64,
    )
    parser.add_argument(
        "-l", "--num_layers", help="Number of hidden layers", type=int, default=4
    )
    parser.add_argument(
        "-r",
        "--regularization",
        help="L2 regularization constant",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "-n",
        "--num_epochs",
        help="Number of epochs to train for",
        type=int,
        default=100,
    )
    parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=32)
    parser.add_argument(
        "-w",
        "--num_workers",
        help="Number of workers for the fitting process",
        type=int,
        default=8,
    )

    args = parser.parse_args()
    args.dir_name = os.path.join(
        "models",
        f"{args.model_type}_l{args.num_layers}_s{args.layer_size}_r{args.regularization}",
    )
    args.saves_dir = os.path.join(args.dir_name, "saves")
    args.now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.model_file = os.path.join(args.dir_name, f"{args.now}.tf")
    args.log_file = os.path.join(args.dir_name, f"{args.now}.log")
    args.hist_file = os.path.join(args.dir_name, f"{args.now}.pkl")
    args.layers = [args.layer_size] * args.num_layers
    args.reg = args.regularization

    if args.model_type == "combined":
        args.num_params = 5
        args.input_size = 7
    elif args.model_type == "radial":
        args.num_params = 3
        args.input_size = 5
    elif args.model_type == "tangential":
        args.num_params = 2
        args.input_size = 4

    os.makedirs(args.dir_name, exist_ok=True)
    os.makedirs(args.saves_dir, exist_ok=True)
    return args


class PointMapNN(tf.keras.Model):
    def __init__(
        self,
        model_type,
        n_params,
        input_size,
        layer_sizes,
        reg,
        activation="relu",
    ):
        super().__init__()
        self.n_params = n_params
        self.model_type = model_type
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        assert layer_sizes != [], "Layer sizes must be non-empty"
        layers = [
            tf.keras.layers.Dense(
                layer_sizes[0],
                activation=activation,
                input_shape=(input_size,),
                activity_regularizer=tf.keras.regularizers.l2(reg) if reg else None,
            )
        ]
        for layer_size in layer_sizes[1:]:
            layers.append(
                tf.keras.layers.Dense(
                    layer_size,
                    activation=activation,
                    activity_regularizer=tf.keras.regularizers.L2(reg) if reg else None,
                )
            )
        self.model = tf.keras.Sequential(
            layers=layers
            + [tf.keras.layers.Dense(2, activation="linear")]
            # + [tf.keras.layers.BatchNormalization()]
        )
        self.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss)

    def call(self, x):
        return self.model(x)

    def train(self, num_epochs, history_file, model_file, batch_size, num_workers):
        # Load datasets
        train_ds, n_train = create_dataset(
            split="train", model_type=self.model_type, n_params=self.n_params
        )
        valid_ds, n_valid = create_dataset(
            split="valid", model_type=self.model_type, n_params=self.n_params
        )

        print(f"Training {self.model_type} model")
        print("Loss:", self.losses)
        print("Optimizer:", self.optimizer)
        print("Metrics:", self.metrics)
        print("Input size:", self.input_size)
        print("Layer sizes:", self.layer_sizes)

        start = time.time()

        hist = self.fit(
            train_ds.repeat(),
            epochs=num_epochs,
            batch_size=batch_size,
            shuffle=True,
            steps_per_epoch=n_train,
            validation_data=valid_ds,
            validation_batch_size=batch_size,
            validation_steps=n_valid,
            workers=num_workers,
            use_multiprocessing=True,
            verbose=2,  # type: ignore
        )

        print(f"Training took {time.time() - start} seconds")

        # Save model and history
        self.save(model_file)
        pickle.dump(hist, open(history_file, "wb"))


class Model:
    model_type: str
    model_files: list[str]

    def predict(self, XY_distorted, K, P):
        raise NotImplementedError()

    def __call__(self, *args, **kwds):
        self.predict(*args, **kwds)


class CombinedPMNN(Model):
    model_type = "combined"

    def __init__(self, model_file):
        self.model_files = [model_file]
        self.model: PointMapNN = tf.keras.models.load_model(model_file)  # type: ignore
        if self.model is None:
            raise ValueError(f"Could not load model from {model_file}")

    def predict(self, XYd, K, P):
        return self.model.predict(self.process_inputs(XYd, K, P))

    def process_inputs(self, XYd, *args):
        params = []
        for p in args:
            params.extend(p)
        return process_inputs(XYd, params)


class SeperateModel(Model):
    model_type = "separate"

    def __init__(self, radial_model_file, tangential_model_file):
        self.model_files = [radial_model_file, tangential_model_file]
        self.radial_model: PointMapNN = tf.keras.models.load_model(radial_model_file)  # type: ignore
        self.tangential_model: PointMapNN = tf.keras.models.load_model(tangential_model_file)  # type: ignore
        if self.radial_model is None:
            raise ValueError(f"Could not load model from {radial_model_file}")
        if self.tangential_model is None:
            raise ValueError(f"Could not load model from {tangential_model_file}")

    def predict(self, XYd, K, P):
        XY = self.tangential_model(process_inputs(XYd, P))
        return self.radial_model(process_inputs(XY, K))


def loss(y_true, y_pred):
    # Average L1 norm loss
    return tf.reduce_mean(tf.reduce_sum(tf.abs((y_true - y_pred)), axis=1))


def main(args):
    sys.stdout = open(args.log_file, "w")
    print(args)
    # Load models

    model = PointMapNN(
        model_type=args.model_type,
        n_params=args.num_params,
        input_size=args.input_size,
        layer_sizes=args.layers,
        reg=args.reg,
    )
    model.train(
        args.num_epochs,
        args.hist_file,
        args.model_file,
        args.batch_size,
        args.num_workers,
    )
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    print(tf.config.list_physical_devices("GPU"))
    args = parse_args()
    print(args)
    main(args)
