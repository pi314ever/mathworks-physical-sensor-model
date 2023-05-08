import argparse
import json
import os
import sys
from typing import Any
import numpy as np
import random
import pickle
import datetime
import time
import re

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

from data.data_util import create_dataset, process_inputs, RESOLUTION


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
        default=16,
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
    parser.add_argument("-L", "--log", action="store_true", help="Log to file")

    args = parser.parse_args()
    args.dir_name = os.path.join(
        "models",
        f"{args.model_type}_l{args.num_layers}_s{args.layer_size}_r{args.regularization}",
    )
    args.now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.log_file = os.path.join(args.dir_name, f"{args.now}.log")
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
        self.model_type = model_type
        self.n_params = n_params
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.reg = reg
        self.activation = activation
        assert layer_sizes != [], "Layer sizes must be non-empty"
        input = tf.keras.layers.Input(shape=(input_size,))
        x = tf.keras.layers.Dense(
            layer_sizes[0],
            activation=activation,
            activity_regularizer=tf.keras.regularizers.L2(reg) if reg else None,
        )(input)
        for layer_size in layer_sizes[1:]:
            x = tf.keras.layers.Dense(
                layer_size,
                activation=activation,
                activity_regularizer=tf.keras.regularizers.L2(reg) if reg else None,
            )(x)
        out = tf.keras.layers.Dense(2, activation="linear")(x)
        super().__init__(input, out, name=f"{model_type}_point_map_nn")
        self.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss)

    def get_config(self):
        return {
            "model_type": self.model_type,
            "n_params": self.n_params,
            "input_size": self.input_size,
            "layer_sizes": self.layer_sizes,
            "reg": self.reg,
            "activation": self.activation,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # def call(self, x):
    #     return self.model(x)

    def train(self, num_epochs, dir_name, batch_size, num_workers):
        # Save config in dir_name
        json.dump(self.get_config(), open(os.path.join(dir_name, "config.json"), "w"))
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

        # Load from latest checkpoint if available
        latest = tf.train.latest_checkpoint(os.path.join(dir_name, "checkpoints"))
        init_epoch = 0
        if latest:
            init_epoch = int(re.findall(r"cp-([0-9]+)-", latest)[0])
            print(f"Loading model from {latest} at epoch {init_epoch}")
            self.load_weights(latest)
        else:
            print("No checkpoint found, starting from scratch")

        checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                dir_name, "checkpoints", "cp-{epoch:04d}-{val_loss:.4f}.ckpt"
            ),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )
        hist = self.fit(
            train_ds.repeat(),
            epochs=num_epochs,
            batch_size=batch_size,
            shuffle=True,
            initial_epoch=init_epoch,
            steps_per_epoch=n_train,
            validation_data=valid_ds,
            validation_batch_size=batch_size,
            validation_steps=n_valid,
            workers=num_workers,
            use_multiprocessing=True,
            verbose=2 if args.log else 1,  # type: ignore
            callbacks=[checkpoints_callback],
        )
        print(f"Training took {time.time() - start} seconds")

        # Save history
        pickle.dump(hist, open(os.path.join(dir_name, "history.pkl"), "wb"))


class Model:
    model_type: str
    model_dirs: list[str]

    def predict(self, XY_distorted, K, P):
        raise NotImplementedError()

    def __call__(self, *args, **kwds):
        self.predict(*args, **kwds)

    def __repr__(self) -> str:
        return f"{self.model_type} model from {self.model_dirs}"


def load_pmnn_model(model_dir):
    config = json.load(open(os.path.join(model_dir, "config.json"), "r"))
    model = PointMapNN.from_config(config)
    latest = tf.train.latest_checkpoint(os.path.join(model_dir, "checkpoints"))
    model.load_weights(latest)
    return model


class CombinedPMNN(Model):
    model_type = "combined"

    def __init__(self, model_dir):
        self.model_dirs = [model_dir]
        self.model = load_pmnn_model(model_dir)

    def predict(self, XYd, K, P, batch_size=RESOLUTION, **kwargs):
        return self.model.predict(
            self.process_inputs(XYd, K, P), batch_size=batch_size, **kwargs
        )

    def process_inputs(self, XYd, *args):
        XYd = tf.reshape(XYd, (-1, *XYd.shape))
        params = []
        for p in args:
            params.extend(p)
        params = tf.constant(params)
        return tf.squeeze(process_inputs(XYd, params))


class SeparatePMNN(Model):
    model_type = "separate"

    def __init__(self, radial_model_dir, tangential_model_dir):
        self.model_dirs = [radial_model_dir, tangential_model_dir]
        self.radial_model = load_pmnn_model(radial_model_dir)
        self.tangential_model = load_pmnn_model(tangential_model_dir)

    def predict(self, XYd, K, P):
        XYd = tf.reshape(XYd, (-1, *XYd.shape))
        XY = self.tangential_model()
        return self.radial_model(
            tf.squeeze(
                process_inputs(
                    tf.reshape(
                        tf.squeeze(
                            self.tangential_model(process_inputs(XYd, tf.constant(P)))
                        ),
                        (-1, *XYd.shape),
                    ),
                    tf.constant(K),
                )
            )
        )


def loss(y_true, y_pred):
    # Average L1 norm loss
    return tf.reduce_mean(tf.reduce_sum(tf.abs((y_true - y_pred)), axis=1))


def main(args):
    sys.stdout = open(args.log_file, "w") if args.log else sys.stdout
    print(args)
    # Load models
    model = PointMapNN(
        model_type=args.model_type,
        n_params=args.num_params,
        input_size=args.input_size,
        layer_sizes=args.layers,
        reg=args.reg,
    )
    model.summary()
    model.train(
        args.num_epochs,
        args.dir_name,
        args.batch_size,
        args.num_workers,
    )
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    print(tf.config.list_physical_devices("GPU"))
    args = parse_args()
    print(args)
    main(args)
