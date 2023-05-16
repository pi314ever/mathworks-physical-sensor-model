import os
import sys
from argparse import ArgumentParser
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from model import load_model_from_dirs
from pipeline import PointMapModelPipeline
from data.data_util import create_dataset
from utils._paths import get_data_path
from utils.images import read_image, write_image
from typing import Literal


def plot_scatter(
    Y_test,
    Y_pred,
):
    plt.scatter(Y_test[:50, 0], Y_test[:50, 1], label="Ground Truth")
    plt.scatter(Y_pred[:50, 0], Y_pred[:50, 1], label="Prediction")


def plot_errors(
    Y_test,
    Y_pred,
    cutoff=0.1,
):
    # Calculate radial distances
    R = np.sqrt((Y_test[:, 0] - Y_pred[:, 0]) ** 2 + (Y_test[:, 1] - Y_pred[:, 1]) ** 2)
    plt.hist(R[R < cutoff], 200, label=f"{len(R[R < cutoff])} of {len(R)} shown")


def example_image(model, K, P, image: Literal["checkerboard", "stanford"]):
    if image == "checkerboard":
        img = read_image(get_data_path("images", "checkerboard.jpg"))
    elif image == "stanford":
        img = read_image(get_data_path("images", "stanford.png"))
    pmnn = PointMapModelPipeline(model, "nearest")
    distorted_img = pmnn(img, K, P)
    return distorted_img


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "visualize_type",
        type=str,
        choices=["scatter", "errors", "checkerboard", "stanford"],
    )
    parser.add_argument("model_dirs", nargs="+", type=str)
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--cutoff", type=float, default=0.1)
    # parser.add_argument("--title", type=str, default="")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    return args


def main(args):
    from utils.distortions import get_distorted_location

    # Get model
    model = load_model_from_dirs(args.model_dirs)
    # Generate sample data
    params = tf.constant([0.1, 0.03, 0.005, 0, 0])
    K, P = params[:3], params[3:]
    X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    Xd, Yd = get_distorted_location(X, Y, K, P)
    XY = np.stack([Xd, Yd], axis=-1)
    XYd = np.stack([X, Y], axis=-1)
    title = f"{args.visualize_type} {list(params)}"
    if args.visualize_type == "scatter":
        XY_pred = model.predict(XYd, K, P)
        plt.figure()
        plot_scatter(XY_pred, XY)
        plt.title(title)
        if args.show:
            plt.show()
        plt.savefig(args.filename)
    elif args.visualize_type == "errors":
        XY_pred = model.predict(XYd, K, P)
        plt.figure()
        plot_errors(XY_pred, XY)
        plt.title(title)
        if args.show:
            plt.show()
        plt.savefig(args.filename)
    elif args.visualize_type == "checkerboard":
        distorted_img = example_image(model, K, P, "checkerboard")
        if args.show:
            plt.figure()
            plt.imshow(distorted_img)
            plt.show()
        write_image(distorted_img, args.filename)
    elif args.visualize_type == "stanford":
        distorted_img = example_image(model, K, P, "stanford")
        if args.show:
            plt.figure()
            plt.imshow(distorted_img)
            plt.show()
        write_image(distorted_img, args.filename)


if __name__ == "__main__":
    # Visualize data
    args = parse_args()
    print(args)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main(args)
